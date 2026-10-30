"""Frozen-triplet loading; paths are relative to the manifest, never the cwd.

No source-image files are modified. Human review is a selection layer only.
Training uses batch size one so landscape and portrait images retain their aspect.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_inside(base: Path, value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("An image path is missing")
    path = (base / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes dataset root: {value}")
    return path


def load_triplets(manifest, dataset_root=None, reviews: Iterable[str] = (), source_manifest=None):
    """Validate both source roles across ALL frozen splits before filtering.

    Diagnostic pairs may reuse train images, but cannot be trained/evaluated as a
    held-out split. If available, the original source manifest additionally checks
    duplicate-group and SHA identities; no new train/val partition is generated.
    Later review files override earlier decisions for the same known pair ID.
    """
    manifest = Path(manifest).resolve()
    root = Path(dataset_root).resolve() if dataset_root else manifest.parent.parent.resolve()
    records = read_json(manifest)["triplets"]
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest contains no triplets")
    candidate = manifest.parent.parent / "Photo_Pairs_v1" / "source_images.json"
    source_path = Path(source_manifest).resolve() if source_manifest else candidate
    sources = {}
    if source_path.exists():
        sources = {r["key"]: r for r in read_json(source_path)["images"]}
    elif source_manifest:
        raise FileNotFoundError(source_path)
    else:
        warnings.warn("Frozen source_images.json was not found: only source key/path split isolation is checked; duplicate-group/SHA isolation is unavailable.", RuntimeWarning)
    seen_ids, identities, resolved = set(), {}, []
    for original in records:
        row = dict(original)
        pair_id = row.get("pair_id")
        split = row.get("split")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen_ids:
            raise ValueError(f"Missing or duplicate pair_id: {pair_id!r}")
        if split not in {"train", "validation", "test", "diagnostic"}:
            raise ValueError(f"Unknown frozen split {split!r} for {pair_id}")
        seen_ids.add(pair_id)
        for role in ("content", "reference", "target"):
            row[f"_{role}_path"] = resolve_inside(manifest.parent, row.get(f"{role}_path"), root)
        for role in ("content", "reference"):
            if split == "diagnostic":
                continue
            key = row.get(f"{role}_key")
            tokens = [("path", str(row[f"_{role}_path"]).casefold())]
            if key:
                tokens.append(("key", key))
            if sources:
                if key not in sources:
                    raise ValueError(f"Source key {key!r} absent from frozen source manifest")
                src = sources[key]
                if src["split"] != split:
                    raise ValueError(f"Source split mismatch in {pair_id}: {key}")
                if src.get("dataset_relative_path"):
                    expected_path = resolve_inside(source_path.parent, src["dataset_relative_path"], root)
                elif src.get("path"):
                    expected_path = resolve_inside(root / "Original Data", src["path"], root)
                else:
                    raise ValueError(f"Frozen source has no path for {key}")
                if expected_path != row[f"_{role}_path"]:
                    raise ValueError(f"Source key/path mismatch in {pair_id}: {key}")
                for field in ("duplicate_group", "sha256"):
                    if src.get(field):
                        tokens.append((field, src[field]))
            for token in tokens:
                previous = identities.setdefault(token, split)
                if previous != split:
                    raise ValueError(f"Source leakage across {previous}/{split}: {token}")
        resolved.append(row)
    decisions = {}
    for review_path in reviews:
        review = read_json(review_path)
        if review.get("collection") != "nju-triplets-60000-v1":
            raise ValueError(f"Wrong review collection in {review_path}; expected triplet review")
        for decision in review.get("reviewed", []):
            pair_id = decision.get("pair_id")
            if pair_id not in seen_ids:
                raise ValueError(f"Review references unknown pair_id: {pair_id!r}")
            value = decision.get("decision")
            if value not in {"reject", "high_quality", "keep"}:
                raise ValueError(f"Unknown review decision: {value!r}")
            decisions[pair_id] = value
    for row in resolved:
        row["_decision"] = decisions.get(row["pair_id"], "unreviewed")
    return resolved


def read_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def shrink(image: Image.Image, long_edge: int):
    if long_edge <= 0:
        return image
    scale = min(1.0, long_edge / max(image.size))
    size = tuple(max(1, round(v * scale)) for v in image.size)
    return image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image


def image_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


class TripletDataset(Dataset):
    def __init__(self, records, split="train", long_edge=256, reference_edge=256,
                 high_quality_only=False, verify_files=True, cache_dir=None):
        if split == "diagnostic":
            raise ValueError("Diagnostic pairs are not a train or held-out evaluation split")
        self.records = [r for r in records if r["split"] == split and r["_decision"] != "reject"
                        and (not high_quality_only or r["_decision"] == "high_quality")]
        if not self.records:
            raise ValueError(f"No usable {split} triplets after selection")
        self.long_edge, self.reference_edge = long_edge, reference_edge
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if any(not r.get("generation", {}).get("max_image_edge") for r in self.records):
            warnings.warn("Some generic triplets lack generation.max_image_edge: content is resized to target dimensions as an explicit fallback, not an exact reproduction of generator downsampling.", RuntimeWarning)
        if verify_files:
            for row in self.records:
                for role in ("content", "reference", "target"):
                    if not row[f"_{role}_path"].is_file():
                        raise FileNotFoundError(row[f"_{role}_path"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        cached = None
        if self.cache_dir:
            # Include all three source stat signatures, including target, so an
            # alignment/target change cannot silently reuse stale content pixels.
            signatures = []
            for role in ("content", "reference", "target"):
                path = row[f"_{role}_path"]
                stat = path.stat()
                signatures.append((role, str(path), stat.st_size, stat.st_mtime_ns))
            key = hashlib.sha256(json.dumps({"version": 2, "sources": signatures,
                "generation": row.get("generation", {}), "long_edge": self.long_edge,
                "reference_edge": self.reference_edge}, sort_keys=True).encode()).hexdigest()
            cached = {role: self.cache_dir / key[:2] / f"{key}_{role}.png"
                      for role in ("content", "reference", "target")}
            if all(path.is_file() for path in cached.values()):
                return {**{role: image_tensor(read_rgb(path)) for role, path in cached.items()}, "pair_id": row["pair_id"]}
        target = read_rgb(row["_target_path"])
        content = read_rgb(row["_content_path"])
        edge = row.get("generation", {}).get("max_image_edge")
        if edge is not None:
            if not isinstance(edge, int) or isinstance(edge, bool) or edge <= 0:
                raise ValueError(f"Invalid generation.max_image_edge in {row['pair_id']}")
            # Match the generation backend's Pillow thumbnail path, including
            # thumbnail's default reducing_gap=2.0 (resize has a different default).
            content.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            if content.size != target.size:
                raise ValueError(f"Generated target raster mismatch for {row['pair_id']}: expected {content.size}, got {target.size}")
        elif content.size != target.size:
            # Explicit generic-manifest fallback; production records carry edge.
            content = content.resize(target.size, Image.Resampling.LANCZOS)
        target = shrink(target, self.long_edge)
        content = content.resize(target.size, Image.Resampling.LANCZOS)
        reference = shrink(read_rgb(row["_reference_path"]), self.reference_edge)
        if cached:
            for role, image in (("content", content), ("reference", reference), ("target", target)):
                path = cached[role]
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                image.save(temporary, format="PNG")
                os.replace(temporary, path)
        return {"content": image_tensor(content), "reference": image_tensor(reference),
                "target": image_tensor(target), "pair_id": row["pair_id"]}

    def fingerprint(self):
        """Bind resumes to selected IDs AND original source/target path mapping."""
        rows = [{k: r.get(k) for k in ("pair_id", "split", "content_path", "reference_path", "target_path")}
                for r in self.records]
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
