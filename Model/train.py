"""Supervised training of the compact tone model; no downloaded features/weights.

Example (local smoke, not a quality experiment):
  python Model/train.py --manifest /path/Dataset/Photo_Triplets_v1/triplets.json \
    --output /local/experiments/tone_v1 --device cpu --max-steps 2

On Trinity, first stage data/code to a local compute-node scratch directory and
run there. This entry point refuses head nodes and network-mounted experiment IO.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import re
import socket
import shutil
import time

import numpy as np
import torch

from tone_model.data import TripletDataset, load_triplets, read_json
from tone_model.model import ModelConfig, ToneTransferModel
from tone_model.losses import LossConfig, ToneLoss


def _mount_type(path):
    """Read the longest matching mount instead of trusting /scratch naming."""
    resolved = Path(path).resolve()
    matches = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1].replace("\\040", " "))
        if resolved.is_relative_to(mount):
            matches.append((len(str(mount)), fields[2]))
    if not matches:
        raise RuntimeError(f"Cannot determine backing filesystem for {resolved}")
    return max(matches)[1]


def guard_experiment_paths(*paths):
    """Do not turn ordinary Linux/Windows CPU development into cluster checks."""
    hostname = socket.gethostname().split(".")[0].lower()
    if "trinity" not in hostname:
        return
    if not re.fullmatch(r"trinity-\d+-\d+", hostname):
        raise RuntimeError(f"Experiment prohibited on Trinity head node: {hostname}")
    allowed = tuple(map(Path, ("/tmp", "/scratch", "/ssd", "/ssd0", "/ssd1")))
    local_fs = {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "zfs", "overlay"}
    for value in (Path.cwd(), *map(Path, paths)):
        resolved = value.resolve()
        if not any(resolved.is_relative_to(base) for base in allowed):
            raise RuntimeError(f"Stage experiment IO to compute-local /tmp, /scratch or /ssd: {resolved}")
        if _mount_type(resolved) not in local_fs:
            raise RuntimeError(f"Experiment IO is not on a verified local filesystem: {resolved}")
        existing = resolved
        while not existing.exists():
            existing = existing.parent
        minimum_gib = 150 if resolved.is_relative_to(Path("/tmp")) else 100
        free = shutil.disk_usage(existing).free
        if free < minimum_gib * 1024**3:
            raise RuntimeError(f"Insufficient local scratch space: {resolved}; {free / 1024**3:.1f} GiB free, requires {minimum_gib} GiB")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(sampler):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "sampler": sampler.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, sampler):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    sampler.set_state(state["sampler"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG resume is unavailable")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def batch(sample, device):
    return {k: sample[k].unsqueeze(0).to(device) for k in ("content", "reference", "target")}


@torch.no_grad()
def validate(model, criterion, dataset, device, limit):
    model.eval()
    totals = {}
    count = min(limit, len(dataset))
    for index in range(count):
        item = batch(dataset[index], device)
        prediction, aux = model(item["content"], item["reference"], return_aux=True)
        losses = criterion(prediction, item["target"], item["content"], aux)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.) + float(value)
    model.train()
    return {key: value / count for key, value in totals.items()}


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset-root")
    ap.add_argument("--source-manifest", help="Frozen source_images.json; auto-discovered by default")
    ap.add_argument("--cache-dir", help="Optional compute-local lossless low-resolution PNG cache; source files remain unchanged")
    ap.add_argument("--review", action="append", default=[], help="Triplet JSON, repeatable; later overrides earlier")
    ap.add_argument("--high-quality-only", action="store_true", help="Only explicit stars for TRAIN; validation stays fixed")
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs/default.json"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", help="Trusted local checkpoint; restores optimizer, RNG and sampler")
    ap.add_argument("--init-checkpoint", help="Trusted weights for a NEW stage; optimizer/step reset")
    ap.add_argument("--global-only", action="store_true", default=None,
                    help="Disable local branch without changing parameter shapes")
    for flag in ("max-steps", "seed", "long-edge", "reference-edge", "validate-every", "validation-limit", "save-every", "log-every"):
        ap.add_argument("--" + flag, type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--gradient-clip", type=float)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.resume and args.init_checkpoint:
        raise ValueError("Use --resume OR --init-checkpoint")
    out = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    guard_experiment_paths(out, manifest, *([args.cache_dir] if args.cache_dir else []))
    config = read_json(args.config)
    training = dict(config["training"])
    for key in training:
        value = getattr(args, key, None)
        if value is not None:
            training[key] = value
    for key in ("max_steps", "long_edge", "reference_edge", "validate_every", "validation_limit", "save_every", "log_every"):
        if training[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0 or training["gradient_clip"] <= 0:
        raise ValueError("Invalid optimizer settings")
    seed_everything(training["seed"])
    model_config, loss_config = ModelConfig(**config["model"]), LossConfig(**config.get("loss", {}))
    records = load_triplets(manifest, args.dataset_root, args.review, args.source_manifest)
    # Also guard actual resolved data: a local manifest cannot smuggle /data reads.
    data_directories = {row[f"_{role}_path"].parent for row in records for role in ("content", "reference", "target")}
    guard_experiment_paths(*data_directories)
    train_set = TripletDataset(records, "train", training["long_edge"], training["reference_edge"], args.high_quality_only, cache_dir=args.cache_dir)
    val_set = TripletDataset(records, "validation", training["long_edge"], training["reference_edge"], cache_dir=args.cache_dir)
    device = torch.device(args.device)
    model = ToneTransferModel(model_config).to(device)
    model.set_local_enabled(not training["global_only"])
    criterion = ToneLoss(loss_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    sampler = torch.Generator().manual_seed(training["seed"] + 1)
    step = 0
    selected_fingerprints = {"train": train_set.fingerprint(), "validation": val_set.fingerprint()}
    if args.resume or args.init_checkpoint:
        # Checkpoints contain Python RNG state. Load only your own trusted files.
        checkpoint = torch.load(args.resume or args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["model_config"] != asdict(model_config):
            raise ValueError("Checkpoint architecture differs from configuration")
        model.load_state_dict(checkpoint["model"])
        if args.resume:
            if checkpoint["data_fingerprints"] != selected_fingerprints:
                raise ValueError("Selected triplets changed; use --init-checkpoint for a new stage")
            saved_training = checkpoint["training_config"]
            mutable = {"max_steps", "log_every", "save_every", "validate_every", "validation_limit"}
            if any(training[k] != saved_training.get(k) for k in training if k not in mutable):
                raise ValueError("Resume settings changed; use --init-checkpoint for a new stage")
            if checkpoint["loss_config"] != asdict(loss_config):
                raise ValueError("Loss settings changed; use --init-checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
            restore_rng(checkpoint["rng"], sampler)
            step = checkpoint["step"]
    if step >= training["max_steps"]:
        raise ValueError("Checkpoint already reached --max-steps")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.pt").exists() and not args.resume:
        raise FileExistsError("Output already contains a run; select a new output or --resume")
    run = {"model": asdict(model_config), "loss": asdict(loss_config), "training": training,
           "manifest": str(manifest), "reviews": [str(Path(p).resolve()) for p in args.review],
           "high_quality_only": args.high_quality_only, "train_count": len(train_set), "validation_count": len(val_set),
           "cache_dir": str(Path(args.cache_dir).resolve()) if args.cache_dir else None,
           "data_fingerprints": selected_fingerprints, "batch_size": 1,
           "note": "Frozen-source supervised training; unreviewed candidates remain unless rejected. No test tuning."}
    (out / "run_config.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    def save_checkpoint():
        payload = {"format_version": 1, "model": model.state_dict(), "model_config": asdict(model_config),
                   "loss_config": asdict(loss_config), "training_config": training,
                   "optimizer": optimizer.state_dict(), "step": step, "rng": rng_state(sampler),
                   "data_fingerprints": selected_fingerprints, "local_enabled": not training["global_only"]}
        temporary = out / "latest.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, out / "latest.pt")
    model.train()
    started = time.monotonic()
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while step < training["max_steps"]:
            index = int(torch.randint(len(train_set), (), generator=sampler))
            item = batch(train_set[index], device)
            optimizer.zero_grad(set_to_none=True)
            prediction, aux = model(item["content"], item["reference"], return_aux=True)
            losses = criterion(prediction, item["target"], item["content"], aux)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            event = {"step": step}
            if step == 1 or step % training["log_every"] == 0 or step == training["max_steps"]:
                event.update(train={k: float(v.detach()) for k, v in losses.items()},
                             elapsed_seconds=round(time.monotonic() - started, 3))
                raw = aux["preclamp_output"].detach()
                basis = aux["proxy_basis"].detach().reshape(raw.shape[0], -1, aux["proxy_basis"].shape[-1])
                residual = aux["proxy_local"].detach().flatten(2).transpose(1, 2)
                correlation = torch.bmm(basis.transpose(1, 2), residual) / basis.shape[1]
                event["diagnostics"] = {
                    "mean_gamma": float(aux["local_scale"].detach().mean()),
                    "beta_l1_max": float(aux["coefficient_bound"].detach().max()),
                    "proxy_basis_correlation": float(correlation.abs().max()),
                    "preclamp_clip_fraction": float(((raw < 0) | (raw > 1)).float().mean()),
                    "gate_mean": float(aux["gates"].detach().mean()) if aux["gates"].numel() else 0.,
                    "active_local_branches": aux["active_local_branches"]}
            if step % training["validate_every"] == 0 or step == training["max_steps"]:
                event["validation"] = validate(model, criterion, val_set, device, training["validation_limit"])
            if len(event) > 1:
                line = json.dumps(event)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if step % training["save_every"] == 0 or step == training["max_steps"]:
                save_checkpoint()
    return out / "latest.pt"


if __name__ == "__main__":
    main()
