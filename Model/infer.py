"""Full-resolution photographic tone transfer from a trained local checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tone_model.data import image_tensor, read_rgb, shrink
from tone_model.model import ModelConfig, ToneTransferModel
from train import guard_experiment_paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Trusted trained checkpoint from train.py")
    ap.add_argument("--content", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True, help="New PNG path; existing files are refused")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    output = Path(args.output).resolve()
    guard_experiment_paths(output, args.checkpoint, args.content, args.reference)
    if output.suffix.lower() != ".png":
        raise ValueError("Use a .png output for lossless storage")
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("step", 0) < 1:
        raise ValueError("Refusing inference from an untrained checkpoint")
    device = torch.device(args.device)
    model = ToneTransferModel(ModelConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    model.set_local_enabled(checkpoint.get("local_enabled", True))
    content_image = read_rgb(args.content)
    training = checkpoint.get("training_config", {})
    conditioning = image_tensor(shrink(content_image, training.get("long_edge", 256))).unsqueeze(0).to(device)
    reference = image_tensor(shrink(read_rgb(args.reference), training.get("reference_edge", 256))).unsqueeze(0).to(device)
    # Keep the native raster uint8 on CPU; only one render strip is uploaded.
    content = torch.from_numpy(np.array(content_image, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
    pixels = np.empty((content_image.height, content_image.width, 3), dtype=np.uint8)
    with torch.inference_mode():
        plan = model.prepare(conditioning, reference)
        for start, end, block in model.iter_render_rows(content, plan):
            pixels[start:end] = (block[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if pixels.shape[:2] != (content_image.height, content_image.width):
        raise RuntimeError("Rendering changed source dimensions")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)
    print(json.dumps({"output": str(output), "width": content_image.width, "height": content_image.height,
                      "checkpoint_step": checkpoint["step"], "note": "Full-resolution rendering; no output upscaler"}))


if __name__ == "__main__":
    main()
