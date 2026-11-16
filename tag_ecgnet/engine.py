"""Portable inference and self-contained checkpoints."""

import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import collate
from .model import TAGECGNet
from .baselines import ImageOnlyBaseline, ARCHS

MODEL_NAMES = ["tag_ecgnet", "tag_ecgnet_image_only", *ARCHS]


def make_model(name, pretrained=True):
    if name in ("tag_ecgnet", "tag_ecgnet_image_only"):
        return TAGECGNet(
            "fusion" if name == "tag_ecgnet" else "typeaware", 23, 40, pretrained
        )
    if name not in ARCHS:
        raise ValueError(f"Unknown model: {name}")
    return ImageOnlyBaseline(name, 23, pretrained)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


@torch.no_grad()
def infer(model, ds, device, batch_size=2, workers=0):
    model.eval()
    probs, attention = [], []
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    for images, lengths, typ, _, _, metadata in loader:
        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with ctx:
            logits, weights = model(
                images.to(device), lengths, typ.to(device), metadata.to(device)
            )
        probs.append(logits.float().sigmoid().cpu().numpy())
        attention.append(weights.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(attention)


def load_checkpoint(path, device):
    # Checkpoints contain optimizer/NumPy state and must come from a trusted source.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format_version") != 1:
        raise ValueError(
            "Legacy checkpoint: use python -m tag_ecgnet.convert_checkpoint first"
        )
    model = make_model(ck["model_name"], pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    return model, ck
