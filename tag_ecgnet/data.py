"""Validated report manifests and waveform image loading."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import choose, clean_image, meta


def load_rows(path, require_labels=True):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        rid = row["report_id"]
        if not isinstance(rid, str) or rid in seen:
            raise ValueError("report_id must be a unique string within a manifest")
        seen.add(rid)
        if row["type"] not in ("standard_ecg", "holter"):
            raise ValueError("type must be standard_ecg or holter")
        if not isinstance(row["images"], list) or not row["images"]:
            raise ValueError("Each report requires at least one waveform image")
        for image in row["images"]:
            p = Path(image)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(
                    "Image paths must be relative to --image-root, without .."
                )
        row.setdefault("values", [None] * 20)
        if len(row["values"]) != 20:
            raise ValueError("values must contain the canonical 20 measurements")
        if any(
            v is not None and (not isinstance(v, (int, float)) or not np.isfinite(v))
            for v in row["values"]
        ):
            raise ValueError("Use finite numeric measurements or JSON null")
        for key in ("labels", "label_mask"):
            if key not in row:
                if require_labels:
                    raise ValueError(f"Training/evaluation requires {key}")
                row[key] = [0] * 23
            if len(row[key]) != 23 or any(v not in (0, 1) for v in row[key]):
                raise ValueError(f"{key} must contain 23 binary entries")
    return rows


def check_disjoint(splits):
    seen = set()
    for name, rows in splits.items():
        ids = {r["report_id"] for r in rows}
        if seen & ids:
            raise ValueError(f"Report IDs overlap across splits at {name}")
        seen |= ids


class Reports(Dataset):
    def __init__(self, rows, scaler, image_root, cache_dir=None):
        self.rows = [dict(r, _selected_images=choose(r)) for r in rows]
        self.metadata = meta(self.rows, scaler)
        self.typ = torch.tensor([int(r["type"] == "holter") for r in self.rows])
        self.y = torch.tensor([r["labels"] for r in self.rows], dtype=torch.float32)
        self.ym = torch.tensor(
            [r["label_mask"] for r in self.rows], dtype=torch.float32
        )
        self.image_root = Path(image_root).resolve()
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row, tensors = self.rows[i], []
        for rel in row["_selected_images"]:
            source = (self.image_root / rel).resolve()
            if not source.is_relative_to(self.image_root):
                raise ValueError("Image resolves outside --image-root")
            stat = source.stat()
            key = hashlib.sha256(
                f"waveform-v1:{row['type']}:{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()
            cached = self.cache / (key + ".png") if self.cache else None
            if cached and cached.exists():
                with Image.open(cached) as im:
                    a = np.array(im.convert("RGB"))
            else:
                with Image.open(source) as im:
                    cleaned = clean_image(im, row["type"])
                a = np.asarray(cleaned)
                if cached:
                    tmp = cached.with_suffix(f".{os.getpid()}.tmp")
                    cleaned.save(tmp, format="PNG")
                    os.replace(tmp, cached)
            a = a.astype("float32") / 255
            a = (a - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array(
                [0.229, 0.224, 0.225], dtype="float32"
            )
            tensors.append(torch.from_numpy(a.transpose(2, 0, 1)))
        return (
            torch.stack(tensors),
            self.typ[i],
            self.y[i],
            self.ym[i],
            self.metadata[i],
        )


def collate(batch):
    images, typ, y, mask, metadata = zip(*batch)
    return (
        torch.cat(images),
        torch.tensor([len(x) for x in images]),
        torch.stack(typ),
        torch.stack(y),
        torch.stack(mask),
        torch.stack(metadata),
    )
