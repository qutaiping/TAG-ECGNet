"""Original waveform masking and train-only numerical preprocessing."""

import numpy as np
import cv2
import torch
from PIL import Image

cv2.setNumThreads(1)


def choose(r, cap=8):
    xs = sorted(r["images"])
    n = min(len(xs), cap if r["type"] == "holter" else 2)
    ids = np.unique(np.linspace(0, len(xs) - 1, n).round().astype(int))
    return [xs[i] for i in ids]


def clean_image(im, typ):
    a = np.asarray(im.convert("RGB")).copy()
    h, w = a.shape[:2]
    if typ == "holter":
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        dark = (gray < 205).astype("uint8") * 255
        kernel = max(5, round(w / 220))
        kernel += 1 - kernel % 2
        closed = cv2.morphologyEx(
            dark, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8)
        )
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 0.6 * w and bh > 0.1 * h and (bw * bh > 0.07 * w * h):
                boxes.append((x, y, bw, bh))
        if boxes:
            out = np.full_like(a, 255)
            for x, y, bw, bh in boxes:
                row = (gray[y : y + bh, x : x + bw] < 210).mean(1)
                strong = np.where(row > 0.5)[0]
                y0 = y + int(strong[0]) if len(strong) else y + int(0.05 * bh)
                y1 = y + int(strong[-1]) + 1 if len(strong) else y + bh
                if y1 - y0 > 0.08 * h:
                    out[y0:y1, x : x + bw] = a[y0:y1, x : x + bw]
            a = out
        else:
            a[: int(0.12 * h)] = 255
            a[int(0.93 * h) :] = 255
    im = Image.fromarray(a)
    im.thumbnail((768, 448), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (768, 448), "white")
    canvas.paste(im, ((768 - im.width) // 2, (448 - im.height) // 2))
    return canvas


def values(rows):
    v = np.asarray(
        [[np.nan if z is None else z for z in r["values"]] for r in rows],
        dtype=np.float32,
    )
    for j in [6, 7, 8]:
        v[:, j] = (v[:, j] + 180) % 360 - 180
    for j in [15, 17]:
        v[:, j] = np.log1p(np.maximum(v[:, j], 0))
    if v.shape[1] > 19:
        v[:, 19] = np.log1p(np.maximum(v[:, 19], 0))
    return v


def fit_scaler(rows):
    v = values(rows)
    med = np.array(
        [
            np.nanmedian(v[:, i]) if np.isfinite(v[:, i]).any() else 0
            for i in range(v.shape[1])
        ]
    )
    vv = np.where(np.isfinite(v), v, med)
    std = vv.std(0)
    std[std < 1e-05] = 1
    return {
        "mean": vv.mean(0).tolist(),
        "std": std.tolist(),
        "median": med.tolist(),
        "observed_in_training": np.isfinite(v).any(0).tolist(),
    }


def meta(rows, scaler):
    v = values(rows)
    missing = ~np.isfinite(v)
    missing |= ~np.asarray(
        scaler.get("observed_in_training", [True] * v.shape[1]), dtype=bool
    )[None, :]
    v = np.where(missing, np.asarray(scaler["median"]), v)
    v = np.clip((v - np.asarray(scaler["mean"])) / np.asarray(scaler["std"]), -8, 8)
    v[missing] = 0
    return torch.tensor(
        np.concatenate([v, (~missing).astype("float32")], 1), dtype=torch.float32
    )
