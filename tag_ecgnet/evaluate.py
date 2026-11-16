"""Evaluate a checkpoint or predict unlabeled reports with its frozen preprocessing."""

import argparse
from pathlib import Path
import numpy as np
import torch
from .data import Reports, load_rows
from .engine import load_checkpoint, infer, dump_json
from .metrics import metrics, save_predictions


def main(predict_only=False):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(a.device)
    model, ck = load_checkpoint(a.checkpoint, device)
    rows = load_rows(a.manifest, require_labels=not predict_only)
    ds = Reports(rows, ck["scaler"], a.image_root, a.cache_dir)
    probs, attention = infer(model, ds, device, a.batch_size, a.workers)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(
        a.output_dir / "predictions.jsonl", ds, probs, attention, ck["labels"]
    )
    thresholds = np.array(ck["thresholds"])
    if thresholds.shape != (2, 23):
        raise ValueError("Expected frozen thresholds with shape (2,23)")
    predicted = probs >= thresholds[ds.typ.numpy()]
    dump_json(
        a.output_dir / "predicted_labels.json",
        [
            {
                "report_id": r["report_id"],
                "labels": [k for k, flag in zip(ck["labels"], flags) if flag],
            }
            for r, flags in zip(rows, predicted)
        ],
    )
    if not predict_only:
        dump_json(
            a.output_dir / "metrics.json",
            metrics(ds, probs, ck["labels"], np.asarray(ck["eligible"]), thresholds),
        )


if __name__ == "__main__":
    main()
