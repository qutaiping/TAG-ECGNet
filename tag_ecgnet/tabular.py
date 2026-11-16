"""Measurement-only per-type, per-category logistic regression control."""

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from .data import load_rows, check_disjoint
from .engine import dump_json
from .preprocessing import fit_scaler, meta
from .metrics import eligible_from_training, metrics, thresholds_from_val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("configs/labels.json"))
    a = ap.parse_args()
    rows = {
        s: load_rows(a.manifest_dir / (s + ".jsonl"))
        for s in ("train", "val", "test", "external")
        if s in ("train", "val") or (a.manifest_dir / (s + ".jsonl")).exists()
    }
    check_disjoint(rows)
    labels = json.loads(a.labels.read_text(encoding="utf-8"))
    if len(labels) != 23:
        raise ValueError("Expected 23 canonical labels")
    scaler = fit_scaler(rows["train"])
    ds = {
        s: SimpleNamespace(
            rows=rr,
            y=torch.tensor([r["labels"] for r in rr], dtype=torch.float32),
            ym=torch.tensor([r["label_mask"] for r in rr], dtype=torch.float32),
            typ=torch.tensor([int(r["type"] == "holter") for r in rr]),
            metadata=meta(rr, scaler),
        )
        for s, rr in rows.items()
    }
    tr = ds["train"]
    eligible = eligible_from_training(tr)
    fitted = {}
    probs = {s: np.zeros((len(d.rows), 23)) for s, d in ds.items()}
    for typ in range(2):
        for j in range(23):
            sel = (tr.typ.numpy() == typ) & (tr.ym.numpy()[:, j] > 0)
            y = tr.y.numpy()[sel, j]
            classifier = None
            if len(np.unique(y)) > 1:
                classifier = LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=500, random_state=42
                ).fit(tr.metadata.numpy()[sel], y)
            fitted[(typ, j)] = classifier
            for s, d in ds.items():
                mask = d.typ.numpy() == typ
                if mask.any():
                    probs[s][mask, j] = (
                        classifier.predict_proba(d.metadata.numpy()[mask])[:, 1]
                        if classifier is not None
                        else float(y.mean()) if len(y) else 0.0
                    )
    thresholds = thresholds_from_val(ds["val"], probs["val"], eligible)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / "measurement_only.pkl").open("wb") as f:
        pickle.dump(
            dict(models=fitted, scaler=scaler, labels=labels, thresholds=thresholds), f
        )
    result = {"model": "measurement-only logistic regression"}
    for s in ds:
        if s != "train":
            result[s] = metrics(ds[s], probs[s], labels, eligible, thresholds)
    dump_json(a.output_dir / "metrics.json", result)


if __name__ == "__main__":
    main()
