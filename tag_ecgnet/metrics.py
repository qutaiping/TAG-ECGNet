"""Research evaluation protocol, including training-defined eligibility.

macro_auprc is sklearn average precision, not trapezoidal PR integration.
Thresholds must be fitted on validation data only.
"""

import json
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from .preprocessing import choose


def eligible_from_training(ds):
    y = ds.y.numpy()
    m = ds.ym.numpy()
    t = ds.typ.numpy()
    eligible = []
    for typ in range(2):
        sel = t == typ
        p = (y[sel] * m[sel]).sum(0)
        n = ((1 - y[sel]) * m[sel]).sum(0)
        eligible.append((p >= 20) & (n >= 20))
    return np.array(eligible)


def metrics(ds, p, labels, eligible, thresholds=None):
    y = ds.y.numpy()
    mask = ds.ym.numpy().astype(bool)
    typ = ds.typ.numpy()
    out = {}
    for t, name in [(None, "all"), (0, "standard_ecg"), (1, "holter")]:
        sel = np.ones(len(y), bool) if t is None else typ == t
        if not sel.any():
            continue
        yy = y[sel]
        pp = p[sel]
        mm = mask[sel]
        pred = pp >= 0.5 if thresholds is None else pp >= thresholds[typ[sel]]
        per = {}
        aucs = []
        aps = []
        brier = []
        for j, k in enumerate(labels):
            obs = mm[:, j]
            pos = int(yy[obs, j].sum())
            neg = int(obs.sum() - pos)
            use = (
                bool(eligible[:, j].any() if t is None else eligible[t, j])
                and pos >= 5
                and (neg >= 5)
            )
            d = {"positive": pos, "negative": neg, "eligible": use}
            if pos and neg:
                d.update(
                    auroc=float(roc_auc_score(yy[obs, j], pp[obs, j])),
                    auprc=float(average_precision_score(yy[obs, j], pp[obs, j])),
                )
                if use:
                    aucs.append(d["auroc"])
                    aps.append(d["auprc"])
                    brier.append(float(np.mean((yy[obs, j] - pp[obs, j]) ** 2)))
            per[k] = d
        evalmask = mm.copy()
        for i, tt in enumerate(typ[sel]):
            evalmask[i] &= eligible[tt]
        shared = [
            k
            for j, k in enumerate(labels)
            if eligible[:, j].all()
            and per[k]["eligible"]
            and (not any((z in k for z in ["其他", "ST段", "正常"])))
        ]
        out[name] = {
            "n": int(sel.sum()),
            "macro_auroc": float(np.mean(aucs)) if aucs else None,
            "macro_auprc": float(np.mean(aps)) if aps else None,
            "micro_f1": (
                float(f1_score(yy[evalmask], pred[evalmask], zero_division=0))
                if evalmask.any()
                else None
            ),
            "macro_brier": float(np.mean(brier)) if brier else None,
            "eligible_classes": len(aucs),
            "shared_rhythm_macro_auroc": (
                float(np.mean([per[k]["auroc"] for k in shared])) if shared else None
            ),
            "shared_rhythm_macro_auprc": (
                float(np.mean([per[k]["auprc"] for k in shared])) if shared else None
            ),
            "shared_rhythm_classes": shared,
            "per_class": per,
        }
    scores = [
        out[t]["macro_auprc"]
        for t in ["standard_ecg", "holter"]
        if t in out and out[t]["macro_auprc"] is not None
    ]
    out["selection_score"] = float(np.mean(scores)) if scores else 0
    return out


def thresholds_from_val(ds, p, eligible):
    y = ds.y.numpy()
    mask = ds.ym.numpy().astype(bool)
    typ = ds.typ.numpy()
    th = np.full((2, y.shape[1]), 0.5)
    for t in range(2):
        for j in range(y.shape[1]):
            s = (typ == t) & mask[:, j]
            if not eligible[t, j] or y[s, j].sum() < 5 or (1 - y[s, j]).sum() < 5:
                continue
            precision, recall, ts = precision_recall_curve(y[s, j], p[s, j])
            f = (
                2
                * precision[:-1]
                * recall[:-1]
                / np.maximum(precision[:-1] + recall[:-1], 1e-08)
            )
            if len(f):
                th[t, j] = float(ts[np.argmax(f)])
    return th


def save_predictions(path, ds, p, att, labels):
    with open(path, "w") as f:
        for i, r in enumerate(ds.rows):
            f.write(
                json.dumps(
                    {
                        "report_id": r["report_id"],
                        "type": r["type"],
                        "probabilities": dict(zip(labels, p[i].astype(float).tolist())),
                        "attention": att[i].astype(float).tolist(),
                        "selected_images": r.get("_selected_images", choose(r)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
