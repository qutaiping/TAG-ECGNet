"""Train TAG-ECGNet or a CNN baseline on pre-curated fixed manifests."""

import argparse
import hashlib
import json
import random
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import Reports, load_rows, check_disjoint, collate
from .engine import make_model, infer, dump_json, MODEL_NAMES
from .preprocessing import fit_scaler
from .metrics import (
    eligible_from_training,
    metrics,
    thresholds_from_val,
    save_predictions,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("configs/labels.json"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--model", choices=MODEL_NAMES, default="tag_ecgnet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accumulation", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Offline smoke tests; not the published protocol",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume last.pt with the same configuration",
    )
    a = ap.parse_args()
    if min(a.epochs, a.batch_size, a.accumulation) < 1 or a.workers < 0:
        ap.error(
            "epochs, batch-size and accumulation must be positive; workers nonnegative"
        )
    run = a.output_dir
    if run.exists() and any(run.iterdir()) and not a.resume:
        ap.error("output-dir is not empty; use a new directory or --resume")
    run.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    device = torch.device(a.device)
    labels = json.loads(a.labels.read_text(encoding="utf-8"))
    if len(labels) != 23 or len(set(labels)) != 23:
        raise ValueError("Exactly 23 distinct labels are required")
    paths = {
        s: a.manifest_dir / (s + ".jsonl") for s in ("train", "val", "test", "external")
    }
    paths = {s: p for s, p in paths.items() if s in ("train", "val") or p.exists()}
    rows = {s: load_rows(p) for s, p in paths.items()}
    check_disjoint(rows)
    scaler = fit_scaler(rows["train"])
    ds = {s: Reports(rr, scaler, a.image_root, a.cache_dir) for s, rr in rows.items()}
    tr = ds["train"]
    eligible = eligible_from_training(tr)
    if not eligible.any():
        raise ValueError("No category has 20 positives and 20 negatives in training")
    config = {
        "model_name": a.model,
        "seed": a.seed,
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "accumulation": a.accumulation,
        "pretrained": not a.no_pretrained,
        "reports": {s: len(d) for s, d in ds.items()},
        "split_sha256": {
            s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in paths.items()
        },
    }
    saved = None
    if a.resume:
        saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["labels"] != labels:
            raise ValueError("Resume configuration, labels or manifests differ")
        if saved["scaler"] != scaler:
            raise ValueError("Resume scaler differs")
    model = make_model(a.model, pretrained=not a.no_pretrained and not a.resume).to(
        device
    )
    bp = list(model.backbone.parameters())
    if a.model not in ("tag_ecgnet", "tag_ecgnet_image_only"):
        hp = [
            p
            for n, p in model.named_parameters()
            if n.startswith(("backbone.fc.", "backbone.classifier."))
        ]
        ids = {id(p) for p in hp}
        bp = [p for p in bp if id(p) not in ids]
    else:
        ids = {id(p) for p in bp}
        hp = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW(
        [{"params": bp, "lr": 1e-4}, {"params": hp, "lr": 3e-4}], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    amp = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    y, mask, typ = tr.y.numpy(), tr.ym.numpy(), tr.typ.numpy()
    weights = []
    for t in range(2):
        pos = (y[typ == t] * mask[typ == t]).sum(0)
        neg = ((1 - y[typ == t]) * mask[typ == t]).sum(0)
        weights.append(np.clip(neg / np.maximum(pos, 1), 1, 20))
    weights = torch.tensor(np.array(weights), dtype=torch.float32, device=device)
    typeweights = torch.tensor(
        [0.5 / (typ == t).mean() if (typ == t).any() else 0 for t in range(2)],
        device=device,
    )
    generator = torch.Generator().manual_seed(a.seed)
    history = []
    best = -1.0
    bestep = 0
    first = 1
    if saved:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        amp.load_state_dict(saved["amp"])
        history = saved["history"]
        best = saved["best"]
        bestep = saved["best_epoch"]
        first = saved["epoch"] + 1
        generator.set_state(saved["loader_rng"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    loader = DataLoader(
        tr,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.workers,
        collate_fn=collate,
        generator=generator,
        persistent_workers=a.workers > 0,
        pin_memory=device.type == "cuda",
    )
    dump_json(run / "config.json", config)
    dump_json(run / "scaler.json", scaler)
    for epoch in range(first, a.epochs + 1):
        if epoch - bestep > 3 and bestep:
            break
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for step, (images, lengths, t, yy, ym, me) in enumerate(loader):
            images, t, yy, ym, me = [v.to(device) for v in (images, t, yy, ym, me)]
            ctx = (
                torch.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                logits, _ = model(images, lengths, t, me)
                raw = nn.functional.binary_cross_entropy_with_logits(
                    logits, yy, reduction="none"
                )
                raw = raw * (1 + (weights[t] - 1) * yy) * ym
                loss = ((raw.sum(1) / ym.sum(1).clamp_min(1)) * typeweights[t]).mean()
                window = min(
                    a.accumulation,
                    len(loader) - (step // a.accumulation) * a.accumulation,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            amp.scale(loss / window).backward()
            if (step + 1) % a.accumulation == 0 or step + 1 == len(loader):
                amp.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                amp.step(opt)
                amp.update()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        probs, _ = infer(model, ds["val"], device, a.batch_size, a.workers)
        score = metrics(ds["val"], probs, labels, eligible)["selection_score"]
        scheduler.step()
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_score": score}
        )
        print(json.dumps(history[-1]), flush=True)
        base = {
            "format_version": 1,
            "model_name": a.model,
            "model": model.state_dict(),
            "labels": labels,
            "scaler": scaler,
            "eligible": eligible,
            "epoch": epoch,
            "config": config,
        }
        if score > best:
            best = score
            bestep = epoch
            torch.save(base, run / "best.pt")
        torch.save(
            dict(
                base,
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                amp=amp.state_dict(),
                history=history,
                best=best,
                best_epoch=bestep,
                loader_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=(
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
            ),
            run / "last.pt",
        )
        dump_json(run / "history.json", history)
        if epoch - bestep >= 3:
            break
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    probs, _ = infer(model, ds["val"], device, a.batch_size, a.workers)
    thresholds = thresholds_from_val(ds["val"], probs, eligible)
    ck["thresholds"] = thresholds.tolist()
    torch.save(ck, run / "best.pt")
    dump_json(run / "thresholds.json", thresholds.tolist())
    result = {
        "model": "TAG-ECGNet" if a.model == "tag_ecgnet" else a.model,
        "best_epoch": ck["epoch"],
    }
    for s, dataset in ds.items():
        if s == "train":
            continue
        probs, attention = infer(model, dataset, device, a.batch_size, a.workers)
        result[s] = metrics(dataset, probs, labels, eligible, thresholds)
        save_predictions(
            run / (s + "_predictions.jsonl"), dataset, probs, attention, labels
        )
    dump_json(run / "metrics.json", result)


if __name__ == "__main__":
    main()
