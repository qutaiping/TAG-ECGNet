"""Package trusted original experiment weights with their scaler and thresholds."""

import argparse
import json
from pathlib import Path
import torch
from .engine import make_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-run", type=Path, required=True)
    ap.add_argument(
        "--model", choices=["tag_ecgnet", "tag_ecgnet_image_only"], default="tag_ecgnet"
    )
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    ck = torch.load(a.legacy_run / "best.pt", map_location="cpu", weights_only=False)
    model = make_model(a.model, pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    ck.update(format_version=1, model_name=a.model)
    for key, fn in [("scaler", "scaler.json"), ("thresholds", "thresholds.json")]:
        ck[key] = json.loads((a.legacy_run / fn).read_text())
    if "eligible" not in ck or "labels" not in ck:
        raise ValueError("Legacy labels or eligibility missing")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.output)


if __name__ == "__main__":
    main()
