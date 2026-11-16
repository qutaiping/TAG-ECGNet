"""CPU tests use random tensors and synthetic manifests; no clinical data."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tag_ecgnet.model import TAGECGNet
from tag_ecgnet.data import load_rows, check_disjoint
from tag_ecgnet.engine import load_checkpoint
from tag_ecgnet.preprocessing import fit_scaler, meta, choose
from tag_ecgnet.metrics import (
    metrics,
    thresholds_from_val,
    eligible_from_training,
    save_predictions,
)

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = TAGECGNet(pretrained=False)
        cls.x = torch.randn(3, 3, 96, 160)
        cls.lengths = torch.tensor([1, 2])
        cls.typ = torch.tensor([0, 1])
        cls.metadata = torch.randn(2, 40)
        cls.metadata[:, 20:] = 1

    def test_parameter_count_and_backward(self):
        m = self.model.train()
        m.zero_grad(set_to_none=True)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 4408308)
        z, attention = m(self.x, self.lengths, self.typ, self.metadata)
        self.assertEqual(tuple(z.shape), (2, 23))
        self.assertTrue(torch.allclose(attention.sum(1), torch.ones(2)))
        torch.nn.functional.binary_cross_entropy_with_logits(
            z, torch.ones_like(z)
        ).backward()
        self.assertTrue(
            all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
        )

    def test_missing_metadata_and_permutation(self):
        m = self.model.eval()
        with torch.no_grad():
            z, _, _ = m.pool(m.backbone(self.x), self.lengths, self.typ)
            actual, _ = m(self.x, self.lengths, self.typ, torch.zeros(2, 40))
            torch.testing.assert_close(actual, m.image_head(z))
            first, _ = m(self.x, self.lengths, self.typ, self.metadata)
            second, _ = m(self.x[[0, 2, 1]], self.lengths, self.typ, self.metadata)
            torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
            repeated, _ = m(self.x, self.lengths, self.typ, self.metadata)
            self.assertTrue(torch.equal(first, repeated))

    def test_frozen_bn_and_checkpoint(self):
        m = self.model.train()
        self.assertTrue(
            all(
                not x.training
                for x in m.modules()
                if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)
            )
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.pt"
            torch.save(
                {
                    "format_version": 1,
                    "model_name": "tag_ecgnet",
                    "model": m.state_dict(),
                },
                p,
            )
            loaded, _ = load_checkpoint(p, torch.device("cpu"))
            with torch.no_grad():
                expected, _ = m.eval()(self.x, self.lengths, self.typ, self.metadata)
                actual, _ = loaded(self.x, self.lengths, self.typ, self.metadata)
            torch.testing.assert_close(actual, expected)


class ProtocolTests(unittest.TestCase):
    def test_prediction_serialization(self):
        ds = SimpleNamespace(
            rows=[dict(report_id="SYNTHETIC", type="holter", images=["synthetic.png"])]
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "predictions.jsonl"
            save_predictions(
                p,
                ds,
                np.full((1, 23), 0.5),
                np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]]),
                [str(i) for i in range(23)],
            )
            row = json.loads(p.read_text())
            self.assertEqual(len(row["probabilities"]), 23)
            self.assertEqual(row["selected_images"], ["synthetic.png"])

    def test_train_only_missing_field_scaling(self):
        a = {"values": [None] * 20}
        a["values"][0] = 60
        b = {"values": [None] * 20}
        b["values"][0] = 80
        scaler = fit_scaler([a, b])
        external = {"values": [None] * 20}
        external["values"][1] = 999
        numeric = meta([external], scaler)
        self.assertEqual(numeric[0, 1].item(), 0)
        self.assertEqual(numeric[0, 21].item(), 0)
        self.assertEqual(scaler["mean"][0], 70)

    def test_masked_metrics_and_validation_thresholds(self):
        y = np.array([[0], [1]] * 20, dtype="float32")
        ds = SimpleNamespace(
            y=torch.tensor(y),
            ym=torch.ones(40, 1),
            typ=torch.zeros(40, dtype=torch.long),
        )
        eligible = eligible_from_training(ds)
        p = y * 0.8 + 0.1
        result = metrics(ds, p, ["synthetic"], eligible)
        self.assertEqual(result["standard_ecg"]["macro_auroc"], 1)
        self.assertEqual(result["standard_ecg"]["macro_auprc"], 1)
        thresholds = thresholds_from_val(ds, p, eligible)
        self.assertEqual(thresholds.shape, (2, 1))
        self.assertAlmostEqual(thresholds[0, 0], 0.9, places=5)

    def test_manifest_validation_and_split_overlap(self):
        row = dict(
            report_id="synthetic",
            type="holter",
            images=["wave.png"],
            labels=[0] * 23,
            label_mask=[1] * 23,
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text(json.dumps(row))
            rows = load_rows(p)
            with self.assertRaises(ValueError):
                check_disjoint({"train": rows, "test": rows})
            row["images"] = ["../outside.png"]
            p.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                load_rows(p)

    def test_type_dependent_image_caps(self):
        row = dict(type="holter", images=[f"{i:02}.png" for i in range(12)])
        self.assertEqual(len(choose(row)), 8)
        row["type"] = "standard_ecg"
        self.assertEqual(len(choose(row)), 2)


if __name__ == "__main__":
    unittest.main()
