# Release validation

Validation date: 2026-11-16. The checks used the existing Linux training runtime:
Python 3.11.5, PyTorch 2.1.1+cu121, torchvision 0.16.1+cu121, NumPy 1.25.2,
Pillow 10.2.0, scikit-learn 1.6.1 and OpenCV 4.8.0.

Completed checks:

- Eight CPU unit tests passed: parameter count/backpropagation, attention
  normalization, frozen BN, missing-metadata fallback, report-segment permutation
  invariance, checkpoint loading, train-only numerical preprocessing, masked
  metrics, validation thresholds, prediction serialization, manifest validation,
  report overlap detection and examination-dependent image caps.
- With identical weights and inputs, the released TAG-ECGNet produced exactly
  equal logits and attention weights to the original final model implementation.
- All five CNN references and the image-only TAG-ECGNet variant passed forward
  checks with finite 23-category logits and normalized report-image weights.
- A synthetic dataset with 80 training reports and 20 reports in each validation,
  test and external split ran through one CPU training epoch, checkpoint resume,
  separate evaluation, unlabeled prediction and measurement-only regression.
- The original trained fusion checkpoint was converted successfully with strict
  parameter loading. No real checkpoint or derived numerical scaler is included
  in this release.
- The Python wheel built successfully, installed into an isolated target
  directory and imported from outside the source tree.
- All source files were formatted with Black 24.10.0; formatting preserved their
  Python abstract syntax trees. The release was checked for private host paths,
  raw images, individual clinical records and checkpoint files.

The original multi-epoch clinical experiment was not repeated after packaging.
The recorded benchmark numbers are historical results from that experiment.
These checks verify software behavior and packaging, not clinical validity or
bitwise reproducibility across hardware and library versions. Installation was
tested against the existing runtime; a fresh full dependency installation was
not repeated.
