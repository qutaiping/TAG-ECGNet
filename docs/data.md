# Data and evaluation protocol

The public interface is a pre-curated report manifest, not a raw-PDF parser.
Use `report_id` as a report identifier. Reports must not overlap across split
files. This check does not verify patient independence: obtain validated patient
linkage before making that claim. Keep image paths relative to the image root.

The 20 measurements must occur in this order:

| Index (zero based) | Field |
|---:|---|
| 0 | hr_bpm |
| 1 | pr_ms |
| 2 | p_ms |
| 3 | qrs_ms |
| 4 | qt_ms |
| 5 | qtc_ms |
| 6 | p_axis_deg |
| 7 | qrs_axis_deg |
| 8 | t_axis_deg |
| 9 | rv5_mv |
| 10 | sv1_mv |
| 11 | rv5_sv1_sum_mv |
| 12 | hr_mean |
| 13 | hr_min |
| 14 | hr_max |
| 15 | total_beats |
| 16 | sdnn |
| 17 | duration_hours |
| 18 | qrs_t_angle_deg |
| 19 | analyzed_duration_hours |

Use milliseconds for intervals and SDNN, degrees for axes/angle, millivolts for
voltages, beats per minute for heart rates, hours for durations and counts for
total beats. Missing fields must be null, not zero. Circular normalization is
applied to the three axes; log1p is applied to total beats and both duration
fields. Medians and population standard deviations are fitted on training rows.
Fields never observed in training remain masked for all other cohorts.

The existing dataset masks exclude the known unreliable Holter normal category
and unreported morphology categories. Pass the original curated masks unchanged.
The loader checks shape and binary values but cannot infer correct clinical
observability or repair erroneous label mappings. Never silently treat an
unobserved diagnosis as an observed negative.

Image selection follows sorted image paths and uniformly spaced indices: up
to two standard ECG images and eight Holter images. These are representative
report segments, not continuous Holter recordings. Images are not digitized to
voltage/time signals. Waveform grids are detected and outside content is masked
for Holter; standard ECG preprocessing preserves the supplied waveform image.
This masking is heuristic and does not establish that all embedded text has
been removed. There is no report-text encoder.

Training eligibility is defined per examination type and label by at least 20
positive and 20 negative observed training reports. Macro metrics additionally
require five positives and five negatives in the evaluation subset. AUPRC is
sklearn average precision. Micro F1 pools training-eligible observed pairs;
validation thresholds maximize per-type, per-label F1 when there are at least
five positives and five negatives, otherwise defaulting to 0.5. Macro Brier
averages the squared probability error over macro-eligible labels.

The `all` group retains the original exploratory combined-report calculation;
primary comparisons should use the separately reported examination-type groups.
Shared-rhythm summaries retain the original canonical Chinese-label filtering.
