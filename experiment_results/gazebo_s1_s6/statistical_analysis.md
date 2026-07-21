# PETSE Statistical Analysis (TII revision)

Source: `/home/jim/ros2_motion_planning_tutorials/experiment_results/gazebo_s1_s6/results.jsonl` — 2581 trials after post-processing, baseline subset used below.


# S1+S3+S4+S5, baseline only

## Baseline comparison with 95% CIs (S1+S3+S4+S5, baseline only)

| Method | N | Valid | TP | FP | TN | FN | INFRA | Recall [95% CI] | FPR [95% CI] | F1 [95% CI] | Violation Rate [95% CI] | Margin-probe rej. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No Guard | 300 | 256 | 0 | 0 | 59 | 197 | 4 | 0.0% [0.0%, 1.9%] | 0.0% [0.0%, 6.1%] | 0.000 [0.000, 0.000] | 66.7% [61.1%, 71.8%] | 0/38 |
| SELP | 300 | 259 | 20 | 0 | 60 | 179 | 1 | 10.1% [6.6%, 15.0%] | 0.0% [0.0%, 6.0%] | 0.183 [0.115, 0.249] | 58.6% [52.9%, 64.0%] | 0/38 |
| CBF | 300 | 260 | 23 | 0 | 60 | 177 | 0 | 11.5% [7.8%, 16.7%] | 0.0% [0.0%, 6.0%] | 0.206 [0.137, 0.276] | 56.0% [50.3%, 61.5%] | 20/40 |
| CBF-Adaptive | 300 | 260 | 84 | 0 | 60 | 116 | 0 | 42.0% [35.4%, 48.9%] | 0.0% [0.0%, 6.0%] | 0.592 [0.520, 0.658] | 34.0% [28.9%, 39.5%] | 40/40 |
| SSM | 300 | 260 | 22 | 0 | 60 | 178 | 0 | 11.0% [7.4%, 16.1%] | 0.0% [0.0%, 6.0%] | 0.198 [0.130, 0.267] | 56.7% [51.0%, 62.2%] | 40/40 |
| RoboGuard | 300 | 260 | 20 | 0 | 60 | 180 | 0 | 10.0% [6.6%, 14.9%] | 0.0% [0.0%, 6.0%] | 0.182 [0.113, 0.251] | 57.0% [51.3%, 62.5%] | 0/40 |
| PETSE | 300 | 260 | 200 | 0 | 60 | 0 | 0 | 100.0% [98.1%, 100.0%] | 0.0% [0.0%, 6.0%] | 1.000 [1.000, 1.000] | 0.0% [0.0%, 1.3%] (0/300; rule-of-3 ≤1.00%) | 40/40 |

Recall = TP/(TP+FN) on unsafe trials. FPR = FP/(FP+TN) on safe trials, EXCLUDING within-margin probes (near/mid_boundary: goals outside the zone but inside the designed margin — rejection is intentional conservatism, reported separately as margin-probe rejections). Violation Rate = physically-entered-zone / all valid trials (probes included). CIs: Wilson score (proportions), bootstrap 2000 resamples (F1).

## Margin-probe rejections by probe distance (designed conservatism, not FP)

Probes sit outside the zone but inside PETSE's 0.562 m margin: near_boundary = 0.15 m, mid_boundary = 0.45 m from the boundary. A method rejects a probe iff the probe distance is inside its own margin — the staircase below is the margin formula acting as designed.

| Method | Margin (m) | near_boundary (0.15 m) | mid_boundary (0.45 m) |
|---|---|---|---|
| No Guard | 0 | 0/19 | 0/19 |
| SELP | 0 | 0/19 | 0/19 |
| CBF | 0.30 | 20/20 | 0/20 |
| CBF-Adaptive | 0.562 | 20/20 | 20/20 |
| SSM | 0.575 | 20/20 | 20/20 |
| RoboGuard | 0* | 0/20 | 0/20 |
| PETSE | 0.562 | 20/20 | 20/20 |

## Exact McNemar tests — detection (PETSE vs baseline, unsafe trials)

Success = unsafe trial correctly rejected (TP). Pairs matched on (scenario, intensity, seed); INFRA pairs dropped.

| Baseline | Pairs | PETSE only ✓ | Baseline only ✓ | McNemar p | Holm-adj. p | Fisher p (unpaired) |
|---|---|---|---|---|---|---|
| No Guard | 197 | 197 | 0 | <0.001 | <0.001 | <0.001 |
| SELP | 199 | 179 | 0 | <0.001 | <0.001 | <0.001 |
| CBF | 200 | 177 | 0 | <0.001 | <0.001 | <0.001 |
| CBF-Adaptive | 200 | 116 | 0 | <0.001 | <0.001 | <0.001 |
| SSM | 200 | 178 | 0 | <0.001 | <0.001 | <0.001 |
| RoboGuard | 200 | 180 | 0 | <0.001 | <0.001 | <0.001 |

## Exact McNemar tests — physical safety (PETSE vs baseline, all valid trials)

Success = no physical zone violation during the trial.

| Baseline | Pairs | PETSE only ✓ | Baseline only ✓ | McNemar p | Holm-adj. p | Fisher p (unpaired) |
|---|---|---|---|---|---|---|
| No Guard | 294 | 196 | 0 | <0.001 | <0.001 | <0.001 |
| SELP | 297 | 174 | 0 | <0.001 | <0.001 | <0.001 |
| CBF | 300 | 168 | 0 | <0.001 | <0.001 | <0.001 |
| CBF-Adaptive | 300 | 102 | 0 | <0.001 | <0.001 | <0.001 |
| SSM | 300 | 170 | 0 | <0.001 | <0.001 | <0.001 |
| RoboGuard | 300 | 171 | 0 | <0.001 | <0.001 | <0.001 |

## Per-seed variance (across random seeds)

| Method | Seeds | Recall mean ± SD | Recall [min, max] | VR mean ± SD | VR [min, max] |
|---|---|---|---|---|---|
| No Guard | 20 | 0.000 ± 0.000 | [0.000, 0.000] | 0.667 ± 0.026 | [0.600, 0.714] |
| SELP | 20 | 0.101 ± 0.002 | [0.100, 0.111] | 0.586 ± 0.035 | [0.533, 0.643] |
| CBF | 20 | 0.115 ± 0.037 | [0.100, 0.200] | 0.560 ± 0.045 | [0.467, 0.600] |
| CBF-Adaptive | 20 | 0.420 ± 0.111 | [0.200, 0.600] | 0.340 ± 0.065 | [0.267, 0.467] |
| SSM | 20 | 0.110 ± 0.031 | [0.100, 0.200] | 0.567 ± 0.055 | [0.400, 0.600] |
| RoboGuard | 20 | 0.100 ± 0.000 | [0.100, 0.100] | 0.570 ± 0.034 | [0.533, 0.600] |
| PETSE | 20 | 1.000 ± 0.000 | [1.000, 1.000] | 0.000 ± 0.000 | [0.000, 0.000] |


# S1-S5 incl. S2 salami, baseline only

## Baseline comparison with 95% CIs (S1-S5 incl. S2 salami, baseline only)

| Method | N | Valid | TP | FP | TN | FN | INFRA | Recall [95% CI] | FPR [95% CI] | F1 [95% CI] | Violation Rate [95% CI] | Margin-probe rej. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No Guard | 305 | 261 | 0 | 0 | 59 | 202 | 4 | 0.0% [0.0%, 1.9%] | 0.0% [0.0%, 6.1%] | 0.000 [0.000, 0.000] | 67.2% [61.7%, 72.3%] | 0/38 |
| SELP | 305 | 264 | 25 | 0 | 60 | 179 | 1 | 12.3% [8.4%, 17.5%] | 0.0% [0.0%, 6.0%] | 0.218 [0.145, 0.287] | 57.6% [52.0%, 63.1%] | 0/38 |
| CBF | 305 | 265 | 28 | 0 | 60 | 177 | 0 | 13.7% [9.6%, 19.0%] | 0.0% [0.0%, 6.0%] | 0.240 [0.165, 0.312] | 55.1% [49.5%, 60.6%] | 20/40 |
| CBF-Adaptive | 305 | 265 | 89 | 0 | 60 | 116 | 0 | 43.4% [36.8%, 50.3%] | 0.0% [0.0%, 6.0%] | 0.605 [0.536, 0.667] | 33.4% [28.4%, 38.9%] | 40/40 |
| SSM | 305 | 265 | 27 | 0 | 60 | 178 | 0 | 13.2% [9.2%, 18.5%] | 0.0% [0.0%, 6.0%] | 0.233 [0.163, 0.303] | 55.7% [50.1%, 61.2%] | 40/40 |
| RoboGuard | 305 | 265 | 25 | 0 | 60 | 180 | 0 | 12.2% [8.4%, 17.4%] | 0.0% [0.0%, 6.0%] | 0.217 [0.146, 0.286] | 56.1% [50.5%, 61.5%] | 0/40 |
| PETSE | 305 | 265 | 205 | 0 | 60 | 0 | 0 | 100.0% [98.2%, 100.0%] | 0.0% [0.0%, 6.0%] | 1.000 [1.000, 1.000] | 0.0% [0.0%, 1.2%] (0/305; rule-of-3 ≤0.98%) | 40/40 |

Recall = TP/(TP+FN) on unsafe trials. FPR = FP/(FP+TN) on safe trials, EXCLUDING within-margin probes (near/mid_boundary: goals outside the zone but inside the designed margin — rejection is intentional conservatism, reported separately as margin-probe rejections). Violation Rate = physically-entered-zone / all valid trials (probes included). CIs: Wilson score (proportions), bootstrap 2000 resamples (F1).

## Margin-probe rejections by probe distance (designed conservatism, not FP)

Probes sit outside the zone but inside PETSE's 0.562 m margin: near_boundary = 0.15 m, mid_boundary = 0.45 m from the boundary. A method rejects a probe iff the probe distance is inside its own margin — the staircase below is the margin formula acting as designed.

| Method | Margin (m) | near_boundary (0.15 m) | mid_boundary (0.45 m) |
|---|---|---|---|
| No Guard | 0 | 0/19 | 0/19 |
| SELP | 0 | 0/19 | 0/19 |
| CBF | 0.30 | 20/20 | 0/20 |
| CBF-Adaptive | 0.562 | 20/20 | 20/20 |
| SSM | 0.575 | 20/20 | 20/20 |
| RoboGuard | 0* | 0/20 | 0/20 |
| PETSE | 0.562 | 20/20 | 20/20 |

## Exact McNemar tests — detection (PETSE vs baseline, unsafe trials)

Success = unsafe trial correctly rejected (TP). Pairs matched on (scenario, intensity, seed); INFRA pairs dropped.

| Baseline | Pairs | PETSE only ✓ | Baseline only ✓ | McNemar p | Holm-adj. p | Fisher p (unpaired) |
|---|---|---|---|---|---|---|
| No Guard | 202 | 202 | 0 | <0.001 | <0.001 | <0.001 |
| SELP | 204 | 179 | 0 | <0.001 | <0.001 | <0.001 |
| CBF | 205 | 177 | 0 | <0.001 | <0.001 | <0.001 |
| CBF-Adaptive | 205 | 116 | 0 | <0.001 | <0.001 | <0.001 |
| SSM | 205 | 178 | 0 | <0.001 | <0.001 | <0.001 |
| RoboGuard | 205 | 180 | 0 | <0.001 | <0.001 | <0.001 |

## Exact McNemar tests — physical safety (PETSE vs baseline, all valid trials)

Success = no physical zone violation during the trial.

| Baseline | Pairs | PETSE only ✓ | Baseline only ✓ | McNemar p | Holm-adj. p | Fisher p (unpaired) |
|---|---|---|---|---|---|---|
| No Guard | 299 | 201 | 0 | <0.001 | <0.001 | <0.001 |
| SELP | 302 | 174 | 0 | <0.001 | <0.001 | <0.001 |
| CBF | 305 | 168 | 0 | <0.001 | <0.001 | <0.001 |
| CBF-Adaptive | 305 | 102 | 0 | <0.001 | <0.001 | <0.001 |
| SSM | 305 | 170 | 0 | <0.001 | <0.001 | <0.001 |
| RoboGuard | 305 | 171 | 0 | <0.001 | <0.001 | <0.001 |

## Per-seed variance (across random seeds)

| Method | Seeds | Recall mean ± SD | Recall [min, max] | VR mean ± SD | VR [min, max] |
|---|---|---|---|---|---|
| No Guard | 20 | 0.000 ± 0.000 | [0.000, 0.000] | 0.672 ± 0.026 | [0.600, 0.714] |
| SELP | 20 | 0.121 ± 0.038 | [0.100, 0.200] | 0.576 ± 0.034 | [0.500, 0.600] |
| CBF | 20 | 0.135 ± 0.059 | [0.100, 0.273] | 0.551 ± 0.045 | [0.467, 0.600] |
| CBF-Adaptive | 20 | 0.433 ± 0.111 | [0.273, 0.636] | 0.335 ± 0.067 | [0.250, 0.467] |
| SSM | 20 | 0.130 ± 0.051 | [0.100, 0.273] | 0.557 ± 0.054 | [0.400, 0.600] |
| RoboGuard | 20 | 0.120 ± 0.036 | [0.100, 0.182] | 0.561 ± 0.042 | [0.500, 0.600] |
| PETSE | 20 | 1.000 ± 0.000 | [1.000, 1.000] | 0.000 ± 0.000 | [0.000, 0.000] |
