
### bench_matrix 2026-11-12

| pairs | chrom len | bin | baseline s | baseline peak MB | new s | new peak MB |
|---|---|---|---|---|---|---|
| 20000 | 5000000 | 10000 | 0.24 | 1.3 | 0.22 | 9.9 |
| 100000 | 50000000 | 10000 | 1.26 | 101.5 | 1.12 | 21.2 |
| 300000 | 100000000 | 5000 | 4.14 | 1605.2 | 3.33 | 48.5 |

### bench_matrix 2026-11-13 (strict_pairs=False default, per-read overhead reduced)

| pairs | chrom len | bin | baseline s | baseline peak MB | new s | new peak MB |
|---|---|---|---|---|---|---|
| 20000 | 5000000 | 10000 | 0.25 | 1.3 | 0.17 | 1.6 |
| 100000 | 50000000 | 10000 | 1.34 | 101.5 | 0.86 | 9.6 |
| 300000 | 100000000 | 5000 | 4.80 | 1605.2 | 2.63 | 29.3 |

> **SUPERSEDED (2026-11-13):** the bench_heldout section below was measured on the detector BEFORE the second fix round (2D F1 0.367, FDP 0.613, clean-map call rate 0.176). Current numbers (2D F1 0.510, FDP 0.164, clean 0.019) are in benchmarks/results/heldout_bench.md. Kept for history only; do not quote.

### bench_heldout 2026-11-13

Script `benchmarks/bench_heldout.py`; CSV/JSON `benchmarks/results/heldout_*`. 504 maps (108 clean, 396 with events, 445 events). NB noise, power-law/plateau decay, compartments, per-bin bias+dropout, ICE before v2, tol +-2, detector seed 7, different seed root than bench_detect.

#### Criteria

| criterion | measured | verdict |
|---|---|---|
| H1 clean: maps with any v2 call <= 0.092 | 0.176 of 108 maps; 0.204 calls/map; legacy 3.26 regions/map | FAIL |
| H2 call-level FDP <= 0.10 (q=0.05, pooled event maps) | 0.613 (single 0.626, mixed 0.526) | FAIL |
| H3 v2 2D F1 >= 0.80 (pooled incl. clean FPs) | 0.367 | FAIL |
| H4 v2 1D F1 > legacy F1 and PR-AUC > legacy | F1 0.426 vs 0.068; AP 0.482 vs 0.031 | PASS |
| H5 v2 2D PR-AUC >= 0.90 | 0.381 | FAIL |
| H6 v2 localization mean <= 2 bins | 1.19 (legacy 4.34) | PASS |
| H7 mixed maps: recall >= 0.7 | R 0.271 P 0.434 F1 0.333 | FAIL |

#### Pooled (event maps + clean maps; FPs from clean maps counted)

| maps | events | leg_P | v2_1d_P | v2_2d_P | leg_R | v2_1d_R | v2_2d_R | leg_F1 | v2_1d_F1 | v2_2d_F1 | leg_AP | v2_1d_AP | v2_2d_AP | leg_loc | v2_loc | v2_FDP | leg_s_per_map | v2_s_per_map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 504.000 | 445.000 | 0.043 | 0.428 | 0.363 | 0.160 | 0.425 | 0.371 | 0.068 | 0.426 | 0.367 | 0.031 | 0.482 | 0.381 | 4.340 | 1.193 | 0.613 | 0.178 | 0.350 |

#### By N

| n | maps | events | leg_P | v2_1d_P | v2_2d_P | leg_R | v2_1d_R | v2_2d_R | leg_F1 | v2_1d_F1 | v2_2d_F1 | leg_AP | v2_1d_AP | v2_2d_AP | leg_loc | v2_loc | v2_FDP | leg_s_per_map | v2_s_per_map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150.000 | 168.000 | 137.000 | 0.067 | 0.547 | 0.474 | 0.161 | 0.299 | 0.263 | 0.094 | 0.387 | 0.338 | 0.052 | 0.542 | 0.451 | 4.720 | 1.000 | 0.493 | 0.166 | 0.110 |
| 300.000 | 168.000 | 152.000 | 0.050 | 0.459 | 0.388 | 0.164 | 0.480 | 0.421 | 0.077 | 0.469 | 0.404 | 0.035 | 0.572 | 0.459 | 4.100 | 1.114 | 0.589 | 0.175 | 0.271 |
| 500.000 | 168.000 | 156.000 | 0.029 | 0.361 | 0.304 | 0.154 | 0.481 | 0.417 | 0.049 | 0.412 | 0.351 | 0.021 | 0.508 | 0.387 | 4.148 | 1.370 | 0.675 | 0.193 | 0.668 |

#### By scenario x N

| scenario | n | maps | events | leg_P | v2_1d_P | v2_2d_P | leg_R | v2_1d_R | v2_2d_R | leg_F1 | v2_1d_F1 | v2_2d_F1 | leg_AP | v2_1d_AP | v2_2d_AP | leg_loc | v2_loc | v2_FDP | leg_s_per_map | v2_s_per_map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_d25 | 150 | 60 | 60 | 0.059 | 0.562 | 0.469 | 0.117 | 0.300 | 0.250 | 0.078 | 0.391 | 0.326 | 0.063 | 0.544 | 0.446 | 5.000 | 1.125 | 0.531 | 0.163 | 0.111 |
| single_d100 | 150 | 60 | 60 | 0.092 | 0.571 | 0.528 | 0.183 | 0.333 | 0.317 | 0.122 | 0.421 | 0.396 | 0.066 | 0.642 | 0.538 | 4.367 | 0.579 | 0.472 | 0.169 | 0.109 |
| mixed | 150 | 12 | 17 | 0.174 | 0.750 | 0.500 | 0.235 | 0.176 | 0.118 | 0.200 | 0.286 | 0.190 | 0.126 | 0.405 | 0.228 | 4.818 | 3.000 | 0.400 | 0.163 | 0.111 |
| single_d25 | 300 | 60 | 60 | 0.045 | 0.393 | 0.317 | 0.133 | 0.400 | 0.333 | 0.068 | 0.397 | 0.325 | 0.036 | 0.467 | 0.355 | 3.731 | 1.500 | 0.683 | 0.175 | 0.271 |
| single_d100 | 300 | 60 | 60 | 0.067 | 0.531 | 0.485 | 0.200 | 0.567 | 0.550 | 0.100 | 0.548 | 0.516 | 0.043 | 0.725 | 0.597 | 3.964 | 1.057 | 0.515 | 0.173 | 0.271 |
| mixed | 300 | 12 | 32 | 0.139 | 0.577 | 0.423 | 0.156 | 0.469 | 0.344 | 0.147 | 0.517 | 0.379 | 0.108 | 0.593 | 0.460 | 4.938 | 0.455 | 0.556 | 0.183 | 0.262 |
| single_d25 | 500 | 60 | 60 | 0.037 | 0.382 | 0.325 | 0.183 | 0.483 | 0.417 | 0.062 | 0.426 | 0.365 | 0.028 | 0.425 | 0.341 | 3.731 | 1.903 | 0.675 | 0.196 | 0.669 |
| single_d100 | 500 | 60 | 60 | 0.024 | 0.323 | 0.288 | 0.117 | 0.533 | 0.500 | 0.040 | 0.403 | 0.366 | 0.018 | 0.570 | 0.475 | 4.762 | 1.094 | 0.712 | 0.194 | 0.670 |
| mixed | 500 | 12 | 36 | 0.103 | 0.609 | 0.435 | 0.167 | 0.389 | 0.278 | 0.128 | 0.475 | 0.339 | 0.078 | 0.648 | 0.391 | 4.000 | 0.600 | 0.520 | 0.174 | 0.637 |

#### By scenario x decay

| scenario | decay | maps | events | leg_P | v2_1d_P | v2_2d_P | leg_R | v2_1d_R | v2_2d_R | leg_F1 | v2_1d_F1 | v2_2d_F1 | leg_AP | v2_1d_AP | v2_2d_AP | leg_loc | v2_loc | v2_FDP | leg_s_per_map | v2_s_per_map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_d25 | pl0.8 | 60 | 60 | 0.051 | 0.417 | 0.333 | 0.167 | 0.333 | 0.267 | 0.078 | 0.370 | 0.296 | 0.037 | 0.316 | 0.251 | 3.630 | 1.118 | 0.667 | 0.176 | 0.344 |
| single_d100 | pl0.8 | 60 | 60 | 0.045 | 0.574 | 0.542 | 0.150 | 0.450 | 0.433 | 0.069 | 0.505 | 0.481 | 0.032 | 0.535 | 0.489 | 4.808 | 0.654 | 0.458 | 0.174 | 0.342 |
| single_d25 | pl1.3 | 60 | 60 | 0.030 | 0.443 | 0.371 | 0.100 | 0.450 | 0.383 | 0.047 | 0.446 | 0.377 | 0.034 | 0.434 | 0.345 | 4.774 | 2.258 | 0.629 | 0.177 | 0.353 |
| single_d100 | pl1.3 | 60 | 60 | 0.066 | 0.391 | 0.356 | 0.217 | 0.567 | 0.533 | 0.102 | 0.463 | 0.427 | 0.040 | 0.599 | 0.504 | 3.750 | 1.306 | 0.644 | 0.180 | 0.354 |
| single_d25 | plateau | 60 | 60 | 0.051 | 0.400 | 0.339 | 0.167 | 0.400 | 0.350 | 0.078 | 0.400 | 0.344 | 0.036 | 0.518 | 0.424 | 4.214 | 1.043 | 0.661 | 0.181 | 0.355 |
| single_d100 | plateau | 60 | 60 | 0.041 | 0.391 | 0.343 | 0.133 | 0.417 | 0.400 | 0.063 | 0.403 | 0.369 | 0.031 | 0.643 | 0.491 | 4.480 | 0.792 | 0.657 | 0.183 | 0.355 |
| mixed | pl0.8 | 12 | 30 | 0.125 | 0.750 | 0.583 | 0.167 | 0.300 | 0.233 | 0.143 | 0.429 | 0.333 | 0.102 | 0.496 | 0.373 | 5.083 | 0.429 | 0.385 | 0.177 | 0.336 |
| mixed | pl1.3 | 12 | 28 | 0.128 | 0.611 | 0.529 | 0.179 | 0.393 | 0.321 | 0.149 | 0.478 | 0.400 | 0.100 | 0.651 | 0.363 | 5.533 | 0.556 | 0.444 | 0.173 | 0.337 |
| mixed | plateau | 12 | 27 | 0.132 | 0.522 | 0.292 | 0.185 | 0.444 | 0.259 | 0.154 | 0.480 | 0.275 | 0.086 | 0.520 | 0.379 | 3.143 | 1.500 | 0.654 | 0.170 | 0.337 |

#### Recall by kind

| kind | events | leg_1d_R | v2_1d_R | v2_2d_R | v2_cnv_R | v2_loc | leg_loc |
|---|---|---|---|---|---|---|---|
| butterfly | 110 | 0.109 | 0.655 | 0.655 | 0.000 | 0.917 | 4.553 |
| deletion | 109 | 0.138 | 0.376 | 0.239 | 0.037 | 2.973 | 4.500 |
| dup | 110 | 0.236 | 0.118 | 0.036 | 0.000 | 2.000 | 3.721 |
| inversion | 116 | 0.155 | 0.543 | 0.543 | 0.000 | 0.400 | 4.731 |

#### Recall by event size (bins)

| size_bin | events | v2_2d_R | v2_1d_R | leg_1d_R |
|---|---|---|---|---|
| (0, 15] | 55 | 0.127 | 0.200 | 0.236 |
| (15, 30] | 157 | 0.248 | 0.287 | 0.134 |
| (30, 60] | 143 | 0.497 | 0.559 | 0.154 |
| (60, 200] | 90 | 0.533 | 0.589 | 0.167 |

#### Recall by allele fraction

| af | events | v2_2d_R | leg_1d_R |
|---|---|---|---|
| 0.350 | 150.000 | 0.247 | 0.120 |
| 0.700 | 159.000 | 0.497 | 0.182 |
| 1.000 | 136.000 | 0.360 | 0.176 |

#### Clean maps

| scenario | n | decay | maps | v2_calls_per_map | v2_cnv_per_map | v2_maps_with_call | leg_regions_per_map |
|---|---|---|---|---|---|---|---|
| clean | 150 | pl0.8 | 6 | 0.000 | 0.167 | 0.000 | 1.833 |
| clean | 150 | pl1.3 | 6 | 0.000 | 0.167 | 0.000 | 2.000 |
| clean | 150 | plateau | 6 | 0.000 | 0.000 | 0.000 | 2.000 |
| clean | 300 | pl0.8 | 6 | 0.000 | 0.167 | 0.000 | 3.000 |
| clean | 300 | pl1.3 | 6 | 0.000 | 0.667 | 0.000 | 2.833 |
| clean | 300 | plateau | 6 | 0.000 | 0.167 | 0.000 | 3.000 |
| clean | 500 | pl0.8 | 6 | 0.000 | 0.833 | 0.000 | 5.000 |
| clean | 500 | pl1.3 | 6 | 0.000 | 0.500 | 0.000 | 5.000 |
| clean | 500 | plateau | 6 | 0.000 | 0.000 | 0.000 | 5.000 |
| clean_stress | 150 | pl0.8 | 6 | 0.000 | 0.000 | 0.000 | 1.667 |
| clean_stress | 150 | pl1.3 | 6 | 0.167 | 0.000 | 0.167 | 2.000 |
| clean_stress | 150 | plateau | 6 | 0.167 | 0.000 | 0.167 | 1.833 |
| clean_stress | 300 | pl0.8 | 6 | 0.000 | 0.333 | 0.000 | 3.000 |
| clean_stress | 300 | pl1.3 | 6 | 0.000 | 0.000 | 0.000 | 2.833 |
| clean_stress | 300 | plateau | 6 | 0.000 | 0.000 | 0.000 | 3.000 |
| clean_stress | 500 | pl0.8 | 6 | 0.000 | 0.167 | 0.000 | 5.000 |
| clean_stress | 500 | pl1.3 | 6 | 0.000 | 0.167 | 0.000 | 4.667 |
| clean_stress | 500 | plateau | 6 | 0.000 | 0.000 | 0.000 | 5.000 |

Clean totals: {'maps': 108, 'v2_false_calls_per_map': 0.2037037037037037, 'v2_maps_with_any_call': 0.17592592592592593, 'v2_pair_calls_per_map': 0.018518518518518517, 'leg_regions_per_map': 3.259259259259259}

> **SUPERSEDED (2026-11-13):** the bench_heldout section below was measured on the detector BEFORE the second fix round (2D F1 0.367, FDP 0.613, clean-map call rate 0.176). Current numbers (2D F1 0.510, FDP 0.164, clean 0.019) are in benchmarks/results/heldout_bench.md. Kept for history only; do not quote.

#### bench_heldout 2026-11-13 analysis (verity-stats)
Verdict FAIL on held-out data (H1,H2,H3,H5,H7 fail; H4,H6 pass). v2 beats legacy by a wide margin but is far below bench_detect claims (F1 0.909 -> 2D 0.367; PR-AUC 0.975 -> 0.381; call FDP 0.154 -> 0.613; clean maps with a call ~0 -> 0.176).
- CNV track is the main FP source: 223/227 cnv calls on event maps and 20/22 calls on clean maps are false (low-coverage/dropout bins passed as raw row sums). Pair-only FDP = 45/210 = 0.21 (still > 0.05).
- Pair calls on clean maps: 2/108 maps (both clean_stress, N=150) -> pair null is OK.
- Recall by kind (2D): butterfly 0.655, inversion 0.543, deletion 0.239, dup 0.036. Homozygous (af=1) deletion recall 0.000 (deleted bins have zero coverage and are masked; no junction/cnv call). Dup junction nearly invisible in obs/exp.
- Recall by size: <=15 bins 0.127; 15-30 0.248. Mixed maps recall 0.271 (2-4 events).
- FDP and runtime grow with N: FDP 0.49/0.59/0.68 at N=150/300/500; v2 0.11/0.27/0.67 s/map (legacy ~0.18). Simulator sanity checked: strong events (af 1, depth 100-400) are visible in log-OE (z>1.5 at junction).

### bench_validate_cli 2026-11-13 (end-to-end CLI, planted synthetic BAM, 20 Mb chr1, screen 250 kb, zoom 50 kb; peak RSS of CLI process, macOS arm64)

| pairs | method | wall s | peak RSS MB |
|---|---|---|---|
| 50000 | v2 | 1.79 | 298 |
| 50000 | legacy | 2.03 | 553 |
| 200000 | v2 | 1.97 | 273 |
| 200000 | legacy | 2.17 | 554 |
| 600000 | v2 | 2.47 | 303 |
| 600000 | legacy | 2.61 | 573 |

Process pool (build_matrices_parallel, 4 contigs, 1.2M pairs, 50 kb bins, outside sandbox): workers=1 1.02 s, workers=4 0.55 s (1.9x). Inside the sandbox the pool is blocked (Errno 1) and falls back to serial with a warning. End-to-end `--chrom all` CLI: workers=1 2.32 s vs workers=4 2.34 s (matrix build is a small share of runtime at this size).
