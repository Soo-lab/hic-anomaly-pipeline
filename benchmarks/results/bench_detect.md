
### bench_detect 2026-11-13

Script: `benchmarks/bench_detect.py`; raw CSV in `benchmarks/results/`. 960 maps, tolerance +-2 bins, het allele fraction 0.5 unless stated, detect seed 42, depth = scale of E(d)=depth*(1/(1+d)+0.005) (low=15, mid=60, high=240). Mean runtime per map: v2 1.74s, legacy 0.29s.

#### Criteria

| criterion | measured | verdict |
|---|---|---|
| C1 clean maps: v2 does not force hits (maps with >=1 call <= 0.082) | v2 0.000 of 180 maps (0.000 calls/map); legacy 2.11 regions/map (3.00 bins/map) | PASS |
| C2 v2 F1 >= legacy F1 (1D, +-2 bins) in every N x depth config | v2 wins 4/6; pooled F1 v2 0.071 vs legacy 0.062 | PARTIAL |
| C3 v2 PR-AUC >= legacy PR-AUC (1D) in every config | v2 wins 6/6; pooled AP v2 0.191 vs legacy 0.042 | PASS |
| C4 v2 pair recall (2D) >= 0.8 pooled | 0.517 (CNV recall 0.000) | PARTIAL |
| C5 v2 localization error of detected junctions <= 2 bins (mean, within 10) | v2 2D 3.22; v2 1D 0.81; legacy 1D 2.57 | FAIL |
| C6 null p-values: frac p<0.05 in [0.04,0.06] and thinned KS p>0.01, every config | frac range 0.0426-0.0493; min thinned KS p 0.013 | PASS |
| C7 null tail: frac p<1e-3 <= 2e-3 in every config | max 9.95e-04 (p<1e-4: max 9.84e-05) | PASS |
| C8 call-level FDR on planted maps: pooled FDP <= 0.05 (strict +-2 / +-10 / calls sharing no breakpoint coordinate) | strict 0.982; +-10 0.967; unrelated 0.806; 29.4 calls/map from 2584 significant pixels/map | FAIL |
| C9 seed reproducibility: identical output for identical seed | 8/8 maps identical; v2 fdr=1 superset consistent on 960/960 maps | PASS |
| C10 residual bias / pipeline robustness: clean maps with >=1 pair call <= 0.130 | bias_raw 0.23; bias_ice 0.93; nobias_ice 1.00; asis_dense 0.00; asis_sparse 0.00; struct_ref 1.00; struct_ice 1.00 | FAIL |
| C11 typing accuracy of matched v2 calls >= 0.8 for each SV kind | deletion 0.00; dup 0.00; inversion 0.88; butterfly 0.61 | PARTIAL |
| C12 crossref type-I at alpha 0.05 (well-specified nulls): beta-binomial in [0.035,0.065] | beta-binomial 0.032-0.042; t-test 0.046-0.059 | PARTIAL |
| C13 crossref power: beta-binomial >= t-test in every well-specified alt cell | wins 14/15; mean power 0.752 vs 0.699 | PARTIAL |
| C14 crossref under misspecification (block-correlated bins, reference bias): beta-binomial type-I <= 0.065 | rho0.02_d200_L10_blockcorr5 0.318 (t 0.442); rho0.02_d200_L10_refbias0.53 0.161 (t 0.194) | FAIL |
| C15 independent reproduction of delve-stats numbers | 7/7 reproduced | PASS |
| C16 PR-curve JSON renders with viz.plot_pr_curves | True | PASS |

#### Legacy vs v2, comparable 1D metric (event hit if any call bin within +-2 of a breakpoint; 5 event kinds x reps + clean maps per row; loc = mean bins)

| n | depth_name | leg_1d_P | v2_1d_P | leg_1d_R | v2_1d_R | leg_1d_F1 | v2_1d_F1 | leg_1d_AP | v2_1d_AP | leg_1d_loc | v2_1d_loc | leg_fp_per_clean | v2_fp_per_clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | low | 0.034 | 0.097 | 0.100 | 0.680 | 0.051 | 0.170 | 0.039 | 0.367 | 3.636 | 1.075 | 1.733 | 0.000 |
| 200 | mid | 0.034 | 0.076 | 0.100 | 0.800 | 0.050 | 0.139 | 0.061 | 0.331 | 0.600 | 0.830 | 1.767 | 0.000 |
| 200 | high | 0.031 | 0.131 | 0.100 | 0.880 | 0.048 | 0.229 | 0.063 | 0.224 | 1.833 | 0.837 | 2.000 | 0.000 |
| 400 | low | 0.016 | 0.026 | 0.080 | 0.860 | 0.026 | 0.050 | 0.022 | 0.180 | 5.071 | 0.587 | 3.067 | 0.000 |
| 400 | mid | 0.061 | 0.020 | 0.240 | 1.000 | 0.098 | 0.040 | 0.044 | 0.166 | 2.353 | 0.360 | 2.100 | 0.000 |
| 400 | high | 0.067 | 0.027 | 0.240 | 0.800 | 0.105 | 0.052 | 0.050 | 0.149 | 0.167 | 1.229 | 2.000 | 0.000 |
| all | all | 0.040 | 0.037 | 0.143 | 0.837 | 0.062 | 0.071 | 0.042 | 0.191 | 2.569 | 0.811 | 2.111 | 0.000 |

#### v2 only, strict 2D metric (pair within +-2 of junction pixel; CNV run ends +-2)

| n | depth_name | v2_2d_P | v2_2d_R | v2_2d_F1 | v2_2d_AP | v2_pair_R | v2_cnv_R | v2_pair_loc | v2_clean_maps_with_call |
|---|---|---|---|---|---|---|---|---|---|
| 200 | low | 0.048 | 0.286 | 0.082 | 0.298 | 0.500 | 0.000 | 2.441 | 0.000 |
| 200 | mid | 0.033 | 0.329 | 0.060 | 0.257 | 0.575 | 0.000 | 2.314 | 0.000 |
| 200 | high | 0.045 | 0.314 | 0.078 | 0.185 | 0.550 | 0.000 | 2.784 | 0.000 |
| 400 | low | 0.010 | 0.271 | 0.020 | 0.157 | 0.475 | 0.000 | 4.056 | 0.000 |
| 400 | mid | 0.007 | 0.286 | 0.014 | 0.144 | 0.500 | 0.000 | 3.722 | 0.000 |
| 400 | high | 0.011 | 0.286 | 0.021 | 0.127 | 0.500 | 0.000 | 4.000 | 0.000 |
| all | all | 0.015 | 0.295 | 0.029 | 0.158 | 0.517 | 0.000 | 3.222 | 0.000 |

#### By event kind (main grid pooled)

| kind | maps | leg_1d_R | v2_1d_R | v2_pair_R | v2_cnv_R | v2_pair_fp_per_map | v2_cnv_fp_per_map | leg_fp_per_map | v2_pair_loc | type_acc | types_called |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deletion | 60 | 0.400 | 0.867 | 0.600 | 0.000 | 17.367 | 0.150 | 2.367 | 2.673 | 0.000 | translocation-like/butterfly=36 |
| dup | 60 | 0.133 | 0.667 | 0.083 | 0.000 | 24.367 | 0.000 | 2.167 | 5.842 | 0.000 | translocation-like/butterfly=5 |
| inversion | 60 | 0.117 | 1.000 | 1.000 | nan | 40.967 | 0.000 | 2.100 | 0.117 | 0.883 | inversion=53; deletion/dup=7 |
| butterfly | 60 | 0.017 | 0.933 | 0.383 | nan | 30.467 | 0.467 | 2.117 | 5.203 | 0.609 | translocation-like/butterfly=14; deletion/dup=9 |
| cnv | 60 | 0.050 | 0.717 | nan | 0.000 | 19.100 | 0.000 | 2.200 | nan | nan |  |

#### SV recall by kind x depth x N

| kind | depth_name | n | leg_1d_R | v2_1d_R | v2_pair_R | v2_pair_loc | v2_pair_calls |
|---|---|---|---|---|---|---|---|
| deletion | low | 200 | 0.300 | 0.500 | 0.200 | 3.167 | 6.000 |
| dup | low | 200 | 0.000 | 0.800 | 0.000 | 5.500 | 14.100 |
| inversion | low | 200 | 0.100 | 1.000 | 1.000 | 0.200 | 8.900 |
| butterfly | low | 200 | 0.100 | 0.900 | 0.800 | 1.800 | 10.700 |
| deletion | mid | 200 | 0.400 | 0.900 | 0.500 | 3.100 | 10.100 |
| dup | mid | 200 | 0.100 | 0.600 | 0.200 | 3.000 | 14.400 |
| inversion | mid | 200 | 0.000 | 1.000 | 1.000 | 0.100 | 21.300 |
| butterfly | mid | 200 | 0.000 | 0.900 | 0.600 | 3.400 | 11.700 |
| deletion | high | 200 | 0.400 | 0.900 | 0.700 | 2.300 | 6.600 |
| dup | high | 200 | 0.100 | 0.600 | 0.300 | 3.714 | 5.000 |
| inversion | high | 200 | 0.000 | 1.000 | 1.000 | 0.000 | 22.500 |
| butterfly | high | 200 | 0.000 | 1.000 | 0.200 | 5.400 | 9.000 |
| deletion | low | 400 | 0.100 | 0.900 | 0.500 | 4.000 | 30.000 |
| dup | low | 400 | 0.000 | 0.900 | 0.000 | 8.429 | 49.400 |
| inversion | low | 400 | 0.200 | 1.000 | 1.000 | 0.400 | 30.900 |
| butterfly | low | 400 | 0.000 | 0.800 | 0.400 | 4.700 | 49.000 |
| deletion | mid | 400 | 0.600 | 1.000 | 0.800 | 1.900 | 37.900 |
| dup | mid | 400 | 0.200 | 1.000 | 0.000 | 6.857 | 52.200 |
| inversion | mid | 400 | 0.200 | 1.000 | 1.000 | 0.000 | 82.200 |
| butterfly | mid | 400 | 0.000 | 1.000 | 0.200 | 7.444 | 55.600 |
| deletion | high | 400 | 0.600 | 1.000 | 0.900 | 1.900 | 17.200 |
| dup | high | 400 | 0.400 | 0.100 | 0.000 | 7.500 | 11.600 |
| inversion | high | 400 | 0.200 | 1.000 | 1.000 | 0.000 | 86.000 |
| butterfly | high | 400 | 0.000 | 1.000 | 0.100 | 8.700 | 49.100 |

#### Robustness scenarios (N=200, mid depth; deletion+inversion planted, 30 clean)

| scenario | n_event_maps | n_clean_maps | leg_1d_F1 | v2_1d_F1 | v2_pair_R | v2_2d_P | v2_pair_loc | leg_fp_per_clean | v2_fp_per_clean | v2_clean_maps_with_call | v2_clean_pair_calls | v2_clean_edge_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| af0.25 | 40 | 0 | 0.069 | 0.143 | 0.500 | 0.042 | 3.514 | nan | nan | nan | nan | nan |
| af1 | 40 | 0 | 0.276 | 0.208 | 0.500 | 0.054 | 2.794 | nan | nan | nan | nan | nan |
| bias_raw | 20 | 30 | 0.107 | 0.260 | 0.900 | 0.115 | 0.850 | 1.900 | 1.900 | 0.467 | 1.100 | 0.533 |
| bias_ice | 20 | 30 | 0.107 | 0.043 | 1.000 | 0.019 | 0.550 | 1.900 | 18.500 | 1.000 | 17.700 | 11.667 |
| nobias_ice | 20 | 30 | 0.071 | 0.045 | 0.950 | 0.019 | 0.750 | 1.767 | 18.567 | 1.000 | 18.567 | 13.433 |
| asis_dense | 20 | 30 | 0.071 | 0.193 | 0.750 | 0.048 | 1.550 | 1.767 | 0.000 | 0.000 | 0.000 | 0.000 |
| asis_sparse | 20 | 30 | 0.071 | 0.216 | 0.750 | 0.048 | 1.500 | 1.767 | 0.000 | 0.000 | 0.000 | 0.000 |
| struct_ref | 20 | 30 | 0.157 | 0.036 | 0.900 | 0.016 | 0.579 | 1.700 | 23.233 | 1.000 | 23.233 | 4.333 |
| struct_ice | 20 | 30 | 0.157 | 0.033 | 0.900 | 0.014 | 1.050 | 1.700 | 25.800 | 1.000 | 25.800 | 7.900 |

#### CNV segment-length sweep (gain x1.5, N=200 mid, coverage = raw row sums)

| size | maps | v2_cnv_R | v2_cnv_fp | v2_pair_calls | leg_1d_R |
|---|---|---|---|---|---|
| 5.000 | 10.000 | 1.000 | 0.000 | 12.300 | 1.000 |
| 10.000 | 10.000 | 1.000 | 0.000 | 18.900 | 0.300 |
| 20.000 | 10.000 | 1.000 | 0.000 | 16.600 | 0.200 |
| 30.000 | 10.000 | 0.000 | 0.000 | 14.200 | 0.000 |
| 50.000 | 10.000 | 0.000 | 0.000 | 11.200 | 0.000 |

#### Null calibration of v2 pair p-values (clean maps; thinned = 25-bin grid, approximately independent pixels)

| scenario | n | depth | maps | tests_per_map | frac_p_lt_0_05 | frac_p_lt_0_01 | frac_p_lt_1e3 | frac_p_lt_1e4 | ks_D_all_mean | thin_n | thin_ks_p | thin_frac_lt_0_05 | maps_with_pair_call | pair_calls_per_map | cnv_calls_per_map | min_p_x_m_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 200 | low | 30 | 1.794e+04 | 0.0436 | 0.008362 | 0.0006855 | 6.502e-05 | 0.03081 | 840 | 0.7372 | 0.05238 | 0 | 0 | 0 | 1.7 |
| main | 200 | mid | 30 | 1.794e+04 | 0.0448 | 0.008352 | 0.0007375 | 6.316e-05 | 0.03648 | 840 | 0.013 | 0.04167 | 0 | 0 | 0 | 1.8 |
| main | 200 | high | 30 | 1.794e+04 | 0.04263 | 0.008362 | 0.0008601 | 7.802e-05 | 0.03718 | 840 | 0.0364 | 0.04762 | 0 | 0 | 0 | 1.2 |
| main | 400 | low | 30 | 7.584e+04 | 0.0493 | 0.009911 | 0.0009955 | 9.581e-05 | 0.02028 | 3600 | 0.2909 | 0.0425 | 0 | 0 | 0 | 1.8 |
| main | 400 | mid | 30 | 7.584e+04 | 0.04555 | 0.008963 | 0.0008944 | 9.23e-05 | 0.02485 | 3600 | 0.7434 | 0.05222 | 0 | 0 | 0 | 1 |
| main | 400 | high | 30 | 7.584e+04 | 0.04772 | 0.009354 | 0.0009621 | 9.845e-05 | 0.02062 | 3600 | 0.03946 | 0.04528 | 0 | 0 | 0 | 1.2 |
| bias_raw | 200 | mid | 30 | 1.794e+04 | 0.08095 | 0.02121 | 0.00428 | 0.0008471 | 0.08397 | 840 | 1.045e-13 | 0.09167 | 0.2333 | 1.1 | 0.8 | 0.4 |
| bias_ice | 200 | mid | 30 | 1.792e+04 | 0.1456 | 0.0603 | 0.01886 | 0.006416 | 0.1301 | 840 | 7.145e-21 | 0.1298 | 0.9333 | 17.7 | 0.8 | 0.0595 |
| nobias_ice | 200 | mid | 30 | 1.794e+04 | 0.1485 | 0.07332 | 0.03153 | 0.01515 | 0.1136 | 840 | 7.437e-10 | 0.1131 | 1 | 18.57 | 0 | 0.001507 |
| asis_dense | 200 | mid | 30 | 1.794e+04 | 0.04478 | 0.008369 | 0.0007784 | 5.016e-05 | 0.03464 | 840 | 0.03111 | 0.03929 | 0 | 0 | 0 | 1.9 |
| asis_sparse | 200 | mid | 30 | 1.794e+04 | 0.0461 | 0.008824 | 0.0008063 | 5.945e-05 | 0.02888 | 840 | 0.1405 | 0.04405 | 0 | 0 | 0 | 1.8 |
| struct_ref | 200 | mid | 30 | 1.794e+04 | 0.1295 | 0.05271 | 0.01962 | 0.00859 | 0.13 | 840 | 1.189e-17 | 0.1131 | 1 | 23.23 | 0 | 0.004021 |
| struct_ice | 200 | mid | 30 | 1.794e+04 | 0.1376 | 0.0562 | 0.02019 | 0.008109 | 0.1375 | 840 | 1.19e-13 | 0.1119 | 1 | 25.8 | 0 | 0.004696 |

#### Empirical FDR on planted SV maps (v2 pair calls at q<=0.05)

| depth | maps_with_calls | mean_FDP_strict | mean_FDP_lenient | pooled_FDP_strict | pooled_FDP_lenient | pooled_FDP_unrelated | pair_calls_per_map | sig_pixels_per_map |
|---|---|---|---|---|---|---|---|---|
| low | 75 | 0.937 | 0.908 | 0.980 | 0.964 | 0.848 | 26.533 | 1451.240 |
| mid | 80 | 0.964 | 0.944 | 0.985 | 0.973 | 0.813 | 35.675 | 2794.488 |
| high | 80 | 0.953 | 0.916 | 0.980 | 0.963 | 0.756 | 25.875 | 3436.537 |
| all | 235 | 0.952 | 0.923 | 0.982 | 0.967 | 0.806 | 29.421 | 2584.362 |

#### Seed reproducibility (N=200 mid; detect seeds varied on a fixed map)

| kind | rep | identical_same_seed | n_seeds | seeds_detecting | calls_min | calls_max | mean_jaccard |
|---|---|---|---|---|---|---|---|
| deletion | 0 | True | 10 | 0.000 | 11 | 15 | 0.416 |
| inversion | 0 | True | 10 | 10.000 | 16 | 26 | 0.356 |
| butterfly | 0 | True | 10 | 10.000 | 6 | 15 | 0.488 |
| dup | 0 | True | 10 | 0.000 | 10 | 14 | 0.309 |
| clean | 0 | True | 10 | nan | 0 | 0 | nan |
| clean | 1 | True | 10 | nan | 0 | 0 | nan |
| clean | 2 | True | 10 | nan | 0 | 0 | nan |
| clean | 3 | True | 10 | nan | 0 | 0 | nan |

#### Crossref: beta-binomial vs legacy t-test (rejection rate at alpha=0.05; decision = cross_validated flag, legacy = |imb|>=0.2 and (t p<0.05 or |z|>=2))

| scenario | effect | regions | betabinom_rej | ttest_rej | new_q05 | new_decision | legacy_decision | rho_hat | fdp_q |
|---|---|---|---|---|---|---|---|---|---|
| rho0_d200_L10 | null | 2840 | 0.032 | 0.052 | 0.001 | 0.000 | 0.000 | 0.001 | 0.024 |
| rho0_d200_L10 | mu=0.55 | 60 | 1.000 | 0.967 | 0.833 | 0.000 | 0.000 | 0.001 | nan |
| rho0_d200_L10 | mu=0.6 | 60 | 1.000 | 1.000 | 1.000 | 0.450 | 0.450 | 0.001 | nan |
| rho0_d200_L10 | mu=0.7 | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | nan |
| rho0.02_d200_L10 | null | 2840 | 0.039 | 0.059 | 0.001 | 0.001 | 0.001 | 0.022 | 0.018 |
| rho0.02_d200_L10 | mu=0.55 | 60 | 0.533 | 0.533 | 0.100 | 0.017 | 0.050 | 0.022 | nan |
| rho0.02_d200_L10 | mu=0.6 | 60 | 0.983 | 0.967 | 0.733 | 0.517 | 0.517 | 0.022 | nan |
| rho0.02_d200_L10 | mu=0.7 | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.022 | nan |
| rho0.05_d200_L10 | null | 2840 | 0.042 | 0.052 | 0.002 | 0.002 | 0.022 | 0.055 | 0.074 |
| rho0.05_d200_L10 | mu=0.55 | 60 | 0.250 | 0.317 | 0.033 | 0.033 | 0.217 | 0.055 | nan |
| rho0.05_d200_L10 | mu=0.6 | 60 | 0.700 | 0.650 | 0.317 | 0.317 | 0.600 | 0.055 | nan |
| rho0.05_d200_L10 | mu=0.7 | 40 | 1.000 | 1.000 | 0.975 | 0.975 | 1.000 | 0.055 | nan |
| rho0.02_d30_L10 | null | 2840 | 0.036 | 0.046 | 0.001 | 0.001 | 0.011 | 0.024 | 0.022 |
| rho0.02_d30_L10 | mu=0.55 | 60 | 0.300 | 0.283 | 0.000 | 0.000 | 0.117 | 0.024 | nan |
| rho0.02_d30_L10 | mu=0.6 | 60 | 0.800 | 0.683 | 0.350 | 0.350 | 0.617 | 0.024 | nan |
| rho0.02_d30_L10 | mu=0.7 | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.024 | nan |
| rho0.02_d200_L3 | null | 9500 | 0.037 | 0.050 | 0.000 | 0.000 | 0.008 | 0.023 | 0.008 |
| rho0.02_d200_L3 | mu=0.55 | 180 | 0.139 | 0.100 | 0.011 | 0.011 | 0.061 | 0.023 | nan |
| rho0.02_d200_L3 | mu=0.6 | 160 | 0.588 | 0.263 | 0.075 | 0.075 | 0.219 | 0.023 | nan |
| rho0.02_d200_L3 | mu=0.7 | 160 | 0.988 | 0.719 | 0.819 | 0.819 | 0.875 | 0.023 | nan |
| rho0.02_d200_L10_blockcorr5 | null | 2840 | 0.318 | 0.442 | 0.187 | 0.060 | 0.054 | 0.023 | 0.826 |
| rho0.02_d200_L10_blockcorr5 | mu=0.55 | 60 | 0.467 | 0.567 | 0.317 | 0.200 | 0.183 | 0.023 | nan |
| rho0.02_d200_L10_blockcorr5 | mu=0.6 | 60 | 0.883 | 0.900 | 0.850 | 0.600 | 0.583 | 0.023 | nan |
| rho0.02_d200_L10_blockcorr5 | mu=0.7 | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.023 | nan |
| rho0.02_d200_L10_refbias0.53 | null | 2840 | 0.161 | 0.194 | 0.015 | 0.006 | 0.010 | 0.026 | 0.284 |
| rho0.02_d200_L10_refbias0.53 | mu=0.55 | 60 | 0.517 | 0.567 | 0.050 | 0.033 | 0.050 | 0.026 | nan |
| rho0.02_d200_L10_refbias0.53 | mu=0.6 | 60 | 1.000 | 1.000 | 0.700 | 0.583 | 0.600 | 0.026 | nan |
| rho0.02_d200_L10_refbias0.53 | mu=0.7 | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.026 | nan |

#### rho recovery (estimate_rho_mom, 3000 bins, depth 200)

| rho_true | signal_frac | rho_hat_mean | rho_hat_sd | bias |
|---|---|---|---|---|
| 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| 0.0000 | 0.0500 | 0.0014 | 0.0001 | 0.0014 |
| 0.0000 | 0.1500 | 0.0198 | 0.0004 | 0.0198 |
| 0.0050 | 0.0000 | 0.0051 | 0.0002 | 0.0001 |
| 0.0050 | 0.0500 | 0.0079 | 0.0003 | 0.0029 |
| 0.0050 | 0.1500 | 0.0242 | 0.0004 | 0.0192 |
| 0.0200 | 0.0000 | 0.0203 | 0.0006 | 0.0003 |
| 0.0200 | 0.0500 | 0.0252 | 0.0009 | 0.0052 |
| 0.0200 | 0.1500 | 0.0409 | 0.0012 | 0.0209 |
| 0.0500 | 0.0000 | 0.0520 | 0.0011 | 0.0020 |
| 0.0500 | 0.0500 | 0.0576 | 0.0014 | 0.0076 |
| 0.0500 | 0.1500 | 0.0736 | 0.0021 | 0.0236 |
| 0.1000 | 0.0000 | 0.1063 | 0.0027 | 0.0063 |
| 0.1000 | 0.0500 | 0.1133 | 0.0031 | 0.0133 |
| 0.1000 | 0.1500 | 0.1300 | 0.0031 | 0.0300 |

#### Independent cross-check of delve-stats numbers

| claim | reported | reproduced | match |
|---|---|---|---|
| clean maps (their sim seeds 100-105): calls | 0 each | [0, 0, 0, 0, 0, 0] | True |
| clean maps, 30 fresh seeds 1000-1029: total calls / maps with call | 0 (extrapolated) | 0 / 0 | True |
| deletion top call | near (65,140), deletion/dup | (65, 140, 'deletion/dup') | True |
| inversion (homozygous) top call | near (70,130), inversion | (70, 130, 'inversion') | True |
| rho recovery (true 0.02, 5% planted) | |err| < 0.006 | 0.02432 | True |
| beta-binomial null frac p<0.05 | 0.03-0.07 | 0.0435 | True |
| legacy via detect_anomalies == run_ai flags | identical | 2 | True |
