
### bench_detect 2026-11-13 (QUICK)

platform: macOS-26.6.2-arm64-arm-64bit; cpu: arm; cores: 12; python 3.12.4; numpy 2.5.4; scipy 1.18.1; sklearn 1.9.1


Script: `benchmarks/bench_detect.py`; raw CSV in `benchmarks/results/`. 124 maps, tolerance +-2 bins, het allele fraction 0.5 unless stated, detect seed 42, depth = scale of E(d)=depth*(1/(1+d)+0.005) (low=15, mid=60, high=240). Mean runtime per map: v2 0.19s, legacy 0.20s.

#### Criteria

| criterion | measured | verdict |
|---|---|---|
| C1 clean maps: v2 does not force hits (maps with >=1 call <= 0.176) | v2 0.000 of 12 maps (0.000 calls/map); legacy 1.83 regions/map (2.00 bins/map) | PASS |
| C2 v2 F1 >= legacy F1 (1D, +-2 bins) in every N x depth config | v2 wins 3/3; pooled F1 v2 0.935 vs legacy 0.055 | PASS |
| C3 v2 PR-AUC >= legacy PR-AUC (1D) in every config | v2 wins 3/3; pooled AP v2 0.946 vs legacy 0.067 | PASS |
| C4 v2 pair recall (2D) >= 0.8 pooled | 0.917 (CNV recall 0.944) | PASS |
| C5 v2 localization error of detected junctions <= 2 bins (mean, within 10) | v2 2D 0.82; v2 1D 0.07; legacy 1D 3.80 | PASS |
| C6 null p-values: frac p<0.05 in [0.04,0.06] and thinned KS p>0.01, every config | frac range 0.0499-0.0499; min thinned KS p nan | PARTIAL |
| C7 null tail: frac p<1e-3 <= 2e-3 in every config | max 1.03e-03 (p<1e-4: max 8.31e-05) | PASS |
| C8 call-level FDR on planted maps: pooled FDP <= 0.05 (strict +-2 / +-10 / calls sharing no breakpoint coordinate) | strict 0.154; +-10 0.154; unrelated 0.115; 1.2 calls/map from 0 significant pixels/map | FAIL |
| C9 seed reproducibility: identical output for identical seed | 8/8 maps identical; v2 fdr=1 superset consistent on 122/124 maps | FAIL |
| C10 residual bias / pipeline robustness: clean maps with >=1 pair call <= 0.130 | bias_raw 0.00; bias_ice 0.00; nobias_ice 0.00; asis_dense 0.00; asis_sparse 0.00; struct_ref 0.00; struct_ice 0.00 | PASS |
| C11 typing accuracy of matched v2 calls >= 0.8 for each SV kind | deletion 0.75; dup 1.00; inversion 1.00; butterfly 1.00 | PARTIAL |
| C12 crossref type-I at alpha 0.05 (well-specified nulls): beta-binomial in [0.035,0.065] | beta-binomial 0.018-0.040; t-test 0.035-0.063 | PARTIAL |
| C13 crossref power: beta-binomial >= t-test in every well-specified alt cell | wins 14/15; mean power 0.759 vs 0.650 | PARTIAL |
| C14 crossref under misspecification (block-correlated bins, reference bias): beta-binomial type-I <= 0.065 | rho0.02_d200_L10_blockcorr5 0.088 (t 0.427); rho0.02_d200_L10_refbias0.53 0.023 (t 0.205) | FAIL |
| C15 independent reproduction of delve-stats numbers | 6/7 reproduced | PARTIAL |
| C16 PR-curve JSON renders with viz.plot_pr_curves | True | PASS |

#### Legacy vs v2, comparable 1D metric (event hit if any call bin within +-2 of a breakpoint; 5 event kinds x reps + clean maps per row; loc = mean bins)

| n | depth_name | leg_1d_P | v2_1d_P | leg_1d_R | v2_1d_R | leg_1d_F1 | v2_1d_F1 | leg_1d_AP | v2_1d_AP | leg_1d_loc | v2_1d_loc | leg_fp_per_clean | v2_fp_per_clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | low | 0.038 | 0.769 | 0.100 | 1.000 | 0.056 | 0.870 | 0.046 | 0.762 | 4.000 | 0.200 | 2.000 | 0.000 |
| 200 | mid | 0.038 | 1.000 | 0.100 | 1.000 | 0.056 | 1.000 | 0.082 | 1.000 | 1.000 | 0.000 | 1.500 | 0.000 |
| 200 | high | 0.037 | 1.000 | 0.100 | 0.900 | 0.054 | 0.947 | 0.096 | 1.000 | 5.000 | 0.000 | 2.000 | 0.000 |
| all | all | 0.038 | 0.906 | 0.100 | 0.967 | 0.055 | 0.935 | 0.067 | 0.946 | 3.800 | 0.069 | 1.833 | 0.000 |

#### v2 only, strict 2D metric (pair within +-2 of junction pixel; CNV run ends +-2)

| n | depth_name | v2_2d_P | v2_2d_R | v2_2d_F1 | v2_2d_AP | v2_pair_R | v2_cnv_R | v2_pair_loc | v2_clean_maps_with_call |
|---|---|---|---|---|---|---|---|---|---|
| 200 | low | 0.800 | 0.857 | 0.828 | 0.698 | 0.750 | 1.000 | 0.833 | 0.000 |
| 200 | mid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.000 |
| 200 | high | 0.929 | 0.929 | 0.929 | 1.000 | 1.000 | 0.833 | 0.750 | 0.000 |
| all | all | 0.907 | 0.929 | 0.918 | 0.914 | 0.917 | 0.944 | 0.818 | 0.000 |

#### By event kind (main grid pooled)

| kind | maps | leg_1d_R | v2_1d_R | v2_pair_R | v2_cnv_R | v2_pair_fp_per_map | v2_cnv_fp_per_map | leg_fp_per_map | v2_pair_loc | type_acc | types_called |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deletion | 6 | 0.333 | 1.000 | 0.667 | 1.000 | 0.000 | 0.000 | 1.667 | 1.000 | 0.750 | deletion/dup=3; translocation-like/butterfly=1 |
| dup | 6 | 0.167 | 1.000 | 1.000 | 1.000 | 0.500 | 0.000 | 1.833 | 1.333 | 1.000 | deletion/dup=6 |
| inversion | 6 | 0.000 | 1.000 | 1.000 | nan | 0.167 | 0.000 | 2.000 | 0.000 | 1.000 | inversion=6 |
| butterfly | 6 | 0.000 | 1.000 | 1.000 | nan | 0.000 | 0.000 | 1.667 | 1.000 | 1.000 | translocation-like/butterfly=6 |
| cnv | 6 | 0.000 | 0.833 | nan | 0.833 | 0.000 | 0.000 | 1.833 | nan | nan |  |

#### SV recall by kind x depth x N

| kind | depth_name | n | leg_1d_R | v2_1d_R | v2_pair_R | v2_pair_loc | v2_pair_calls |
|---|---|---|---|---|---|---|---|
| deletion | low | 200 | 0.500 | 1.000 | 0.000 | nan | 0.000 |
| dup | low | 200 | 0.000 | 1.000 | 1.000 | 1.500 | 2.500 |
| inversion | low | 200 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| butterfly | low | 200 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| deletion | mid | 200 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| dup | mid | 200 | 0.000 | 1.000 | 1.000 | 1.500 | 1.000 |
| inversion | mid | 200 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| butterfly | mid | 200 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| deletion | high | 200 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| dup | high | 200 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| inversion | high | 200 | 0.000 | 1.000 | 1.000 | 0.000 | 1.500 |
| butterfly | high | 200 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

#### Robustness scenarios (N=200, mid depth; deletion+inversion planted, 30 clean)

| scenario | n_event_maps | n_clean_maps | leg_1d_F1 | v2_1d_F1 | v2_pair_R | v2_2d_P | v2_pair_loc | leg_fp_per_clean | v2_fp_per_clean | v2_clean_maps_with_call | v2_clean_pair_calls | v2_clean_edge_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| af0.25 | 8 | 0 | 0.095 | 0.933 | 0.750 | 1.000 | 0.667 | nan | nan | nan | nan | nan |
| af1 | 8 | 0 | 0.348 | 0.941 | 1.000 | 0.750 | 0.750 | nan | nan | nan | nan | nan |
| bias_raw | 4 | 4 | 0.000 | 1.000 | 1.000 | 0.800 | 0.500 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bias_ice | 4 | 4 | 0.000 | 1.000 | 1.000 | 0.571 | 0.500 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| nobias_ice | 4 | 4 | 0.111 | 1.000 | 1.000 | 1.000 | 0.500 | 1.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| asis_dense | 4 | 4 | 0.111 | 1.000 | 1.000 | 1.000 | 0.500 | 1.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| asis_sparse | 4 | 4 | 0.111 | 1.000 | 1.000 | 1.000 | 0.500 | 1.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| struct_ref | 4 | 4 | 0.375 | 0.667 | 0.250 | 1.000 | 1.000 | 1.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| struct_ice | 4 | 4 | 0.375 | 0.667 | 0.000 | 1.000 | nan | 1.500 | 0.000 | 0.000 | 0.000 | 0.000 |

#### CNV segment-length sweep (gain x1.5, N=200 mid, coverage = raw row sums)

| size | maps | v2_cnv_R | v2_cnv_fp | v2_pair_calls | leg_1d_R |
|---|---|---|---|---|---|
| 5.000 | 2.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| 10.000 | 2.000 | 1.000 | 0.000 | 0.000 | 0.500 |
| 20.000 | 2.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 30.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 50.000 | 2.000 | 1.000 | 0.000 | 0.000 | 0.000 |

#### Null calibration of v2 pair p-values (clean maps; thinned = 25-bin grid, approximately independent pixels)

| scenario | n | depth | maps | tests_per_map | frac_p_lt_0_05 | frac_p_lt_0_01 | frac_p_lt_1e3 | frac_p_lt_1e4 | ks_D_all_mean | thin_n | thin_ks_p | thin_frac_lt_0_05 | maps_with_pair_call | pair_calls_per_map | cnv_calls_per_map | min_p_x_m_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 200 | low | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.0009468 | 0 | 0.000681 | 0 | nan | nan | 0 | 0 | 0 | 2 |
| main | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.0009634 | 4.983e-05 | 0.0005884 | 0 | nan | nan | 0 | 0 | 0 | 1.398 |
| main | 200 | high | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.00103 | 8.305e-05 | 0.000717 | 0 | nan | nan | 0 | 0 | 0 | 1.265 |
| bias_raw | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.001163 | 0 | 0.0008188 | 0 | nan | nan | 0 | 0 | 0 | 1.925 |
| bias_ice | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.0009966 | 1.661e-05 | 0.0005964 | 0 | nan | nan | 0 | 0 | 0 | 1.557 |
| nobias_ice | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.001113 | 6.644e-05 | 0.0005284 | 0 | nan | nan | 0 | 0 | 0 | 0.7183 |
| asis_dense | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.001013 | 4.983e-05 | 0.0005762 | 0 | nan | nan | 0 | 0 | 0 | 1.566 |
| asis_sparse | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.0009468 | 3.322e-05 | 0.0006906 | 0 | nan | nan | 0 | 0 | 0 | 1.147 |
| struct_ref | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.001046 | 0 | 0.0005325 | 0 | nan | nan | 0 | 0 | 0 | 2 |
| struct_ice | 200 | mid | 4 | 1.505e+04 | 0.0499 | 0.01003 | 0.001163 | 0 | 0.0005874 | 0 | nan | nan | 0 | 0 | 0 | 2 |

#### Empirical FDR on planted SV maps (v2 pair calls at q<=0.05)

| depth | maps_with_calls | mean_FDP_strict | mean_FDP_lenient | pooled_FDP_strict | pooled_FDP_lenient | pooled_FDP_unrelated | pair_calls_per_map | sig_pixels_per_map |
|---|---|---|---|---|---|---|---|---|
| low | 6 | 0.194 | 0.194 | 0.333 | 0.333 | 0.333 | 1.500 | 0.000 |
| mid | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| high | 8 | 0.062 | 0.062 | 0.111 | 0.111 | 0.000 | 1.125 | 0.000 |
| all | 22 | 0.076 | 0.076 | 0.154 | 0.154 | 0.115 | 1.182 | 0.000 |

#### Seed reproducibility (N=200 mid; detect seeds varied on a fixed map)

| kind | rep | identical_same_seed | n_seeds | seeds_detecting | calls_min | calls_max | mean_jaccard |
|---|---|---|---|---|---|---|---|
| deletion | 0 | True | 3 | 3.000 | 1 | 1 | 1.000 |
| inversion | 0 | True | 3 | 3.000 | 1 | 1 | 1.000 |
| butterfly | 0 | True | 3 | 3.000 | 1 | 1 | 1.000 |
| dup | 0 | True | 3 | 3.000 | 1 | 1 | 1.000 |
| clean | 0 | True | 3 | nan | 0 | 0 | nan |
| clean | 1 | True | 3 | nan | 0 | 0 | nan |
| clean | 2 | True | 3 | nan | 0 | 0 | nan |
| clean | 3 | True | 3 | nan | 0 | 0 | nan |

#### Crossref: beta-binomial vs legacy t-test (rejection rate at alpha=0.05; decision = cross_validated flag, legacy = |imb|>=0.2 and (t p<0.05 or |z|>=2))

| scenario | effect | regions | betabinom_rej | ttest_rej | new_q05 | new_decision | legacy_decision | rho_hat | fdp_q |
|---|---|---|---|---|---|---|---|---|---|
| rho0_d200_L10 | null | 171 | 0.018 | 0.053 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 |
| rho0_d200_L10 | mu=0.55 | 3 | 1.000 | 1.000 | 0.667 | 0.000 | 0.000 | 0.001 | nan |
| rho0_d200_L10 | mu=0.6 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.001 | nan |
| rho0_d200_L10 | mu=0.7 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | nan |
| rho0.02_d200_L10 | null | 171 | 0.023 | 0.041 | 0.000 | 0.000 | 0.000 | 0.024 | 0.000 |
| rho0.02_d200_L10 | mu=0.55 | 3 | 0.667 | 1.000 | 0.000 | 0.000 | 0.000 | 0.024 | nan |
| rho0.02_d200_L10 | mu=0.6 | 3 | 1.000 | 0.667 | 0.333 | 0.333 | 0.333 | 0.024 | nan |
| rho0.02_d200_L10 | mu=0.7 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.024 | nan |
| rho0.05_d200_L10 | null | 171 | 0.023 | 0.053 | 0.000 | 0.000 | 0.018 | 0.052 | 0.000 |
| rho0.05_d200_L10 | mu=0.55 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.052 | nan |
| rho0.05_d200_L10 | mu=0.6 | 3 | 0.667 | 0.667 | 0.333 | 0.333 | 0.667 | 0.052 | nan |
| rho0.05_d200_L10 | mu=0.7 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.052 | nan |
| rho0.02_d30_L10 | null | 171 | 0.035 | 0.035 | 0.000 | 0.000 | 0.012 | 0.025 | 0.000 |
| rho0.02_d30_L10 | mu=0.55 | 3 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.025 | nan |
| rho0.02_d30_L10 | mu=0.6 | 3 | 1.000 | 0.667 | 0.333 | 0.333 | 0.333 | 0.025 | nan |
| rho0.02_d30_L10 | mu=0.7 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.025 | nan |
| rho0.02_d200_L3 | null | 570 | 0.040 | 0.063 | 0.000 | 0.000 | 0.012 | 0.022 | 0.000 |
| rho0.02_d200_L3 | mu=0.55 | 12 | 0.167 | 0.083 | 0.000 | 0.000 | 0.000 | 0.022 | nan |
| rho0.02_d200_L3 | mu=0.6 | 9 | 0.556 | 0.111 | 0.222 | 0.222 | 0.111 | 0.022 | nan |
| rho0.02_d200_L3 | mu=0.7 | 9 | 1.000 | 0.556 | 0.333 | 0.333 | 0.667 | 0.022 | nan |
| rho0.02_d200_L10_blockcorr5 | null | 171 | 0.088 | 0.427 | 0.006 | 0.006 | 0.047 | 0.025 | 0.111 |
| rho0.02_d200_L10_blockcorr5 | mu=0.55 | 3 | 0.333 | 0.667 | 0.000 | 0.000 | 0.333 | 0.025 | nan |
| rho0.02_d200_L10_blockcorr5 | mu=0.6 | 3 | 0.667 | 1.000 | 0.333 | 0.333 | 0.667 | 0.025 | nan |
| rho0.02_d200_L10_blockcorr5 | mu=0.7 | 3 | 1.000 | 1.000 | 0.667 | 0.667 | 1.000 | 0.025 | nan |
| rho0.02_d200_L10_refbias0.53 | null | 171 | 0.023 | 0.205 | 0.000 | 0.000 | 0.012 | 0.027 | 0.000 |
| rho0.02_d200_L10_refbias0.53 | mu=0.55 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.027 | nan |
| rho0.02_d200_L10_refbias0.53 | mu=0.6 | 3 | 0.667 | 1.000 | 0.333 | 0.333 | 0.667 | 0.027 | nan |
| rho0.02_d200_L10_refbias0.53 | mu=0.7 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.027 | nan |

#### rho recovery (estimate_rho_mom, 3000 bins, depth 200)

| rho_true | signal_frac | rho_hat_mean | rho_hat_sd | bias |
|---|---|---|---|---|
| 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | 0.0500 | 0.0014 | 0.0001 | 0.0014 |
| 0.0000 | 0.1500 | 0.0201 | 0.0004 | 0.0201 |
| 0.0050 | 0.0000 | 0.0053 | 0.0002 | 0.0003 |
| 0.0050 | 0.0500 | 0.0078 | 0.0003 | 0.0028 |
| 0.0050 | 0.1500 | 0.0244 | 0.0004 | 0.0194 |
| 0.0200 | 0.0000 | 0.0200 | 0.0005 | 0.0000 |
| 0.0200 | 0.0500 | 0.0245 | 0.0004 | 0.0045 |
| 0.0200 | 0.1500 | 0.0404 | 0.0013 | 0.0204 |
| 0.0500 | 0.0000 | 0.0516 | 0.0012 | 0.0016 |
| 0.0500 | 0.0500 | 0.0572 | 0.0015 | 0.0072 |
| 0.0500 | 0.1500 | 0.0735 | 0.0015 | 0.0235 |
| 0.1000 | 0.0000 | 0.1073 | 0.0022 | 0.0073 |
| 0.1000 | 0.0500 | 0.1161 | 0.0020 | 0.0161 |
| 0.1000 | 0.1500 | 0.1303 | 0.0026 | 0.0303 |

#### Independent cross-check of delve-stats numbers

| claim | reported | reproduced | match |
|---|---|---|---|
| clean maps (their sim seeds 100-105): calls | 0 each | [0, 0, 0, 0, 0, 0] | True |
| clean maps, 30 fresh seeds 1000-1029: total calls / maps with call | 0 (extrapolated) | 0 / 0 | True |
| deletion top call | near (65,140), deletion/dup | (60, 146, 'deletion/dup') | False |
| inversion (homozygous) top call | near (70,130), inversion | (70, 130, 'inversion') | True |
| rho recovery (true 0.02, 5% planted) | |err| < 0.006 | 0.02432 | True |
| beta-binomial null frac p<0.05 | 0.03-0.07 | 0.0435 | True |
| legacy via detect_anomalies == run_ai flags | identical | 2 | True |
