# Methods recommendation: hic-anomaly-pipeline

Status: written from prior knowledge of the literature; URLs not re-fetched this session. Verify versions before citing in README. Fact = published/documented. Opinion = marked "Opinion".

Code facts observed (universal_hic_pipeline.py): dense int32 NxN matrix (l.184); `read_pairs` dict (l.187) never evicted; no MAPQ filter; `extract_bin_features` is a per-bin Python loop with fixed +-5/+-10 windows on raw counts; `IsolationForest(contamination=0.01)` forces ~1% positives (l.292-298); `robust_z` already exists (l.239).

## 1. ICE on sparse matrices

Reference: Imakaev 2012, https://www.nature.com/articles/nmeth.2148 ; cooler `balance` implementation https://github.com/open2c/cooler ; cooltools https://github.com/open2c/cooltools

Model: observed O_ij = b_i b_j T_ij. Find bias vector b.

Algorithm (matrix-free, sparse COO of upper triangle, mirrored):
1. Mask bins: drop bins with coverage s_i = sum_j O_ij below threshold. Use MAD-based cutoff on log10(s_i): mask if log s_i < median(log s) - 2*MAD_scaled (cooler default `--mad-max 5`, `--min-nnz 10`). Also mask diagonal and first `ignore_diags` (cooler default 2) diagonals before balancing. Set masked rows/cols to NaN/0.
2. Init w = 1 (length N).
3. Iterate:
   - For each nonzero (i,j,v): v' = v / (w_i w_j)
   - s_i = sum_j v'_ij (use np.bincount(i, v', N) + np.bincount(j, v', N) for upper-triangle storage)
   - s_i /= mean(s_i over unmasked) (so mean = 1)
   - w_i *= s_i
   - Masked bins: s_i = 1 (no update).
4. Converge when max_i |s_i - 1| (unmasked) < tol, tol=1e-5, max_iter=200 (cooler default 200). Typically 20-60 iters.
5. Final scale: divide w so that mean row sum of balanced matrix = 1 (cooler `scale`), report `weight = 1/w`. Store bias, do not densify.
6. Balanced pixel = v / (w_i w_j), computed lazily.

Cost: O(nnz) per iteration, memory O(nnz + N). Single-chrom 10 kb of 250 Mb = 25k bins; sparse nnz is far below N^2 (625M).

Trade-offs: ICE assumes equal visibility, fails when copy-number alterations are present (CNV/aneuploid). Opinion: for an anomaly tool, CNV is itself a signal, so ALSO keep the raw per-bin coverage z-score as a separate feature and flag bins where balancing bias is extreme (bias z > 3), rather than silently removing them.
KR alternative (Knight-Ruiz 2012): faster convergence, may fail on sparse/low-coverage; ICE is the safer default.
Bins with all-zero rows (centromeres, gaps) must stay masked and excluded from every downstream statistic (current code only excludes at training time).

## 2. Distance-decay expected model

Reference: cooltools `expected-cis` https://cooltools.readthedocs.io/en/latest/ ; Lieberman-Aiden 2009 https://www.science.org/doi/10.1126/science.1181369

Per diagonal d = |i-j| on the balanced matrix, over valid (unmasked) pixels only:
- E(d) = sum_{i} B_{i,i+d} / n_valid(d), where n_valid(d) = count of pairs (i,i+d) with both bins unmasked.
- Use mean of balanced values (cooltools default), or trimmed mean / median for robustness against the very SVs we seek. Opinion: use a 10% trimmed mean, or median-of-nonzeros with a zero-inflation correction, because outlier pixels inflate the mean.
- Smoothing: low-count diagonals at large d are noisy. Options:
  a. Log-binned diagonals (cooltools `logbin_expected`, bin width factor ~1.3), then interpolate in log-log space (monotone PCHIP or LOWESS).
  b. Isotonic regression on log E(d) vs log d, enforcing monotone non-increasing (fine for cis 0.1-100 Mb).
- Regions with n_valid(d) < 10: stop at d_max, set E to the last reliable value, and exclude those pixels from scoring.
- Obs/exp: R_ij = B_ij / E(|i-j|). Score on log2((B_ij + eps)/ (E + eps)) with eps ~ 0.1*E_min to avoid -inf. 
- Counts-aware option: under H0 count ~ Poisson/NB with mean mu_ij = b_i b_j * E(d) (use raw counts and ICE biases). p_ij = P(X >= O_ij). This gives real p-values for pixel enrichment (HiCCUPS-style, Rao 2014 https://doi.org/10.1016/j.cell.2014.11.021). Prefer this for statistically defensible hits.
- Expected for chromosome translocations: trans expected is flat per chromosome pair; mean of trans balanced counts, computed per pair.

## 3. Local features for SV signatures

Per-bin features should be computed on obs/exp (or log2 obs/exp) matrix L_ij = log2(R_ij), with masked bins as NaN (use nan-aware means).

Signature facts:
- Translocation (inter-chrom, or cis far): "butterfly/bow-tie" pattern at breakpoint on both sides, two off-diagonal blocks of enriched contacts forming an X or corner pattern across the breakpoint (Dixon 2018, https://www.nature.com/articles/s41588-018-0150-z ; Harewood 2017 hic_breakfinder, https://doi.org/10.1186/s13059-017-1253-7; EagleC Wang 2022 https://www.nature.com/articles/s41467-022-31247-4).
- Deletion: new contacts between flanks, a sharp square block of enriched contacts at (breakpoint1, breakpoint2) off the diagonal, with a loss of signal in deleted region (coverage dip).
- Inversion: enrichment pattern runs along the ANTI-diagonal inside the box [bp1, bp2]^2 (reversed orientation), with a bow-tie at each breakpoint. In the pair orientation, FR/RF orientation of reads is the more direct evidence. Keep strand in the pair counter if wanted (see 6).
- Duplication: bright off-diagonal block at the tandem junction, coverage rise.

Features (window w bins; use w in {5,10,25} for multi-scale):
1. Insulation score (Crane 2015, https://www.nature.com/articles/nature14450): IS_i = mean_{a in [i-w, i), b in (i, i+w]} B_ab, then delta_i = log2(IS_i / mean(IS)). Features: delta_i and its local min/max. Boundary strength = prominence of the minimum. Breakpoint-induced boundary shifts show as new strong minima. Cost O(N w^2) naive; use summed-area table (2D cumsum of valid-mask and values) for O(1) per window.
2. Off-diagonal block enrichment (cross-window): for breakpoint candidate pair (i,j) with |i-j|>2w, K_ij = mean over the (2w+1)x(2w+1) box of L. Candidate score = K_ij vs null (robust z over all boxes at the same distance d). Practical: do this only for pixels in the top 0.1% of obs/exp, then cluster pixels (DBSCAN/connected components), like HiCCUPS peak calling.
3. Bow-tie / quadrant contrast at diagonal point i: Q_i = (mean UL + mean LR) - (mean UR + mean LL) where quadrants are w x w blocks around (i,i) offset along anti-diagonal. Translocation/inversion break gives strong |Q_i|. Compute using summed-area tables.
4. Row/column asymmetry (breakpoint side): A_i = (mean_{j in [i-w,i)} L_ij_right) - ... Concretely, for bin i compare upstream profile u_i(d) = L_{i,i-d} with downstream v_i(d) = L_{i,i+d}, d=1..D. Asymmetry = mean_d (u_i(d) - v_i(d)) / pooled MAD. A one-sided drop-off is a breakpoint marker (contacts truncate across the break).
5. Coverage/CNV: log2 of raw coverage relative to a rolling median (width ~ 50 bins) and ICE bias z. Do not remove this.
6. Anti-diagonal inversion score for candidate box [i,j]: S_inv = corr(L restricted to the box, anti-diagonal-reflected expectation) or simply mean(L along anti-diagonal strip |x+y-(i+j)| <= w) minus mean(L along the main diagonal strip of same size inside the box). Compute only for candidate boxes from steps 2-3.
7. Distal fraction: replace fixed local10 vs distal ratio with obs/exp-based: mean L over |i-j| > 10 bins vs expected zero.
Vectorise all of the above with scipy.ndimage / summed-area tables; the current Python loop is O(N * window) per bin and uses raw counts so distance decay dominates.

Opinion: the Isolation Forest on 8 global bin descriptors is a weak detector for SV. The strongest published detectors are supervised (EagleC, CNN) or breakpoint-pair callers. Position this tool as unsupervised, calibrated, and fast, and report pair-level calls (bin_i, bin_j, type, score, p/q) rather than only per-bin flags.

## 4. Ensemble scoring and thresholds without fixed contamination

- Components: (a) IsolationForest score, (b) max |robust z| across feature columns, (c) pixel-level Poisson/NB q-value from section 2.
- Robust z: z = 0.6745 (x - median)/MAD (already in code). Combine by rank-average or Fisher/Stouffer on one-sided p-values. Rank-average is simple and scale-free.
- Threshold selection options (pick one, report clearly):
  1. Empirical null via permutation/simulation: shuffle bins within each distance-matched stratum (or resample counts from NB with mean mu_ij from the fitted expected model), recompute features and scores for n=100-200 replicates, take score threshold such that FWER or FDR on the null <= alpha. Costly but defensible. Opinion: do on a subsampled chromosome or at coarse bin size.
  2. Null from the data itself: fit a Gaussian (median, MAD) to the bulk of the score distribution (central 90%), then p = 1 - Phi(z), then BH-FDR. Cheap, assumes mostly-null bins (true for most chromosomes).
  3. Extreme-value / POT: fit generalized Pareto to top 10% of scores; flag where tail p < alpha.
- Remove `contamination`. Use `contamination="auto"` in sklearn and threshold on the score from the null procedure above. Report "0 hits" legitimately in clean data.
- Replicate/consistency: require support from at least 2 of 3 components, and merge adjacent bins (existing merge function) then recompute region-level p (min p with Bonferroni on number of merged bins, or Simes).
- Reference: BH 1995 https://www.jstor.org/stable/2346101 ; sklearn IsolationForest https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

## 5. Allelic imbalance test

Setup per region: k = reads (or contacts) on haplotype A, n = total phased, p0 = expected (0.5; or reference-bias-corrected, e.g. 0.5 modified by a genome-wide estimate).

- Binomial: p = 2 * min(P(X<=k), P(X>=k)) or scipy.stats.binomtest(k, n, p0). Problem: real data is overdispersed (mapping bias, phasing errors, local CNV), so binomial gives inflated significance, especially at high n.
- Beta-binomial: k ~ BetaBin(n, alpha, beta) with mean mu = alpha/(alpha+beta) = p0 and overdispersion rho = 1/(alpha+beta+1). Under H0 mu=p0 fixed, rho estimated genome-wide from null regions.
- Estimating rho:
  - Method of moments on regions with n>=n_min: with f = k/n, rho_hat = (n * Var(f) / (mu(1-mu)) - 1)/(n - 1) averaged (weighted); more robust: fit rho by maximum likelihood across all regions (scipy.optimize.minimize on sum of betabinom.logpmf with mu fixed or free), using scipy.stats.betabinom (SciPy >= 1.4).
  - Use only the bulk (trim top/bottom 5% of |f-0.5|) to avoid estimating rho from the signal itself.
  - Alternative: quasi-binomial dispersion phi = X^2/(df), scale variance by phi (simpler, less exact).
- Test: two-sided p = sum of pmf over outcomes with pmf <= pmf(k) (exact), or 2*min tail. Use betabinom.cdf/sf with discrete correction.
- Multiple testing: Benjamini-Hochberg, q_i = min_{j>=i} (m/j) p_(j). Use scipy.stats.false_discovery_control(p, method="bh") (SciPy >= 1.11) or statsmodels.stats.multitest.multipletests(method="fdr_bh"). Run over ALL tested regions (not only the anomaly candidates) when reporting a genome-wide FDR; for the candidate-only test, report that the family is the candidate set.
- Report effect size too: log2((k+0.5)/(n-k+0.5)) with Wilson/Jeffreys CI, and require |effect| > 0.2 as an optional filter.
- References: scipy betabinom https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html ; WASP allelic mapping bias https://www.nature.com/articles/nmeth.3582 ; MBASED/overdispersion practice https://doi.org/10.1186/s13059-014-0405-3 (Mayba 2014, verify)

## 6. Fast streaming BAM pair counting (pysam)

Problems today: dict of unmatched reads grows without bound (name-grouped input is assumed but not guaranteed); `fetch(chrom)` coordinate-sorted means mates are far apart, so the dict holds many entries.

Options:
- A. Coordinate-sorted + proper mate use: count a pair once from the leftmost read only. For each read with `read.is_read1`... better: use `reference_start <= next_reference_start` rule (ties by `is_read1`), and use `read.next_reference_start` and `read.reference_start` directly, no dict at all. This works because both mates' positions are in each record's header (RNEXT/PNEXT). Memory O(1). Duplicate flags and MAPQ of the OTHER mate are not known from a single record: use `MQ` tag if present (bwa mem/samtools fixmate -m adds `MQ` and `ms`), or require the read's own MAPQ only. Opinion: this is the best default for Hi-C, where mates are often at different loci (cis-long-range) but all fields are present.
- B. Name-sorted (`samtools sort -n`): mates adjacent, read pairs by `next()` iteration, exact MAPQ on both mates, O(1) memory. Cost: a name sort of a large BAM (tens of GB, minutes to hours) and no region-restricted `fetch`. Good for a one-time ingest to a sparse store.
- C. pairtools (https://github.com/open2c/pairtools): `parse` gives .pairs (with walks/chimeric handling, MAPQ filters), `dedup`, then `cooler cload pairs`. This is the standard; ICE then via `cooler balance`. Opinion: offer a `--pairs`/`.cool`/`.mcool` input path to skip BAM entirely, and make pysam the fallback.
- Storage: accumulate (bin1, bin2) into numpy arrays in chunks (e.g., 5M pairs), then `np.unique`/`coo_matrix` merge chunk by chunk; or a `dict[(b1,b2)]->count` only for sparse. Final: scipy.sparse.coo_array -> csr (sum duplicates). Avoid `np.zeros((N,N))`.
- Speed: per-read Python loop ~ 1-3 M reads/s is not achievable; expect ~200-500k reads/s with pysam. Mitigate with `bam.fetch(until_eof=True)`, multiple processes per contig (`multiprocessing` with one region each, merging COO), or `bam.fetch(chrom)` with `threads=4` in AlignmentFile (htslib decompression threads: `pysam.AlignmentFile(path, "rb", threads=4)`).
- Filters: MAPQ >= 30 is the 4DN pipeline standard (https://data.4dnucleome.org/resources/data-analysis/hi_c-processing-pipeline); MAPQ >= 1 used in HiC-Pro default with unique mapping (https://github.com/nservant/HiC-Pro). Recommend default 30, flag `--min-mapq`. Also: skip `is_duplicate`, secondary/supp, optionally discard pairs with separation < 1 kb (self-circle / dangling ends), and, for strong reproducibility, remove same-strand-short-range artefacts (distance < 20 kb for FR / inward pairs, see HiC-Pro/ Hi-C QC).
- Keep both: cis counts matrix and trans (inter-chrom) counts per chrom pair, needed for translocations (current code discards trans: l.202).

## 7. Competing tools and public data

| Tool | Purpose | Notes |
|---|---|---|
| hic_breakfinder (Dixon 2018) | SV (translocation/inversion/deletion) from Hi-C BAM | https://github.com/dixonlab/hic_breakfinder ; C++; needs expected-matrix files (inter/intra) from the repo (download); BAM input |
| EagleC (Wang 2022) | SV calling via deep learning on .mcool | https://github.com/XiaoTaoWang/EagleC ; pretrained models downloaded with pip/conda |
| HiNT (Wang 2021) | CNV + translocation | https://github.com/parklab/HiNT ; Python, hg19/hg38 reference downloads |
| NeoLoopFinder (Wang 2021) | neoloops after SV | https://github.com/XiaoTaoWang/NeoLoopFinder ; needs SV list; downstream, not an SV caller itself (it has CNV/SV-breakpoint detection too) |
| HiCExplorer | Matrix tools, TAD insulation, `hicCorrectMatrix`, plots | https://github.com/deeptools/HiCExplorer ; useful for ICE/insulation sanity comparison |
| cooltools | expected, insulation, dots | https://github.com/open2c/cooltools ; use as correctness oracle for sections 1-3 |
| HiCtrans (Chakraborty 2017) / Hi-C breakfinder alternatives | translocation detection | verify availability before listing |

Runnable offline (after pip/conda install): cooltools, cooler (on a local .cool), HiCExplorer, pairtools, scipy/sklearn parts. Need download: EagleC models, hic_breakfinder expected files, NeoLoopFinder/HiNT references.

Opinion on benchmarking: (1) Unit-level oracle: compare our ICE weights to `cooler balance` (Pearson r > 0.999 on bias), our expected to `cooltools.expected_cis` (max rel error < 1e-3 on unsmoothed). (2) Synthetic ground truth: take a clean matrix (GM12878) and inject SVs by in-silico rearrangement of reads (swap/invert a segment in the BAM coordinates) or by modifying sparse counts according to known breakpoints (as in the EagleC and HiNT papers). Score by precision/recall at breakpoint within +-2 bins, FPR on clean data (this tests the no-fixed-contamination claim). (3) Real SV data: cancer lines with known SVs.

Datasets:
- K562, HeLa, A549, SK-N-MC, Panc1, T47D, LNCaP, and others with published translocation lists: used in Dixon 2018 (GEO GSE63525 for GM12878; SV Hi-C in GEO GSE97...: verify accession), K562 translocation BCR-ABL1 (chr9/22) is the classic positive control. Rao 2014 GM12878/K562: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 (download; large, tens to hundreds of GB raw).
- 4DN data portal https://data.4dnucleome.org/ (download, mcool/pairs available; smaller .pairs/.mcool preferable). 4DN Hi-C processed files for H1-hESC/HFFc6 are standard references.
- ENCODE Hi-C https://www.encodeproject.org/ (download; processed hic/mcool).
- HG002 / Genome in a Bottle phased Hi-C for allelic imbalance: GIAB HG002 data https://github.com/genome-in-a-bottle/giab_data_indexes ; HG002 Hi-C from HPRC/ Human Pangenome https://humanpangenomics.org/ ; phased truth VCF from GIAB (download). GM12878 has phased variants from Platinum Genomes / 1000G trios (https://www.internationalgenome.org/) (download). Verify accessions before commit.
- Small offline fixture for CI: create a synthetic tiny BAM (pysam writes it) with a planted deletion/inversion/translocation, and a small chr21 downsample (e.g., 5 M pairs) stored under tests/data (a few MB) for pytest.

## Concrete implementation order

1. Replace dense matrix with `scipy.sparse.coo_array` built from chunked arrays; add `--min-mapq` default 30; coordinate-sorted dict-free pairing via `next_reference_start`.
2. Add `ice_balance()` and `expected_cis()` with tests vs cooler/cooltools on small `.cool`.
3. Add vectorised features (insulation, quadrant contrast, asymmetry, CNV z) on obs/exp.
4. Replace contamination with null-calibrated threshold + BH q-values; output `q_value` column.
5. Beta-binomial test with genome-wide rho in `cross_refrence_hits.py`; BH over all tested regions.
6. Synthetic-SV benchmark harness comparing to hic_breakfinder/EagleC; publish precision/recall + runtime + peak RAM.

## Risks

- ICE biased by CNV; masking too aggressively removes true SV sites. Mitigated by keeping coverage feature.
- Per-diagonal expected computed from SV-containing data can be slightly biased; trimmed mean mitigates.
- Single-record pairing (option A) cannot see the mate's MAPQ without the `MQ` tag; document it.
- Overdispersion estimated genome-wide may mismatch local regions (CNV/phasing error); consider local-window estimation or require min n.
- Synthetic benchmark may overstate performance versus real data; include at least one real cell line with orthogonal SV truth (WGS-based).
- Accessions and tool versions above are from memory; I did not fetch pages this session. Verify each URL before publishing in README.
