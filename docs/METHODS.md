# Methods

This document describes what the v2 pipeline (`--method v2`, the default) computes, with the equations as implemented in `hic_anomaly/matrix.py`, `hic_anomaly/detect.py`, `hic_anomaly/crossref.py` and `hic_anomaly/stats_utils.py`. Where a choice is a heuristic or is validated only on simulated data, that is stated. The README gives the user-facing summary. The accuracy numbers are in `benchmarks/results/heldout_bench.md`, from a simulator the detector was not originally tuned on.

Notation: $O_{ij}$ is the raw contact count between bins $i$ and $j$ (symmetric, $N$ bins per chromosome); $B_{ij}$ is the balanced count; $d = |i - j|$.

## 1. Read filtering and pairing (`build_contact_matrix_sparse`)

A read pair enters the cis matrix only if both records pass these filters, in order (first failure is tallied in `run_info.json`):

1. not unmapped and mate not unmapped (SAM flags 0x4, 0x8);
2. not secondary, supplementary or duplicate (SAM flags 0x100, 0x800, 0x400);
3. paired (default mode only; in strict mode an unpaired read never pairs and is tallied as `orphan`);
4. MAPQ $\geq$ `--min-mapq` (default 30);
5. both mates on the same reference (trans pairs are tallied, not binned).

**Default pairing** (`strict_pairs=False`) is dict-free and name-free. Each pair is counted once, from its leftmost record, using the mate position stored in that record (RNEXT/PNEXT). Ties at equal position are broken by READ1. Pairs are accumulated as sorted `(bin_lo, bin_hi)` keys in chunks of one million pairs and reduced with `np.unique`, so memory is proportional to the number of nonzero bin pairs, not $N^2$.

Limits of the default mode: the mate's own flags are not seen; the mate's MAPQ is checked only through the `MQ` tag (`samtools fixmate -m`, bwa mem + fixmate, Picard). If the tag is absent, only the leftmost mate's MAPQ is tested, and the number of such pairs is logged and stored as `no_mq_tag`.

**Strict pairing** (`--strict-pairs`) joins mates by read name and counts a pair only when both records pass every filter on their own flags. It keeps a pending-mate table (a dict keyed by read name). A record whose mate lies upstream and was not kept is discarded at once. The table therefore holds only pairs spanning the current position. Its peak size is stored as `max_pending`. This mode is exact but is not dict-free.

## 2. Iterative correction (`ice_balance`)

Model: $O_{ij} \approx b_i b_j T_{ij}$, where $T$ is the unobserved true contact frequency. Iterative correction (Imakaev et al., 2012) finds per-bin weights $w_i$ such that the balanced matrix $B_{ij} = O_{ij} / (w_i w_j)$ has equal row sums.

Procedure as implemented:

- Entries with $|i - j| <$ `--ignore-diags` (default 2) are dropped before balancing.
- Bins are masked (excluded and given NaN bias) when any of these hold:
  - no coverage;
  - fewer stored nonzeros than $\min(10, \max(1, \lfloor 0.25 \cdot \mathrm{median\ nnz} \rfloor))$;
  - low coverage: $\log_{10} c_i < \min(m - 5s,\; m - \log_{10} 3)$, where $m$ is the median of $\log_{10} c_i$ over covered bins and $s$ the MAD, floored at the value Poisson noise implies at the median depth. The fold guard means a bin is masked only if its coverage is also more than 3x below the median.
- Iteration, starting from $w = 1$, with $s_i = \sum_j O_{ij} / (w_i w_j)$ and $s_i \leftarrow s_i / \overline{s}$ (mean over unmasked bins): $w_i \leftarrow w_i s_i$. Stop when $\max_{i\,\text{unmasked}} |s_i - 1| < 10^{-5}$ or after 200 iterations.
- Output scaling (`rescale="counts"`, default): the balanced total equals the raw total of the same entries, so values remain on the read-count scale used by the pseudo-count in section 4. The returned bias is $1/w_i$ (NaN for masked bins).

Assumption: equal visibility of bins. Copy-number changes violate this. Coverage is therefore kept as a separate signal (section 6) rather than removed.

## 3. Distance-decay expected model (`expected_by_diagonal`)

For each diagonal $d$, over valid (unmasked) pairs with $|i-j| = d$, the expected value is a 2%-trimmed mean that includes zeros:

$$E_d^{\text{raw}} = \frac{1}{n_d - 2t_d} \sum_{k=t_d+1}^{n_d - t_d} x_{(k)}$$

where $n_d$ is the number of valid positions on diagonal $d$ (zeros included), $t_d = \lfloor 0.02\, n_d \rfloor$, and $x_{(k)}$ are the sorted values. The trimmed mean keeps a few strong events from inflating the expectation. The raw series is smoothed with a moving average of width 5 (clamped to $N$), normalized by the count of valid diagonal entries in the window.

## 4. Log2 observed/expected (`obs_over_exp`)

$$L_{ij} = \log_2\!\frac{B_{ij} + 1}{E_{|i-j|} + 1}$$

The pseudo-count of 1 is on the balanced read-count scale (section 2). Masked bins are NaN. With `--no-oe` the map is $\log_2(1 + B_{ij})$, and p-values are then not calibrated (the CLI warns).

## 5. Residual bias removal and pair statistics (`detect_anomalies`)

**Median polish.** Values are centered by a distance-band median (up to 12 log-spaced bands from $d \geq k_{\min} = 2 w_{\min} + 1$). A symmetric two-way median polish (4 iterations) then estimates a per-bin effect $r_i$, and the residual map is

$$R_{ij} = L_{ij} - r_i - r_j.$$

The row effect is the per-bin signal that is constant along a row, so it absorbs residual bias (imperfect balancing, copy number). Consequence: copy-number-like events are not visible to the pair track and are called from the coverage track (section 6).

**Feature maps** (window half-widths $w \in \{w_1, w_2\}$, default 5 and 10 bins; $\overline{R}$ is the mean over a box, computed with summed-area tables):

- block: $K_{ij}(w) = \overline{R}$ over the $(2w+1)^2$ box centered at $(i,j)$;
- bow-tie: $Q_{ij}(w) = (\overline{R}_{q_1} + \overline{R}_{q_2}) - (\overline{R}_{q_3} + \overline{R}_{q_4})$, where $q_1$ and $q_2$ are the upper-left and lower-right $w \times w$ quadrants at the corner and $q_3$, $q_4$ the upper-right and lower-left;
- anti-diagonal contrast: mean of $R$ along the anti-diagonal strip minus mean along the main-diagonal strip, both inside a $(2w+1)$ box.

Each feature is standardized within its distance band by the band median and $\sigma = 1.4826 \cdot \mathrm{MAD}$ (bands with fewer than 10 values are left unscaled):

$$z_k(i,j) = \frac{F_k(i,j) - \mathrm{med}_{b(d)}}{\sigma_{b(d)}}.$$

**Combined score.** An Isolation Forest (100 trees, `max_samples=256`, fitted on up to 20,000 valid pairs drawn with `--seed`) gives a score $s_{ij}$. With $z^{\max}_{ij}$ the largest block z and absolute bow-tie z:

$$T_{ij} = \tfrac{1}{2}\Big( z^{\max}_{ij} + \frac{s_{ij} - \mathrm{med}(s)}{\sigma_{\mathrm{MAD}}(s)} \Big).$$

Only pairs with $i \geq e$ and $j < N - e$ are scored, where $e = w_{\max}$ (default 10 bins). Events within $e$ bins of a chromosome end are not tested. A map needs at least $2 w_{\min} + 1 + 2 w_{\max} + 2$ bins (33 with the default windows) and at least 50 valid pairs.

**Peaks and breakpoints.** Non-maximum suppression in a $(2w_{\max}+1)^2$ window gives one peak per local maximum of $T$. Each peak is moved to the bow-tie corner within $2 w_{\max}$ bins if that corner's $|z|$ exceeds the peak's by at least 1.0 and $T$ at the corner is at least a floor (the map median plus 10% of the peak's excess over it). Peaks that collapse onto the same corner are merged, keeping the smallest p.

**Call-level null and p-values.** The null is fitted to the population of all peak heights in the same map, not to a fixed constant. At least 5 peaks are needed.

1. Gumbel fitted by quantile matching on the interquartile range: $\beta = \mathrm{IQR}/1.5725$, $\mu = \mathrm{med} + 0.3665\,\beta$, $p_1 = 1 - \exp(-\exp(-(T - \mu)/\beta))$.
2. If at least 60 peaks exist, type-II-censored exponential tails are fitted to exceedances over the 60th, 75th and 88th percentiles. The top 10% of exceedances (and at least the 4 largest) are treated as censored events.
3. Empirical rank p in the bulk: for peaks at or below the map median, $p_{\text{rank}} = (1 + \#\{\text{null} \geq t\})/(1 + n)$, where the null is the set of all peak heights. It never equals 0.

The reported peak p is the pointwise maximum of these estimators (the most conservative). The q-value is Benjamini-Hochberg over peaks within the chromosome. A pair call is reported at $q \leq$ `--fdr` (default 0.05).

**Pair types.** Quadrant masses at the breakpoint (mean of $L$ in each box, at the two window sizes, averaged) assign a type heuristically. The rules are applied in this order:

- translocation-like/butterfly: $\min(q_3, q_4) \geq 0.5\, \max_k q_k$ and $\min(q_3, q_4) > 0$;
- inversion: $\min(q_1, q_2) \geq 0.5\, \max_k q_k$, $\min(q_1, q_2) > 0$ and $\min(q_1, q_2) \geq \min(q_3, q_4)$. Inversion overrides butterfly on ties.
- deletion/dup: the default label when neither pattern dominates.

The mapping from quadrant pattern to event type was derived from simulation and has not been checked on real data. The label `deletion/dup` does not separate deletions from duplications. Use the coverage track and the raw map to decide.

**Pixel-level p-values** (`p_pix`) are computed for diagnostics only and do not drive calls.

## 6. Coverage (CNV) segments (`_cnv_calls`)

**Coverage track.** $c_i$ is the raw row sum of the screening matrix (before balancing). The signal is $y_i = \log_2 c_i$ over unmasked bins. A bin is masked when fewer than 20% of its map entries are finite (ICE-masked or empty), or when its coverage is zero.

**Noise scale.** The sigma is

$$\sigma = \frac{1.4826\,\mathrm{MAD}(\Delta y)}{\sqrt{2}},$$

where $\Delta y$ are the finite first differences of $y$. The $\sqrt{2}$ converts the variance of a difference to the variance of one bin.

**Outlier handling.** Bins more than $4\sigma$ from a 9-bin running median are treated as missing. The rest are winsorized at $\pm 2.5\sigma$ around that running median, so isolated dropouts carry bounded or no weight, while a genuine large shift is kept.

**Scan.** For each window scale $w \in \{6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256\}$ with $w \leq \min(256,\, n - 2e - 2 \cdot 16 - 10)$, where $e = \max(5, \min(15, \lfloor n/10 \rfloor))$ is the edge margin, the window mean is compared with flanks of up to 16 bins on each side. The flank width is $\min(w, 16)$. A window needs at least 70% valid bins and at least 60% valid bins in each flank. A call needs both edges to step in the same direction (intersection-union rule: the statistic is the signed minimum of the two edge z-scores).

Each scale's z is divided by a robust empirical null $\max(1, 1.4826\,\mathrm{med}|z|)$, computed over that scale's valid windows. For scales above 16 bins, the null is capped at the 16-bin value times $\sqrt{w/16}$, which is the bound for perfectly correlated bins. This stops a large event from inflating its own null.

**Peaks and p-values.** Candidates are local maxima of $|z|$ along position for each scale, then one representative per event, chosen greedily by height so that segments do not overlap. The per-peak p-values are:

$$p_{\text{scan}} = \min\!\big(1,\; 2\,\mathrm{sf}(|z|)\, m_{\text{eff}}\big), \qquad m_{\text{eff}} = \sum_{\text{scales}} \frac{\#\text{valid windows}}{\max(2,\, w/2)},$$

where $\mathrm{sf}$ is the standard normal survival function. This is a Bonferroni-type scan adjustment for the approximately independent (window, scale) tests. It is always computed.

- If at least 12 peaks exist: $p = \max(p_{\text{peak-null}}, p_{\text{scan}})$, where $p_{\text{peak-null}}$ comes from the same Gumbel and censored-tail null as section 5.
- If fewer than 12 peaks exist: $p = p_{\text{scan}}$ alone.

BH is applied over all CNV peaks of the chromosome. A segment is reported at $q \leq$ `--fdr` when it has at least 6 bins, at least 70% valid bins, and a shift $|d| \geq 0.3$ log2 at both edges (the smaller of the two edge shifts). Boundaries are then refined by maximizing a standardized step statistic near each end, subject to the minimum length. The reported p stays the scan window's p.

**Homozygous-deletion candidates.** Masked runs are first bridged across single unmasked bins. A run $m$ bins long qualifies when all of these hold:

- $m \geq 5$ masked bins, and at least 90% of its span is masked;
- the run is at least 5 bins from either chromosome end;
- within the 5 bins on each side, the masked fraction is at most 0.4.

Each qualifying run gets

$$p = \min\!\big(1,\; (n - m + 1)\,(1 - p_0)\,p_0^{\,m}\big),$$

where $n$ is the number of bins and $p_0$ is the fraction of all bins that lie in masked runs shorter than 5 bins (floor $1/n$, cap 0.5). This is the per-bin masking rate under independent masking. Independence is optimistic for clustered masks (repeats), so these are candidates only. BH is applied over the candidates, and each candidate with $q \leq$ `--fdr` is reported with score $-\log_{10} p$. They are never reported as pair calls.

**Assumptions.** Most bins are copy-neutral. Slow coverage bias is not modeled (pass bias-corrected coverage if available). Segments covering more than about a third of the map inflate the null and are under-called. Events closer than about 16 bins share flanks.

## 7. Multiple testing

All BH adjustment is per chromosome and per test family: pair peaks, CNV peaks, and homozygous-deletion candidates. There is no genome-wide FDR. When several chromosomes are run together, a q-value of 0.05 means that 5% of calls on that chromosome are expected to be null under the model, not 5% of all calls in the run.

## 8. Allelic-imbalance cross-reference (`cross_reference`)

Input is a per-bin haplotype count table with `chrom, start, end, hap1_count, hap2_count` (or an `imbalance` column). Candidate regions come from the detector output, or any table with `chrom, start, end` (or `region`).

For a candidate region overlapping $m$ bins (padded by `--buffer`, default 250 kb):

$$k = \sum_b h^{(1)}_b, \qquad n = \sum_b \big(h^{(1)}_b + h^{(2)}_b\big)$$

Under the null $k \sim \mathrm{BetaBinomial}(n, \mu, \rho)$, with $\mu$ the null mean and $\rho$ the overdispersion. The p-value is the exact two-sided beta-binomial probability (sum of pmf values no larger than the observed one). For $n > 20\,000$, a doubled-tail approximation is used.

- $\rho$ is estimated at the region scale. Sliding windows of $m$ bins are pooled over each chromosome, and $\rho$ is fitted by median matching (`estimate_rho_median`), so correlated neighboring bins are represented in the null. If there are fewer than 20 windows, the independent-bin estimate is used (method of moments, `estimate_rho_mom`, rescaled for the sum). A fixed `--rho` also uses the independent-bin path.
- $\mu$ is the median window fraction (`--center auto`, default; an empirical null center that absorbs genome-wide reference bias), or `--p0` (default 0.5; `--center fixed`). With `--independent-bins` or a fixed `--rho`, $\mu$ is `--p0`, and `--center` has no effect (a warning is emitted for `--independent-bins` when `--center` is left at `auto`).
- $q$ is Benjamini-Hochberg over all candidate regions. `q_all` pools the candidate p-values with the per-bin p-values.
- Effect: $\log_2\frac{k + 0.5}{n - k + 0.5}$ with a Jeffreys 95% interval.
- `cross_validated` is true when the mean absolute per-bin imbalance $\geq$ `--min_imbalance` (default 0.15 in the library and the CLI) and $q \leq$ `--fdr` (default 0.05).
- If the input has no `hap1_count`/`hap2_count` columns, no p-value can be computed. p and q are NaN, a warning is emitted, and `cross_validated` is false for every region.

Assumptions: bins are not fully independent (handled only at the window scale). The window $\rho$ is estimated from the same data and is noisy with few windows. A genuine chromosome-wide imbalance is absorbed by the auto center. The p-values are not conservative for structure correlated at scales larger than the region. No causal interpretation is implied.

## 9. Legacy method (`--method legacy`)

The original pipeline. It builds a dense matrix of raw counts with name-based pairing (both mates mapped, primary, non-duplicate, cis) and no MAPQ filter. It has no bin cap. It computes eight per-bin features:

1. $\log(1 + \text{total})$, the row sum;
2. $\log(1 + \text{local}_5)$, the sum within $\pm 5$ bins;
3. $\log(1 + \text{local}_{10})$, the sum within $\pm 10$ bins;
4. $\log(1 + \text{distal})$, where distal = total minus local$_{10}$ (clipped at 0);
5. $\log(1 + \text{diagonal})$, the diagonal entry;
6. $\log(1 + \text{local}_{10}/(\text{distal} + 1))$;
7. robust z of the total, $0.6745\,(x - \mathrm{med})/\mathrm{MAD}$;
8. robust z of local$_{10}$.

An Isolation Forest (300 trees, fitted on bins with nonzero total, `random_state` from `--seed`) uses a fixed `--contamination` (default 0.01). Flagged bins are merged when separated by at most one unflagged bin. The method flags about 1% of bins regardless of the data. It is kept so earlier results can be reproduced; it is not calibrated.

## 10. What is not implemented

- KR or other balancing methods; ICE only.
- Trans (inter-chromosomal) calls. Trans counts can be computed (`build_contact_matrix_sparse(keep_trans=True)`), but the detector runs on cis maps only.
- Genome-wide FDR across chromosomes.
- Input from `.pairs`, `.cool` or `.mcool` files.
- Validation on real Hi-C data with orthogonal SV truth. See the README limitations.

## References

- Imakaev M, et al. (2012). Iterative correction of Hi-C data reveals hallmarks of chromosome organization. *Nature Methods*.
- Lieberman-Aiden E, et al. (2009). Comprehensive mapping of long-range interactions reveals folding principles of the human genome. *Science*.
- Rao SSP, et al. (2014). A 3D map of the human genome at kilobase resolution. *Cell*.
- Dixon JR, et al. (2018). Integrative detection and analysis of structural variation in cancer genomes. *Nature Genetics*.
- Liu FT, Ting KM, Zhou ZH (2008). Isolation forest. *ICDM*.
- Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. *JRSS B*.

References were cited from the methods notes in `.agent-reports/research/methods.md` and were not re-checked against the publisher pages. Verify volume and page details before publication.
