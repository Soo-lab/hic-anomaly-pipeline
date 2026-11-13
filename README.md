# hic-anomaly-pipeline

Unsupervised detection of candidate structural variants and copy-number segments in Hi-C contact maps, with empirical-null q-values and an allelic-imbalance cross-check.

> **Status: v0.2.0, development branch (`upgrade/full-overhaul`). Research software.** Every accuracy number in this repository comes from simulated maps. No real-data validation exists yet. Read [Limitations](#limitations) before using any call for anything.

## Why this tool

The original pipeline in this repository is now `--method legacy`. It flags about 1% of bins whatever the data looks like. The default `--method v2` was compared with it on a second simulator (504 maps; [`benchmarks/results/heldout_bench.md`](benchmarks/results/heldout_bench.md), Pooled and Clean totals). Simulated data only; see [Accuracy](#accuracy-simulated-data-only).

| | legacy | v2 |
|---|---|---|
| Clean maps (no events): calls | 3.26 regions per map | 0.019 calls per map (2 of 108 maps have any call) |
| 1D F1 (pair calls, one breakpoint within 2 bins) | 0.068 | 0.577 |
| Mean localization error (bins) | 4.34 | 0.77 |
| Pair-level (2D) F1 | not defined | 0.510 |

Matrix construction is also leaner. It is sparse and applies a MAPQ filter. In the default mode, pairing is dict-free: each pair is counted once from its leftmost record, using mate coordinates. `--strict-pairs` joins mates by read name and keeps a bounded pending-mate table instead. At the largest synthetic matrix benchmark, peak Python allocation was about 55x lower than the dict-based dense baseline (29.3 MB vs 1605.2 MB; see [Performance](#performance)).

## What it does

```
BAM / SAM / FASTQ ──► sparse contact matrix (MAPQ >= 30, leftmost-mate pairing)
                  ──► ICE balancing (Imakaev 2012) ──► distance-decay expected
                  ──► log2 observed/expected map ──► pair-level SV-like calls + CNV segments
                  ──► BH q-values per chromosome ──► TSV/CSV, PNG figures, optional HTML report
                  ──► allelic-imbalance cross-reference (beta-binomial, BH over regions)
```

Pair calls are typed as `deletion/dup`, `inversion`, `translocation-like/butterfly`, or `cnv` (coverage segment). A fifth label, `homozygous-deletion-candidate`, marks a long masked run (a gap, centromere, or deletion candidate). It is never a pair call.

Calls are statistical outliers in local contact structure. They are consistent with a rearrangement or copy-number change, but they are not validated structural variants. The method is described in [docs/METHODS.md](docs/METHODS.md).

## Install

Requires Python 3.9 to 3.12. Runtime dependencies: `pysam`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`. `samtools` is needed for SAM input. `bwa` and `samtools` are needed for FASTQ/FASTA input.

**pip (from a checkout):**

```bash
git clone https://github.com/Soo-lab/hic-anomaly-pipeline.git
cd hic-anomaly-pipeline
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"        # runtime + pytest, ruff, pre-commit
# runtime only: pip install -r requirements.txt && pip install -e .
```

This repository does not reference a PyPI release. There is no `numba` dependency.

**conda:**

```bash
conda env create -f environment.yml   # includes bwa and samtools
conda activate hic-anomaly
```

**Docker** (image includes bwa and samtools; entrypoint is `hic-anomaly`, working directory `/data`):

```bash
docker build -t hic-anomaly .
docker run --rm -v "$PWD":/data hic-anomaly --input sample.bam --chrom chr21 --outdir results
```

The conda and Docker definitions have not been built or solved in the environment where this README was written. CI (`.github/workflows/ci.yml`) runs lint on Python 3.12 and tests on Python 3.9 to 3.12, with pip only.

## Quick start

Demo on a planted synthetic BAM: 20 Mb chr1, 100k pairs, an off-diagonal block joining 5-6 Mb with 12-13 Mb, and a coverage gain (x3) at 15-16 Mb.

```bash
mkdir -p demo
python -c "import sys; sys.path.insert(0, 'tests'); from synth_planted import make_planted_bam; make_planted_bam('demo/planted.bam')"
hic-anomaly --input demo/planted.bam --chrom chr1 --screen-bin 100000 --zoom-bin 25000 --outdir demo/out --report
```

Then open `demo/out/report.html`, and read `demo/out/anomaly_calls.csv`.

Expected output, from runs of this demo on 2026-11-13 (no reference output is checked in): 5 calls (4 `cnv`, 1 `deletion/dup`). The off-diagonal block is reported as a `deletion/dup` call at its corner, and the coverage gain gives `cnv` calls. Two other settings show the resolution limit:

- `--screen-bin 250000` gives 0 calls. The chromosome then has 80 bins, the block spans about 4x4 bins, and the CNV spans 4-5 bins, which is below the minimum segment length of 6 bins.
- `--screen-bin 10000` gives 25 calls (3 `cnv`, 7 `inversion`, 7 `translocation-like/butterfly`, 8 `deletion/dup`). The 8 `deletion/dup` calls span 3-19 Mb with q-values of 0.02-0.05 and are probably noise peaks on a very sparse map (100k pairs). This is a documented limitation (see [Limitations](#limitations)).

### Choosing --screen-bin

- Aim for at least 150 bins per chromosome. The CLI warns below 150 bins. v2 refuses more than 3000 bins per chromosome (exit code 2) and suggests a larger bin.
- Events must span several bins. A 1 Mb event needs a bin size well below 1 Mb. CNV segments need at least 6 bins.
- Finer bins give more bins but sparser maps. At 10 kb on about 100k pairs, most pixels hold zero or one contact, and noise peaks appear. Treat long-span, low-score calls on such maps with caution.
- Check the log warnings and `run_info.json` (`bins`, `masked_bins` per chromosome) before interpreting calls.

On your own data (coordinate-sorted BAM, mapped with a MAPQ tag, such as `samtools fixmate -m` output):

```bash
hic-anomaly --input sample.bam --chrom chr21 --screen-bin 100000 --zoom-bin 25000 \
  --outdir results --report
```

Choose `--screen-bin` so each chromosome is at most 3000 bins. `--chrom all` runs every contig in the header and skips contigs too short to score.

`python -m hic_anomaly ...` is equivalent to `hic-anomaly ...`. `universal_hic_pipeline.py` remains as a backward-compatible wrapper. It runs the same CLI, so it also defaults to `--method v2`. Add `--method legacy` to reproduce the original behavior.

## CLI reference

`hic-anomaly [options]`. Run `hic-anomaly --help` for the authoritative list.

**Input and output**

| Option | Default | Meaning |
|---|---|---|
| `--input` | required | BAM, SAM, or R1 FASTQ/FASTA (`.gz` ok). SAM is sorted with samtools. FASTQ/FASTA needs `--input2` and `--ref-fasta` and is aligned with bwa mem. |
| `--input2` | none | R2 mate file (FASTQ/FASTA only) |
| `--ref-fasta` | none | Reference FASTA (FASTQ/FASTA only; `bwa index` runs if the index is missing) |
| `--chrom` | required | Chromosome name (`chr1` or `1`). Repeat the flag or pass a comma-separated list. `all` runs every contig in the header. |
| `--outdir` | `universal_hic_output` | Output directory |
| `--threads` | 4 | Threads for bwa and samtools sort (SAM and FASTQ/FASTA input) |

**Method**

| Option | Default | Meaning |
|---|---|---|
| `--method {v2,legacy}` | `v2` | `v2`: balanced, distance-normalized detection with q-values. `legacy`: the original flow (dense name-paired counts, no MAPQ filter, Isolation Forest with fixed contamination). `legacy` writes `anomaly_candidates_<chrom>.tsv` instead of `anomaly_calls.csv`, and has no q-values. |
| `--screen-bin` | 1000000 | Whole-chromosome bin size (bp). v2 caps each chromosome at 3000 bins. Legacy has no cap and builds a dense matrix, so memory grows with bins squared. |
| `--zoom-bin` | 250000 | Zoom bin size (bp). v2 skips a zoom figure that would need more than 3000 bins. |
| `--max-zoom` | 5 | Zoom figures for at most this many calls per chromosome; 0 disables. v2 zoom figures count only pairs with both mates inside the zoom window (call plus 500 kb padding). |
| `--seed` | 42 | Random seed. Legacy: the Isolation Forest. v2: the Isolation Forest and its 20,000-pair fitting subsample. |

**v2 options** (ignored by `legacy`)

| Option | Default | Meaning |
|---|---|---|
| `--min-mapq` | 30 | Drop pairs with MAPQ below this value. The mate's MAPQ is checked only through the `MQ` tag. |
| `--strict-pairs` | off | Join mates by read name and filter both records (exact; memory bounded by the pending-mate table, not dict-free). |
| `--workers` | 4 | Processes for per-chromosome matrix builds |
| `--bam-threads` | auto | htslib decompression threads per reader. Auto is 1 per worker process, or up to 4 when running serially. |
| `--cache-dir` | none | Cache sparse matrices as `.npz`, keyed by input path, size and modification time, chromosome, bin size, MAPQ and pairing mode |
| `--balance / --no-balance` | on | ICE balancing |
| `--ignore-diags` | 2 | Near-diagonal diagonals excluded from balancing |
| `--oe / --no-oe` | on | Distance normalization (log2 obs/exp). Without it, p-values and q-values are not calibrated, and the CLI warns. |
| `--fdr` | 0.05 | Benjamini-Hochberg level, applied per chromosome and per test family |
| `--n-null` | 5 | Accepted for backward compatibility. Not used. |
| `--windows W1,W2` | 5,10 | Feature window half-widths (bins). The minimum detectable matrix size depends on these (33 bins with the default). |
| `--triangular` | off | Draw the screening heatmap as a rotated upper triangle |
| `--no-log-heatmap` | off | Raw counts instead of log(1 + contacts) |

**Legacy options** (ignored by `v2`)

| Option | Default | Meaning |
|---|---|---|
| `--contamination` | 0.01 | Isolation Forest contamination, in [0, 0.5). It forces about this fraction of bins to be flagged, whatever the data. |

**Reporting**

| Option | Meaning |
|---|---|
| `--report` | Write a self-contained `report.html` (figures inline, sortable call table) |
| `-v`, `--verbose` / `-q`, `--quiet` | Debug logging / warnings only |
| `--version` | Print the version |

**Exit codes**

- `0`: success.
- `1`: runtime or I/O error. Examples: an unreadable BAM, a missing `--input2` or `--ref-fasta` for FASTQ/FASTA input, an unsupported input extension, a missing `bwa` or `samtools`. A failed bwa or samtools step exits with that tool's own exit code, which need not be 1.
- `2`: usage error. Examples: argparse errors, including non-numeric or non-positive values such as `--screen-bin 0`; a chromosome not found in the BAM header; more than 3000 bins per chromosome in v2; or `--chrom all` when no contig has enough bins.

**Crossref** (`hic-anomaly-crossref`)

```bash
hic-anomaly-crossref --anomalies results/anomaly_calls.csv --imbalance haplo_bins.tsv --out xref.tsv
```

| Option | Default | Meaning |
|---|---|---|
| `--anomalies` | required | Calls CSV/TSV with `chrom,start,end` (or `start_bp,end_bp`, or `region`). `anomaly_calls.csv` and the legacy TSV both work. |
| `--imbalance` | required | Per-bin table with `chrom,start,end,hap1_count,hap2_count` (or an `imbalance` column) |
| `--out` | `cross_validated_anomalies.tsv` | Output TSV |
| `--min_imbalance` | 0.15 | Minimum mean absolute per-bin imbalance for `cross_validated`. Same default in the library function and the CLI. |
| `--buffer` | 250000 | bp padding around each call |
| `--p0` | 0.5 | Expected hap1 fraction |
| `--fdr` | 0.05 | BH level across candidate regions |
| `--rho` | estimated | Fix the beta-binomial overdispersion. The null mean then stays at `--p0`. |
| `--independent-bins` | off | Use the independent-bin null, which is anti-conservative if bins are correlated. `--center` is ignored in this mode, and a warning is emitted when `--center` is left at `auto`. |
| `--center {auto,fixed}` | auto | `auto`: null center is the median window fraction. `fixed`: test against `--p0`. |

The imbalance table is not produced by this repository. You need to supply it from your phased data. If the imbalance table has no `hap1_count`/`hap2_count` columns, p-values and q-values are NaN, a warning is emitted, and no region is validated.

## Outputs

**v2** (`--method v2`, default), all in `--outdir`:

| File | Content |
|---|---|
| `anomaly_calls.csv` | One row per call across chromosomes. Columns: `chrom, start, end, bin_i, bin_j, start_bp_i, start_bp_j, type, score, p_value, q_value` |
| `hic_<chrom>_screening.png` | Whole-chromosome contact heatmap with called regions marked |
| `hic_<chrom>_scores.png` | Per-bin anomaly score with threshold, and the coverage z-score track |
| `zoomed_regions/zoom_<chrom>_<NN>_<Mb>Mb_<zoom>kb.png` | Zoom heatmaps for the top calls (built from the BAM, MAPQ-filtered) |
| `run_info.json` | Command, versions, parameters, read accounting per chromosome (`total, unmapped, secondary_supp, duplicate, low_mapq, trans, unpaired, out_of_range, orphan, mate_second, used`, plus `max_pending` in strict mode), bins, masked bins, cache hits, call counts |
| `report.html` | With `--report`: parameters, figures and calls, self-contained |

Column meanings:

- `bin_i`, `bin_j`: bin indices of the two breakpoints for pair calls. For `cnv` calls, the first and last bins of the segment.
- `start`, `end`: genomic span. For pair calls, from `start_bp_i` to the end of bin `bin_j`.
- `type`: one of
  - `deletion/dup`: the default pair label when neither quadrant pattern dominates. It does not separate deletions from duplications.
  - `inversion`
  - `translocation-like/butterfly`
  - `cnv`: coverage segment, tested on the coverage track only.
  - `homozygous-deletion-candidate`: a masked run (gap, centromere, or deletion candidate). Not a pair call.
- `score`: pair calls, the combined z-type score; `cnv`, the absolute segment z; homozygous candidates, -log10 p.
- `p_value`, `q_value`: empirical-null p and Benjamini-Hochberg q within each chromosome and test family. Every reported call has `q_value <= --fdr`. CNV p-values are the larger of the peak-null p and a Bonferroni scan p (see METHODS).
- Calls are sorted by q within each chromosome. The combined file is not re-sorted across chromosomes.

**legacy** (`--method legacy`):

| File | Content |
|---|---|
| `anomaly_candidates_<chrom>.tsv` | Merged flagged regions: `chrom, start_bin, end_bin, start_bp, end_bp, start_Mb, end_Mb, size_Mb, max_ai_score` |
| `hic_<chrom>_screening.png` | Original-style heatmap |
| `zoomed_regions/` | Original-style zoom PNGs for the first `--max-zoom` regions. File names are `zoom_<NN>_<Mb>Mb_<zoom>kb.png`, with the chromosome prefixed when several chromosomes are run. |
| `run_info.json`, `report.html` | As above |

Legacy writes no `anomaly_calls.csv` and no q-values.

**Cross-reference** writes a TSV (default `cross_validated_anomalies.tsv`) with one row per candidate region: `chrom, start, end, overlap_bin_count, total_phased_reads, mean_imbalance, mean_abs_imbalance, max_abs_imbalance, skew_direction, z_score, p_value, q_value, q_all, log2_effect, ci_low, ci_high, rho, rho_region, p0_used, cross_validated`, plus the legacy t-test p-value `p_value_ttest` (not used for decisions). `rho_region`, `p0_used`, `p_value`, `log2_effect`, `ci_low` and `ci_high` are filled only when read counts are present.

## Method overview

The full description, with equations and assumptions, is in [docs/METHODS.md](docs/METHODS.md). In brief:

1. **Pairing.** Each pair is counted once, after filters on unmapped, secondary/supplementary, duplicate, and MAPQ below 30. The default mode counts each pair from its leftmost record. `--strict-pairs` requires both records to pass.
2. **Balancing.** Iterative correction, $O_{ij} \approx b_i b_j T_{ij}$, with $B_{ij} = O_{ij}/(w_i w_j)$, masking of extreme low-coverage bins, and rescaling to the count scale.
3. **Expected.** Per-diagonal 2%-trimmed mean $E_d$, moving-average smoothed.
4. **Observed/expected.** $L_{ij} = \log_2 \frac{B_{ij}+1}{E_{|i-j|}+1}$.
5. **Residual bias.** Two-way median polish, $R_{ij} = L_{ij} - r_i - r_j$.
6. **Pair features.** Box mean, bow-tie contrast and anti-diagonal contrast, standardized per distance band, combined with an Isolation Forest score into $T_{ij}$.
7. **Peaks and nulls.** Non-maximum suppression; a Gumbel and censored-exponential null fitted to the peak population; BH q-values per chromosome.
8. **Coverage.** Scan of log2 coverage for steps at both edges (CNV segments); homozygous-deletion candidates from long masked runs.
9. **Cross-reference.** Beta-binomial test of hap1 counts per candidate region, with overdispersion measured at the region scale and BH across regions.

## Performance

Benchmarks are on synthetic BAMs (uniform distance-decay background, plus planted events in the CLI test). Real BAMs will differ in size, depth, and read structure. Hardware is not recorded for the matrix table. The CLI table was measured on macOS arm64. Source notes are given under each table.

**Matrix build** (`benchmarks/bench_matrix.py`, run 2026-11-13; source: `.agent-reports/benchmarks.md`, section "bench_matrix 2026-11-13"). "Baseline" is the original dict-based dense builder, re-implemented in the benchmark. Memory is the tracemalloc peak of Python and NumPy allocations, not process RSS.

| pairs | chrom length | bin | baseline s | baseline peak MB | v2 s | v2 peak MB |
|---|---|---|---|---|---|---|
| 20,000 | 5 Mb | 10 kb | 0.25 | 1.3 | 0.17 | 1.6 |
| 100,000 | 50 Mb | 10 kb | 1.34 | 101.5 | 0.86 | 9.6 |
| 300,000 | 100 Mb | 5 kb | 4.80 | 1605.2 | 2.63 | 29.3 |

At the largest case, peak allocation is about 55x lower (1605.2 / 29.3) and build time about 1.8x lower. At the smallest case, peak allocation is slightly higher (1.6 vs 1.3 MB).

**End-to-end CLI** (`benchmarks/bench_validate_cli.py`, run 2026-11-13; source: `.agent-reports/benchmarks.md`, section "bench_validate_cli 2026-11-13"). Peak RSS of the CLI process, 20 Mb chr1 with planted events, screen bin 250 kb, zoom 50 kb.

| pairs | v2 wall s | v2 peak RSS MB | legacy wall s | legacy peak RSS MB | RSS reduction |
|---|---|---|---|---|---|
| 50,000 | 1.79 | 298 | 2.03 | 553 | 46% |
| 200,000 | 1.97 | 273 | 2.17 | 554 | 51% |
| 600,000 | 2.47 | 303 | 2.61 | 573 | 47% |

Wall time differs by less than 0.3 s at these sizes. The matrix build is a small share of runtime.

**Parallel matrix build** (`--workers`; source: `.agent-reports/benchmarks.md`, same section). On 4 contigs with 1.2M pairs and 50 kb bins, the matrix build took 1.02 s with 1 worker and 0.55 s with 4 workers (about 1.9x). This was measured outside a sandbox. End-to-end `--chrom all` on the same data took 2.32 s (1 worker) and 2.34 s (4 workers), so the speed-up is not visible end to end.

**Detection step** is slower than legacy, not faster. Source: `benchmarks/results/heldout_bench.md`, Pooled and By N tables (pooled per-map time in the benchmark harness; hardware not recorded). Pooled over 504 maps, v2 took 0.445 s per map and legacy 0.202 s. v2 time grows with bin count: 0.133, 0.341 and 0.860 s per map at N = 150, 300 and 500 bins.

## Accuracy (simulated data only)

**Simulated data, not real data.** The second simulator (`benchmarks/bench_heldout.py`) is a simulator the detector was not originally tuned on. The detector was revised in two rounds after failure analyses on earlier simulators (including this one), so the reported numbers are not a blind test. In the last round, tuning used separate development seeds in a scratch harness that is not checked in; the reported seeds were run once at the end. Criteria were not pre-registered. The numbers describe this simulator only, and a different simulator, or real Hi-C, would give different values.

The simulator uses negative-binomial counts, power-law or plateau distance decay, compartment-like background, per-bin bias and dropout, and ICE balancing before detection. It produces 504 maps (108 clean, 396 with events) and 445 planted events of four kinds (deletion, duplication, inversion, butterfly). Matching tolerance is ±2 bins. The detector seed is 7. Source: `benchmarks/results/heldout_bench.md`, Pooled table and header.

| Metric (pooled over all 504 maps) | legacy (1D) | v2 1D | v2 2D (pair) |
|---|---|---|---|
| Precision | 0.043 | 0.873 | 0.806 |
| Recall | 0.160 | 0.431 | 0.373 |
| F1 | 0.068 | 0.577 | 0.510 |
| Average precision | 0.031 | 0.584 | 0.506 |

- **1D**: a pair call matches an event if either breakpoint is within 2 bins of an event breakpoint. **2D**: both breakpoints must match. Legacy is region-level, so only 1D is defined for it.
- Call-level false-discovery proportion (q ≤ 0.05): **0.164** (single-event maps 0.173, mixed maps 0.100).
- Clean maps: 2 of 108 (1.9%) had any call. 0.019 calls per clean map.
- Mean localization error: **0.77 bins** for v2 versus 4.34 for legacy.

Recall by event kind (v2; source: `heldout_bench.md`, Recall by kind):

| Kind | Events | 1D recall | 2D recall | CNV recall |
|---|---|---|---|---|
| butterfly | 110 | 0.509 | 0.509 | 0.000 |
| deletion | 109 | 0.624 | 0.486 | 0.174 |
| dup | 110 | 0.127 | 0.027 | 0.064 |
| inversion | 116 | 0.466 | 0.466 | 0.000 |

2D recall by allele fraction: 0.187 (af 0.35), 0.384 (af 0.70), 0.566 (af 1.0). By event size (bins): 0.200 (≤15), 0.274 (15-30), 0.441 (30-60), 0.544 (60-200).

Criteria, as listed in `heldout_bench.md` ("Criteria"):

| Criterion | Measured | Result |
|---|---|---|
| H1 clean maps with any call ≤ 0.092 | 0.019 | pass |
| H2 call FDP ≤ 0.10 | 0.164 | **fail** |
| H3 v2 2D F1 ≥ 0.80 | 0.510 | **fail** |
| H4 v2 1D beats legacy on F1 and AP | F1 0.577 vs 0.068; AP 0.584 vs 0.031 | pass |
| H5 v2 2D AP ≥ 0.90 | 0.506 | **fail** |
| H6 mean localization ≤ 2 bins | 0.77 | pass |
| H7 mixed maps recall ≥ 0.7 | 0.259 (precision 0.880, F1 0.400) | **fail** |

Four of seven criteria failed. On this simulator, v2 output is useful for triage, but its calls are not controlled to the 5% FDR target at the call level, and pair recall is well below the target. Earlier numbers from a different simulator (`bench_detect`) were tuned on that simulator and are not used here.

## Limitations

- **No real-data validation.** Every accuracy number above is from simulation. There is no benchmark against real Hi-C with orthogonal structural-variant truth, and no comparison with hic_breakfinder, EagleC, HiNT or NeoLoopFinder.
- **Duplications are nearly invisible to pair calls.** 2D recall is 0.027 (1D 0.127). Duplication junctions are weak in the obs/exp pair map, so only the coverage track picks up some (CNV recall 0.064).
- **Weak and low-allele-fraction events are missed.** 2D recall is 0.187 at allele fraction 0.35 and 0.200 for events of 15 bins or fewer.
- **Mixed maps.** With 2 to 4 events per map, 2D recall is 0.259.
- **False-discovery control is weak.** Call FDP on the simulated maps is 0.164, above the 0.10 target. q-values are Benjamini-Hochberg within each chromosome and test family. There is no genome-wide FDR. Running several chromosomes does not control the FDR across them.
- **Sparse maps at fine bins.** At fine bin sizes, many pixels hold few contacts, and long-span `deletion/dup` peaks with low scores appear next to real calls. These are not filtered, because a span cap would also remove genuine deletion junctions. Interpret low-score, long-span calls with caution.
- **Resolution limit.** Matrices below about 150 bins give few usable peaks, and events of about 1 Mb span only a few bins. The CLI warns below 150 bins. See [Choosing --screen-bin](#choosing---screen-bin).
- **Scale limit.** v2 pair detection uses dense N x N maps, capped at 3000 bins per chromosome (exit code 2 above that). Legacy has no cap and also builds a dense matrix. In simulation the call FDP varied with N without a clear trend (0.169, 0.071 and 0.253 at N = 150, 300 and 500 bins).
- **CNV calls need coverage.** They rely on coverage shifts, so low-depth data gives few CNV calls. Compartment-scale coverage or density steps can look like CNV, and the scan has no model for slow coverage bias. Bias-corrected coverage helps if you have it.
- **Typing is heuristic.** The inversion, translocation-like/butterfly, and deletion/dup labels were derived from simulation and are not checked on real data. Translocation calls (`translocation-like/butterfly`) are not validated, and the `deletion/dup` label does not separate deletions from duplications.
- **Edges and small chromosomes.** With the default windows, pairs within 10 bins of either chromosome end are not scored. Chromosomes shorter than 33 bins at the chosen screen bin produce no calls.
- **Balancing assumptions.** ICE assumes equal bin visibility. Copy-number changes and dropout bias the balanced map. Extreme low-coverage bins are masked, so calls near masked regions are less reliable. Only ICE is implemented (no KR).
- **Mate MAPQ.** The default pairing checks the mate's MAPQ only through the `MQ` tag. Without it, a pair is accepted if the leftmost mate passes, and the CLI logs how many pairs were affected. Use `samtools fixmate -m` or `--strict-pairs`.
- **Legacy has no MAPQ filter** and produces about 1% flagged bins regardless of the data. It is kept to reproduce earlier results.
- **Cross-reference assumes partial independence.** The default null estimates overdispersion at the scale of the candidate region from sliding windows, which corrects for correlation only up to that scale. The hap1/hap2 table must come from your own phasing pipeline. A chromosome-wide imbalance is absorbed by the automatic center (`--center auto`) and is not reported.
- **No trans calls.** Each chromosome is analyzed on its own. Inter-chromosomal contacts are counted in the matrix build but not called.
- **FASTQ/SAM paths are untested here.** The bwa and samtools steps were not run in the environment that produced this README.
- **Calls are not validated SVs.** Treat them as candidates for inspection.

## Reproducing the benchmarks

Run from the repository root with the dev environment active. Output goes under `benchmarks/results/` (`heldout_*`) or to the file you name.

```bash
python benchmarks/bench_heldout.py benchmarks/results/heldout_bench.md --workers 4
python benchmarks/bench_heldout.py out.md --quick             # small smoke run
python benchmarks/bench_matrix.py out_matrix.md
python benchmarks/bench_validate_cli.py --sizes 50000 200000 600000
python benchmarks/bench_validate_cli.py --workers-test
```

The second-simulator run regenerates `heldout_summary.csv`, `heldout_by_*.csv`, `heldout_pr_curves.json` and `heldout_bench.md`. Reported times depend on the hardware.

## Testing

```bash
pip install -e ".[dev]"
pytest          # unit and integration tests on synthetic BAMs (tests/synth.py, tests/synth_planted.py)
ruff check .    # lint, as in CI
```

CI runs the tests on Python 3.9 to 3.12 and the lint check on 3.12. The tests do not cover real Hi-C data, the bwa/samtools alignment path, or multiprocessing in a restricted sandbox (which falls back to serial).

## Contributing

Issues and pull requests are welcome. Before opening a PR:

1. Run `pytest` and `ruff check .`, and `pre-commit install` if you want the hooks.
2. Keep numbers in the README and docs traceable to a file in `benchmarks/results/` or to a reproducible command.
3. If you change detection, rerun `benchmarks/bench_heldout.py` and update the accuracy table. Do not replace the simulator results with numbers from a simulator the detector was tuned on.
4. Describe any new heuristic in `docs/METHODS.md`, including the assumption it makes.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this software, cite the repository. See [CITATION.cff](CITATION.cff). No DOI is listed in this repository. Please also cite the methods the pipeline builds on:

- Imakaev M, et al. (2012). Iterative correction of Hi-C data reveals hallmarks of chromosome organization. *Nature Methods*.
- Liu FT, Ting KM, Zhou ZH (2008). Isolation forest. *ICDM*.
- Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*.
