# hic-anomaly-pipeline overhaul: handoff summary (final, 2026-11-13)

Repo: github.com/Soo-lab/hic-anomaly-pipeline. Branch `upgrade/full-overhaul`. State at handoff: 161 pytest passing, `ruff check .` clean. All benchmark data is SIMULATED; no real Hi-C validation yet.

## Original state
Two scripts, 18-line README, misspelled requirments.txt, crucial.gitignore, no tests/CI. Dense int32 NxN matrix, unbounded mate dict, no MAPQ filter, no normalization, Isolation Forest with fixed contamination=0.01 (forces ~1% hits), no p-values.

## What was built
- Package `hic_anomaly/` (io_utils, matrix, detect, stats_utils, crossref, viz, report, cli) + root shims; pyproject (console scripts `hic-anomaly`, `hic-anomaly-crossref`), requirements.txt, real .gitignore, environment.yml, Dockerfile, pre-commit, MIT LICENSE, GitHub Actions CI (3.9-3.12), CITATION.cff, docs/METHODS.md, rewritten README.
- matrix.py: sparse dict-free leftmost-mate counting (`--strict-pairs` for exact baseline pairing), MAPQ filter (default 30), ICE balancing (adaptive masking), per-diagonal expected, single obs/exp definition log2((obs+1)/(exp+1)), per-chrom parallel builder, npz cache, trans-contact option.
- detect.py v2: pair-level calls (deletion/dup, inversion, translocation-like/butterfly, cnv, homozygous-deletion-candidate) via summed-area-table features, two-way median polish, Isolation Forest + robust-z score, data-adaptive peak null (Gumbel + censored exponential tails), BH q-values per chromosome, CNV scan with Bonferroni floor. `return_scores=True` gives per-bin scores. Legacy kept as `--method legacy`.
- crossref.py: beta-binomial allelic imbalance, region-level overdispersion, BH FDR, log2 effect + CI; warns on imbalance-only input.
- viz/report: heatmaps with call markers, triangular view, score track, comparison, PR curves, self-contained HTML report; `style="legacy"` reproduces the original figure byte-for-byte.
- cli.py: v2/legacy flows, multi-chrom/all, cache, run_info.json, exit code 2 for usage errors, small-matrix warning, 3000-bin dense limit.

## Measured results (synthetic; sources: .agent-reports/benchmarks.md, benchmarks/results/heldout_bench.md)
- Matrix build: tracemalloc peak 1605 MB -> 29.3 MB (55x) at the largest case; time 4.80 s -> 2.63 s.
- End to end CLI: peak RSS ~46-51% lower than legacy; wall time only 8-12% lower. `--workers 4` 1.9x faster on matrix build alone, no end-to-end gain at these sizes.
- Detection, second simulator (NB noise, compartments, bias, 504 maps, default seeds): v2 2D F1 0.510, pooled 1D F1 0.577 vs legacy 0.068, call-level FDP 0.164, clean maps with a call 1.9% (legacy 3.26 forced regions/map), localization 0.77 bins (legacy 4.34). Four of seven pre-set criteria FAIL: FDP (target 0.10), 2D F1, 2D PR-AUC, mixed-map recall (0.26 vs 0.7).
- NOT a blind test: detector revised in two rounds after failure analyses; an earlier quick benchmark (F1 0.909) was tuned on its own simulator and must not be quoted. The old held-out section in benchmarks.md is marked SUPERSEDED.

## Known limitations / open issues
1. Dup recall ~0.03; weak/low-allele-fraction events missed (af 0.35 recall 0.19); events <=15 bins recall 0.20.
2. Sparse maps at fine bins produce long-span low-score deletion/dup noise calls (demo at 10 kb: 8 of 25 calls). Needs a minimum-contact-density check (requires raw counts in detect).
3. FDR is per chromosome, not genome-wide. Permutation-null residual-bias sensitivity on real ICE data unverified. Typing is heuristic, validated on simulation only (translocation barely tested).
4. CNV track needs a coverage vector; compartment-scale coverage steps look like CNV.
5. Dense N^2 detection: practical limit 3000 bins per chromosome; v2 detection is ~2x slower than legacy per map (0.86 s at N=500).
6. Crossref assumes partial bin independence; slightly conservative; rho estimate noisy with <60 windows.
7. Fast pairing mode counts a pair where only the rightmost mate is a duplicate (strict mode matches baseline).
8. Not tested: FASTQ->BAM alignment (bwa absent), Docker build, conda solve, Python 3.9-3.11, real-data benchmarks vs hic_breakfinder/EagleC/HiNT/NeoLoopFinder. Literature URLs/accessions in methods.md and METHODS references are from memory, unchecked.
9. Cosmetic: detect.py comments/docstring still mention a within-diagonal permutation null (function defined, unused) and a "3 bins" breakpoint window vs code 2*w_max; `n_null` flag unused; viz cmap not exposed on CLI; hic_fall colormap not perceptually uniform (cividis/viridis available via API).

## Team (Claude Code agents in ~/.claude/agents)
insight-lead (Tier 1, sonnet); delve-research, -viz, -pkg, -report (haiku), delve-ingest, delve-stats (sonnet) = builders; verity-data, -stats (sonnet), verity-viz (haiku) = validators. Reports in .agent-reports/.

## Suggested next steps
Real-data benchmark (e.g. K562 BCR-ABL1 translocation, 4DN/GSE63525 data) against competitor tools; minimum contact-density filter; improve dup/weak-event recall; genome-wide FDR; sparse/banded detection for fine bins; verify references; fix the cosmetic comment/docstring items.
