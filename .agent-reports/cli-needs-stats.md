# CLI needs (delve-stats)

Add to `hic_anomaly/cli.py`:
- `--method {v2,legacy}` (default v2). v2: `detect_anomalies(logoe, bin_size, method, seed, fdr, ws, n_null, coverage)`
  on a dense/sparse log2 obs/exp map (NaN = masked). legacy: raw-count matrix, old IF (contamination 0.01).
- `--seed INT` (default 42) -> `seed=`; `--fdr FLOAT` (default 0.05); `--n-null INT` (default 5);
  `--windows 5,10` -> `ws=`.
- Pass per-bin coverage (raw row sums or ICE bias) as `coverage=` to enable CNV calls.
- Output DataFrame columns: bin_i, bin_j, start_bp_i, start_bp_j, type, score, p_value, q_value
  (cnv rows: bin_i..bin_j is the run). Zero rows is valid. `merge_adjacent_bins` is not needed for v2.
- Crossref (already in `hic_anomaly/crossref.py` main): `--p0`, `--fdr`, `--rho`. Its anomalies input
  accepts start/end or start_bp/end_bp, so v2 output (start_bp_i etc.) needs renaming to
  chrom,start,end (region = start_bp_i..start_bp_j+bin_size) before calling cross_reference.
