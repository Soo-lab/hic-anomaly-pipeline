# CLI needs (ingest)
- `--min-mapq INT` (default 30) -> build_contact_matrix_sparse(min_mapq=)
- `--keep-trans` flag (tallies trans in stats only)
- `--workers INT` (default 4) -> build_matrices_parallel(bam, chroms, bin, workers)
- `--cache-dir PATH` -> save_matrix_npz/load_matrix_npz (path per chrom+bin+mapq)
- `--balance {none,ice}`, `--ignore-diags INT` (default 2) -> ice_balance
- `--oe` -> expected_by_diagonal + obs_over_exp
- Log stats dict from build_contact_matrix_sparse (reads dropped by reason).
