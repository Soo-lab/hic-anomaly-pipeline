"""Backward-compatible entry point for cross-referencing.

The implementation lives in ``hic_anomaly.crossref``. Prefer
``hic-anomaly-crossref`` (console script). This shim keeps the old
``python cross_refrence_hits.py --anomalies ... --imbalance ...`` command working.
"""

from hic_anomaly.crossref import cross_reference, main, normalize_chrom, parse_region

__all__ = ["parse_region", "normalize_chrom", "cross_reference", "main"]

if __name__ == "__main__":
    main()
