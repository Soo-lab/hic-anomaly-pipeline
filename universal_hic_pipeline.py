"""Backward-compatible entry point.

The implementation lives in the ``hic_anomaly`` package. Prefer
``hic-anomaly`` (console script) or ``python -m hic_anomaly``. This shim keeps
``python universal_hic_pipeline.py ...`` and old ``from universal_hic_pipeline
import ...`` imports working.
"""

from hic_anomaly.cli import main
from hic_anomaly.detect import (
    extract_bin_features,
    merge_adjacent_bins,
    robust_z,
    run_ai,
)
from hic_anomaly.io_utils import (
    align_fastq_or_fasta,
    check_tool_exists,
    convert_sam_to_bam,
    run_cmd,
    standardize_input_to_bam,
)
from hic_anomaly.matrix import (
    create_contact_matrix,
    resolve_chrom_and_length,
    resolve_name,
)
from hic_anomaly.viz import plot_matrix, zoom_region_from_bam

__all__ = [
    "run_cmd",
    "check_tool_exists",
    "align_fastq_or_fasta",
    "convert_sam_to_bam",
    "standardize_input_to_bam",
    "resolve_name",
    "resolve_chrom_and_length",
    "create_contact_matrix",
    "robust_z",
    "extract_bin_features",
    "run_ai",
    "merge_adjacent_bins",
    "plot_matrix",
    "zoom_region_from_bam",
    "main",
]

if __name__ == "__main__":
    main()
