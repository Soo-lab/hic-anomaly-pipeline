import numpy as np

from hic_anomaly.crossref import normalize_chrom, parse_region
from hic_anomaly.detect import extract_bin_features, merge_adjacent_bins, robust_z, run_ai
from hic_anomaly.matrix import resolve_name


def test_resolve_name_adds_or_strips_chr_prefix():
    assert resolve_name("chr21", ["21", "chr1"]) == "21"
    assert resolve_name("1", ["chr1"]) == "chr1"
    assert resolve_name("chrX", ["chr1"]) is None


def test_robust_z_zero_mad_returns_zeros():
    assert np.all(robust_z([5, 5, 5, 5]) == 0)


def test_merge_adjacent_bins_merges_gap_of_one():
    regions = merge_adjacent_bins([2, 3, 6], np.linspace(0, 1, 10), 100, "chr1", 1000)
    assert [(r["start_bin"], r["end_bin"]) for r in regions] == [(2, 3), (6, 6)]


def test_extract_bin_features_shape(toy_matrix):
    assert extract_bin_features(toy_matrix).shape == (toy_matrix.shape[0], 8)


def test_run_ai_shapes_and_block_detection(toy_matrix):
    anomalies, scores = run_ai(toy_matrix, contamination=0.05)
    assert anomalies.shape == (toy_matrix.shape[0],)
    assert scores.shape == (toy_matrix.shape[0],)
    assert np.all((scores >= 0) & (scores <= 1))


def test_parse_region_and_normalize_chrom():
    assert parse_region("chr1:45,000,000-47,000,000") == ("chr1", 45000000, 47000000)
    assert normalize_chrom("chr21") == "21"
