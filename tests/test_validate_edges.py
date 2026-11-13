"""Hand-computed edge cases for the contact-matrix builder (independent validation)."""

import numpy as np
import pysam
import pytest

from hic_anomaly.matrix import (
    build_contact_matrix_sparse,
    create_contact_matrix,
    resolve_chrom_and_length,
)

L = 50
BIN = 10_000
CLEN = 100_000


def _rec(h, name, tid, pos, ntid, npos, flag, mapq=60, mq=None):
    a = pysam.AlignedSegment(h)
    a.query_name, a.query_sequence = name, "A" * L
    a.flag, a.reference_id, a.reference_start = flag, tid, pos
    a.next_reference_id, a.next_reference_start = ntid, npos
    a.mapping_quality, a.cigarstring = mapq, f"{L}M"
    a.query_qualities = pysam.qualitystring_to_array("I" * L)
    if mq is not None:
        a.set_tag("MQ", mq)
    return a


def _write(path, recs_fn, sq=(("chr1", CLEN), ("chr2", CLEN))):
    h = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": n, "LN": ln} for n, ln in sq]}
    )
    recs = recs_fn(h)
    recs.sort(key=lambda a: (a.reference_id, a.reference_start))
    with pysam.AlignmentFile(str(path), "wb", header=h) as f:
        for a in recs:
            f.write(a)
    pysam.index(str(path))
    return str(path)


def _pair(h, name, p1, p2, f1=0, f2=0, mq1=60, mq2=60, t1=0, t2=0, tag=True):
    r1 = _rec(h, name, t1, p1, t2, p2, 1 | 64 | 32 | f1, mq1, mq2 if tag else None)
    r2 = _rec(h, name, t2, p2, t1, p1, 1 | 128 | 16 | f2, mq2, mq1 if tag else None)
    return [r1, r2]


def _edge_bam(path):
    def f(h):
        r = []
        r += _pair(h, "ok_a", 100, 55_000)  # bins 0,5
        r += _pair(h, "ok_same", 100, 5_000)  # bin 0,0 diagonal
        r += _pair(h, "dup_both", 200, 55_000, 1024, 1024)  # dropped both modes
        r += _pair(h, "dup_left", 300, 55_000, 1024, 0)  # dropped both modes
        r += _pair(h, "mq29", 400, 65_000, mq1=29)  # dropped (min_mapq=30)
        r += _pair(h, "mq30", 500, 75_000, mq1=30, mq2=30)  # kept (boundary)
        r += _pair(h, "mq_mate29", 600, 85_000, mq2=29)  # dropped via MQ tag / name
        r += _pair(h, "supp", 700, 55_000, 2048, 0)  # dropped
        r += _pair(h, "sec", 800, 55_000, 256, 0)  # dropped
        r += _pair(h, "trans", 900, 55_000, t2=1)  # trans: never in cis matrix
        # unpaired single read
        r.append(_rec(h, "single", 0, 1000, -1, -1, 0, 60))
        # edge bin: last base of the contig
        r += _pair(h, "edge", 99_000, CLEN - L)  # bins 9,9
        return r

    return _write(path, f)


def test_hand_computed_matrix_both_modes(tmp_path):
    bam = _edge_bam(tmp_path / "e.bam")
    for strict in (True, False):
        m, st = build_contact_matrix_sparse(
            bam, "chr1", CLEN, BIN, min_mapq=30, strict_pairs=strict
        )
        d = m.toarray()
        exp = np.zeros((10, 10), dtype=np.int32)
        exp[0, 5] = exp[5, 0] = 1  # ok_a
        exp[0, 0] = 1  # ok_same (diagonal counted once)
        exp[0, 7] = exp[7, 0] = 1  # mq30
        exp[9, 9] = 1  # edge bin
        assert (d == exp).all(), strict
        assert (d == d.T).all()
        assert st["used"] == 4
        assert st["trans"] >= 1 and st["unpaired"] >= 1


def test_mapq_zero_keeps_low_mapq_pairs(tmp_path):
    bam = _edge_bam(tmp_path / "e.bam")
    m, st = build_contact_matrix_sparse(bam, "chr1", CLEN, BIN, min_mapq=0, strict_pairs=True)
    # ok_a, ok_same, mq30, edge + mq29 + mq_mate29 = 6; dup/supp/sec/trans/unpaired still dropped
    assert st["used"] == 6 and st["low_mapq"] == 0
    assert m.sum() == 2 * 6 - m.diagonal().sum()


def test_fast_equals_strict_on_clean_tagged_bam(tmp_path):
    bam = _edge_bam(tmp_path / "e.bam")
    a, _ = build_contact_matrix_sparse(bam, "chr1", CLEN, BIN, strict_pairs=True)
    b, _ = build_contact_matrix_sparse(bam, "chr1", CLEN, BIN, strict_pairs=False)
    assert (a != b).nnz == 0


def test_one_mate_duplicate_divergence_is_documented(tmp_path):
    def f(h):
        return _pair(h, "dup_right", 100, 55_000, 0, 1024)

    bam = _write(tmp_path / "d.bam", f)
    s, _ = build_contact_matrix_sparse(bam, "chr1", CLEN, BIN, strict_pairs=True)
    q, _ = build_contact_matrix_sparse(bam, "chr1", CLEN, BIN, strict_pairs=False)
    assert s.sum() == 0 and q.sum() == 2  # documented: fast mode counts it


def test_create_contact_matrix_is_symmetric_and_counts(tmp_path):
    bam = _edge_bam(tmp_path / "e.bam")
    m, total, used = create_contact_matrix(bam, "chr1", CLEN, BIN)
    assert (m == m.T).all() and m.dtype == np.int32
    assert total > used
    # original semantics: no MAPQ filter -> mq29/mq_mate29 pairs count (6 pairs)
    assert used == 6
    assert m.sum() == 2 * 6 - np.trace(m)


@pytest.mark.parametrize("name", ["chr1", "1"])
def test_chrom_naming_chr_prefix(tmp_path, name):
    bam = _edge_bam(tmp_path / "e.bam")
    assert resolve_chrom_and_length(bam, name) == ("chr1", CLEN)
    bam2 = _write(
        tmp_path / "n.bam", lambda h: _pair(h, "x", 100, 55_000), sq=(("1", CLEN), ("2", CLEN))
    )
    assert resolve_chrom_and_length(bam2, "chr1") == ("1", CLEN)
    assert resolve_chrom_and_length(bam2, "1") == ("1", CLEN)


def test_unknown_chrom_raises_clear_error(tmp_path):
    bam = _edge_bam(tmp_path / "e.bam")
    with pytest.raises(ValueError, match="not found in BAM header"):
        resolve_chrom_and_length(bam, "chrZ")


def test_empty_bam_gives_zero_matrix(tmp_path):
    bam = _write(tmp_path / "empty.bam", lambda h: [])
    for strict in (True, False):
        m, st = build_contact_matrix_sparse(bam, "chr1", CLEN, BIN, strict_pairs=strict)
        assert m.shape == (10, 10) and m.sum() == 0 and st["used"] == 0 and st["total"] == 0


def test_tiny_contig_single_bin(tmp_path):
    bam = _write(
        tmp_path / "tiny.bam",
        lambda h: _pair(h, "t", 0, 60),
        sq=(("chr1", 200),),
    )
    for strict in (True, False):
        m, st = build_contact_matrix_sparse(bam, "chr1", 200, BIN, strict_pairs=strict)
        assert m.shape == (1, 1) and m.toarray()[0, 0] == 1 and st["used"] == 1
