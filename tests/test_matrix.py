import numpy as np
import scipy.sparse as sp

from hic_anomaly.matrix import (
    build_contact_matrix_sparse,
    build_matrices_parallel,
    create_contact_matrix,
    expected_by_diagonal,
    ice_balance,
    load_matrix_npz,
    obs_over_exp,
    save_matrix_npz,
)
from tests.synth import make_bam


def baseline(bam_path, chrom, chrom_len, bin_size):
    import pysam

    nb = int(np.ceil(chrom_len / bin_size))
    m = np.zeros((nb, nb), dtype=np.int32)
    pend, tot, used = {}, 0, 0
    for r in pysam.AlignmentFile(bam_path).fetch(chrom):
        tot += 1
        if (
            r.is_unmapped
            or r.mate_is_unmapped
            or r.is_secondary
            or r.is_supplementary
            or r.is_duplicate
        ):
            continue
        if r.next_reference_id != r.reference_id:
            continue
        if r.query_name not in pend:
            pend[r.query_name] = r
            continue
        r1 = pend.pop(r.query_name)
        b1, b2 = r1.reference_start // bin_size, r.reference_start // bin_size
        if b1 == b2:
            m[b1, b2] += 1
        else:
            m[b1, b2] += 1
            m[b2, b1] += 1
        used += 1
    return m, tot, used


def test_golden_dense(tmp_path):
    bam = make_bam(tmp_path / "a.bam", 3000, 200_000, seed=1)
    m, t, u = create_contact_matrix(bam, "chr1", 200_000, 10_000)
    bm, bt, bu = baseline(bam, "chr1", 200_000, 10_000)
    assert (m == bm).all() and (t, u) == (bt, bu)
    assert m.dtype == np.int32


def test_sparse_mapq_stats(tmp_path):
    bam = make_bam(tmp_path / "a.bam", 3000, 200_000, seed=2)
    s, st = build_contact_matrix_sparse(bam, "chr1", 200_000, 10_000, min_mapq=30)
    assert sp.issparse(s) and (s != s.T).nnz == 0
    assert st["low_mapq"] > 0 and st["duplicate"] > 0
    assert s.sum() >= st["used"]
    s0, st0 = build_contact_matrix_sparse(bam, "chr1", 200_000, 10_000, min_mapq=0)
    assert st0["used"] > st["used"]


def test_ice_flattens_coverage():
    rng = np.random.default_rng(0)
    n = 60
    bias = rng.uniform(0.5, 2, n)
    i, j = np.indices((n, n))
    t = 1000.0 / (1 + np.abs(i - j))
    a = rng.poisson(t * np.outer(bias, bias)).astype(float)
    a = np.triu(a) + np.triu(a, 1).T
    bal, w = ice_balance(sp.csr_matrix(a), mask_low_mad=False)
    rs = np.asarray(bal.sum(1)).ravel()
    assert rs.std() / rs.mean() < 0.02
    assert np.isfinite(w).all()


def test_expected_and_oe():
    n = 40
    i, j = np.indices((n, n))
    a = 100.0 / (1 + np.abs(i - j))
    e = expected_by_diagonal(sp.csr_matrix(a), smooth=0)
    assert np.allclose(e, 100.0 / (1 + np.arange(n)), rtol=1e-6)
    oe = obs_over_exp(sp.csr_matrix(a), e)
    assert np.abs(oe.data).max() < 1e-3
    assert np.nanmax(np.abs(obs_over_exp(a, e))) < 1e-3


def test_parallel_and_cache(tmp_path):
    bam = make_bam(tmp_path / "a.bam", 2000, 100_000, seed=3, chroms=("chr1", "chr2"))
    res = build_matrices_parallel(bam, ["chr1", "chr2"], 10_000, workers=2)
    s, st = res["chr1"]
    single, _ = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000)
    assert (s != single).nnz == 0
    p = tmp_path / "c.npz"
    save_matrix_npz(p, s, st, bias=np.ones(s.shape[0]))
    s2, st2, b = load_matrix_npz(p)
    assert (s != s2).nnz == 0 and st2 == st and b.shape[0] == s.shape[0]
    assert load_matrix_npz(tmp_path / "none.npz") is None


# --- integration fixes -------------------------------------------------------


def write_bam(path, pairs, chroms=("chr1", "chr2"), chrom_len=100_000):
    """pairs: dicts with tid1,pos1,tid2,pos2 and optional dup1,dup2,mq1,mq2,mq_tag."""
    import pysam

    h = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": c, "LN": chrom_len} for c in chroms],
        }
    )
    recs = []
    for k, p in enumerate(pairs):
        ends = [
            (p["tid1"], p["pos1"], p.get("mq1", 60), p.get("dup1", False), 64),
            (p["tid2"], p["pos2"], p.get("mq2", 60), p.get("dup2", False), 128),
        ]
        for e, (tid, pos, mq, dup, which) in enumerate(ends):
            mtid, mpos, mmq = ends[1 - e][:3]
            a = pysam.AlignedSegment(h)
            a.query_name, a.query_sequence = f"q{k}", "A" * 20
            a.flag = 1 | which | (1024 if dup else 0)
            a.reference_id, a.reference_start = tid, pos
            a.mapping_quality, a.cigarstring = mq, "20M"
            a.next_reference_id, a.next_reference_start = mtid, mpos
            if p.get("mq_tag", False):
                a.set_tag("MQ", mmq)
            recs.append(a)
    recs.sort(key=lambda a: (a.reference_id, a.reference_start))
    with pysam.AlignmentFile(str(path), "wb", header=h) as f:
        for a in recs:
            f.write(a)
    pysam.index(str(path))
    return str(path)


def baseline_mapq(bam_path, chrom, chrom_len, bin_size, min_mapq):
    """Original name pairing, extended with a per-read MAPQ filter."""
    import pysam

    nb = int(np.ceil(chrom_len / bin_size))
    m = np.zeros((nb, nb), dtype=np.int32)
    pend = {}
    for r in pysam.AlignmentFile(bam_path).fetch(chrom):
        if r.is_unmapped or r.mate_is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if r.is_duplicate or r.mapping_quality < min_mapq:
            continue
        if r.next_reference_id != r.reference_id:
            continue
        if r.query_name not in pend:
            pend[r.query_name] = r
            continue
        b1 = pend.pop(r.query_name).reference_start // bin_size
        b2 = r.reference_start // bin_size
        m[b1, b2] += 1
        if b1 != b2:
            m[b2, b1] += 1
    return m


def mixed_pairs(mq_tag, n=400, seed=5):
    """Cis pairs on chr1 where either mate alone may be duplicate or low MAPQ."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, b = sorted(int(x) for x in rng.integers(0, 99_000, 2))
        kind = int(rng.integers(6))
        out.append(
            {
                "tid1": 0,
                "pos1": a,
                "tid2": 0,
                "pos2": b,
                "dup1": kind == 1,
                "dup2": kind == 2,
                "mq1": 5 if kind == 3 else 60,
                "mq2": 5 if kind == 4 else 60,
                "mq_tag": mq_tag,
            }
        )
    return out


def test_expected_small_n():
    for n in range(0, 8):
        a = sp.csr_matrix(np.ones((n, n)))
        e = expected_by_diagonal(a)  # default smooth=5 > n used to raise IndexError
        assert e.shape == (n,) and np.isfinite(e).all()
        if n:
            assert np.allclose(e, 1.0)
            assert obs_over_exp(a, e, dense_output=True).shape == (n, n)
    e = expected_by_diagonal(sp.csr_matrix(np.ones((3, 3))), mask=np.array([True, True, True]))
    assert (e == 0).all()


def test_strict_pairs_matches_name_pairing(tmp_path):
    for tag in (False, True):
        bam = write_bam(tmp_path / f"m{tag}.bam", mixed_pairs(tag))
        for mq in (0, 30):
            ref = baseline_mapq(bam, "chr1", 100_000, 10_000, mq)
            s, st = build_contact_matrix_sparse(
                bam, "chr1", 100_000, 10_000, min_mapq=mq, strict_pairs=True
            )
            assert (s.toarray() == ref).all()
            assert st["used"] == np.triu(ref).sum() and st["orphan"] > 0
            assert 0 < st["max_pending"] <= 400
    # create_contact_matrix is the exact original: one-mate duplicates are dropped.
    m, _, used = create_contact_matrix(bam, "chr1", 100_000, 10_000)
    assert (m == baseline(bam, "chr1", 100_000, 10_000)[0]).all()
    assert used == baseline(bam, "chr1", 100_000, 10_000)[2]


def test_fast_pairs_documented_divergence(tmp_path):
    pairs = mixed_pairs(True)
    n_dup_right = sum(p["dup2"] for p in pairs)
    n_lowq_right = sum(p["mq2"] < 30 for p in pairs)
    assert n_dup_right and n_lowq_right
    # MQ tag present: the only divergence is "rightmost mate alone is duplicate".
    bam = write_bam(tmp_path / "tag.bam", pairs)
    strict, sst = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000, strict_pairs=True)
    fast, fst = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000)
    assert fst["used"] - sst["used"] == n_dup_right and fst["no_mq_tag"] == 0
    assert (fast - strict).min() >= 0
    # No MQ tag: rightmost-only low MAPQ also slips through, and is reported.
    bam = write_bam(tmp_path / "notag.bam", mixed_pairs(False))
    fast2, fst2 = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000)
    assert fst2["used"] - sst["used"] == n_dup_right + n_lowq_right
    assert fst2["no_mq_tag"] == fst2["used"]
    # No filters in play -> both modes are identical.
    clean = [dict(p, dup1=False, dup2=False, mq1=60, mq2=60) for p in pairs]
    bam = write_bam(tmp_path / "clean.bam", clean)
    a, _ = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000, strict_pairs=True)
    b, _ = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000)
    assert (a != b).nnz == 0 and a.sum() > 0


def test_keep_trans(tmp_path):
    pairs = [{"tid1": 0, "pos1": 100 * k, "tid2": 0, "pos2": 50_000} for k in range(10)]
    pairs += [{"tid1": 0, "pos1": 25_000, "tid2": 1, "pos2": 10 * k} for k in range(7)]
    pairs += [{"tid1": 0, "pos1": 95_000, "tid2": 2, "pos2": 500} for k in range(3)]
    pairs += [{"tid1": 0, "pos1": 95_000, "tid2": 2, "pos2": 500, "dup1": True, "dup2": True}]
    pairs += [{"tid1": 0, "pos1": 5_000, "tid2": 1, "pos2": 9, "mq2": 3, "mq_tag": True}]
    bam = write_bam(tmp_path / "t.bam", pairs, chroms=("chr1", "chr2", "chr3"))
    plain = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000)
    assert len(plain) == 2 and plain[1]["trans"] == 10 and "trans_by_chrom" not in plain[1]
    for strict in (False, True):
        mat, st, trans = build_contact_matrix_sparse(
            bam, "chr1", 100_000, 10_000, keep_trans=True, strict_pairs=strict
        )
        assert (mat != plain[0]).nnz == 0 and mat.sum() == 20  # trans never enters cis
        assert trans["chroms"] == ["chr1", "chr2", "chr3"]
        t = trans["counts"].toarray()
        assert t.shape == (10, 3) and t.sum() == st["trans"] == 10
        assert t[2, 1] == 7 and t[9, 2] == 3 and t[:, 0].sum() == 0
        assert st["trans_by_chrom"] == {"chr2": 7, "chr3": 3}
        assert st["duplicate"] == 1 and st["low_mapq"] == 1
    # the partner chromosome sees the same contacts from its side
    _, st2, tr2 = build_contact_matrix_sparse(bam, "chr2", 100_000, 10_000, keep_trans=True)
    assert tr2["counts"].toarray()[0, 0] == 7 and st2["trans_by_chrom"] == {"chr1": 7}
    # forwarded by the parallel builder, and cached
    res = build_matrices_parallel(bam, ["chr1", "chr2"], 10_000, workers=2, keep_trans=True)
    m1, s1, t1 = res["chr1"]
    assert (t1["counts"] != trans["counts"]).nnz == 0
    assert len(build_matrices_parallel(bam, ["chr1"], 10_000)["chr1"]) == 2
    p = tmp_path / "t.npz"
    save_matrix_npz(p, m1, s1, trans=t1)
    assert len(load_matrix_npz(p)) == 3
    m2, s2, _, t2 = load_matrix_npz(p, with_trans=True)
    assert s2 == s1 and t2["chroms"] == t1["chroms"]
    assert (t2["counts"] != t1["counts"]).nnz == 0


def test_bam_threads_forwarded(tmp_path, monkeypatch):
    import pysam

    from hic_anomaly import matrix as mx

    bam = make_bam(tmp_path / "a.bam", 500, 100_000, seed=4, chroms=("chr1", "chr2"))
    seen = []
    real = pysam.AlignmentFile

    def spy(*a, **k):
        if "threads" in k:
            seen.append(k["threads"])
        return real(*a, **k)

    monkeypatch.setattr(mx.pysam, "AlignmentFile", spy)
    a, _ = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000)
    b, _ = build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000, bam_threads=3)
    assert seen == [1, 3] and (a != b).nnz == 0
    seen.clear()
    build_matrices_parallel(bam, ["chr1", "chr2"], 10_000, workers=1, bam_threads=2)
    assert seen == [2, 2]


def test_obs_over_exp_single_definition():
    rng = np.random.default_rng(3)
    n = 50
    i, j = np.indices((n, n))
    d = np.abs(i - j)
    a = rng.poisson(40.0 / (1 + d) ** 3).astype(float)
    a = np.triu(a) + np.triu(a, 1).T
    assert (a == 0).any()
    s = sp.csr_matrix(a)
    e = expected_by_diagonal(s)
    ref = np.log2((a + 1) / (e[d] + 1))
    dense = obs_over_exp(a, e)
    assert np.allclose(dense, ref)
    # sparse -> exact dense is the same map
    assert np.allclose(obs_over_exp(s, e, dense_output=True), ref)
    # sparse -> sparse: formula on the stored pattern, unstored pixels read as 0
    oe = obs_over_exp(s, e)
    assert sp.issparse(oe) and oe.nnz == s.nnz
    got = oe.toarray()
    assert np.allclose(got[a > 0], ref[a > 0])
    assert (got[a == 0] == 0).all()
    # ... which matches the formula where expected << 1 and nowhere else
    gap = np.abs(got - ref)[a == 0]
    assert np.allclose(gap, np.log2(1 + e[d][a == 0]))
    far = (a == 0) & (e[d] < 0.01)
    assert far.any() and np.abs(got - ref)[far].max() < 0.015
    # pseudo is the one knob, shared by both paths
    assert np.allclose(obs_over_exp(a, e, pseudo=0.5), np.log2((a + 0.5) / (e[d] + 0.5)))


def test_ice_masking_adaptive_on_synth(tmp_path):
    # Uniform synthetic data: nothing is unmappable, so (almost) nothing is masked.
    for n_pairs, length, ignore in [
        (2000, 1_000_000, 2),
        (20_000, 1_000_000, 2),
        (3000, 200_000, 0),
    ]:
        bam = make_bam(tmp_path / f"u{n_pairs}_{length}.bam", n_pairs, length, seed=1)
        m, _ = build_contact_matrix_sparse(bam, "chr1", length, 10_000, min_mapq=30)
        bal, bias = ice_balance(m, ignore_diags=ignore)
        assert np.isnan(bias).sum() <= 1, (n_pairs, length, np.isnan(bias).sum())
        ok = np.isfinite(bias)
        rs = np.asarray(bal.sum(1)).ravel()[ok]
        assert rs.std() / rs.mean() < 0.01
    # tiny matrix: the fixed min_nnz=10 would mask every bin
    a = np.full((8, 8), 30.0)
    _, bias = ice_balance(sp.csr_matrix(a))
    assert np.isfinite(bias).all()
    _, bias = ice_balance(sp.csr_matrix(a), min_nnz_frac=None)
    assert np.isnan(bias).all()


def test_ice_still_masks_bad_bins_and_keeps_count_scale():
    rng = np.random.default_rng(1)
    n = 120
    i, j = np.indices((n, n))
    a = rng.poisson(300.0 / (1 + np.abs(i - j))).astype(float)
    a = np.triu(a) + np.triu(a, 1).T
    a[10, :] = a[:, 10] = 0  # no coverage
    a[40, :] = a[:, 40] = rng.poisson(0.02 * a[40, :])  # ~50-fold depleted
    a[40, :] = a[:, 40]
    a[70, :] = a[:, 70] = np.round(0.6 * a[70, :])  # mild dip: ICE corrects it
    bal, bias = ice_balance(sp.csr_matrix(a))
    bad = np.flatnonzero(np.isnan(bias))
    assert set(bad) == {10, 40}
    kept = np.abs(i - j) >= 2
    kept &= ~np.isin(i, bad) & ~np.isin(j, bad)
    assert np.isclose(bal.sum(), a[kept].sum())  # read-count scale
    unit, _ = ice_balance(sp.csr_matrix(a), rescale="unit")
    rs = np.asarray(unit.sum(1)).ravel()
    assert np.isclose(rs[np.isfinite(bias)].mean(), 1.0)


def test_summary_log_levels(tmp_path, caplog):
    import logging

    bam = make_bam(tmp_path / "a.bam", 300, 100_000, seed=6)
    with caplog.at_level(logging.DEBUG, logger="hic_anomaly.matrix"):
        create_contact_matrix(bam, "chr1", 100_000, 10_000)
    assert caplog.records and all(r.levelno == logging.DEBUG for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hic_anomaly.matrix"):
        build_contact_matrix_sparse(bam, "chr1", 100_000, 10_000, min_mapq=0)
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "pairs used; dropped" in caplog.records[0].getMessage()


def test_ice_small_uniform_not_overmasked():
    rng = np.random.default_rng(0)
    n = 100
    a = rng.poisson(3, (n, n))
    a = np.triu(a) + np.triu(a, 1).T
    _, w = ice_balance(sp.csr_matrix(a))
    assert np.isfinite(w).sum() >= 0.95 * n
