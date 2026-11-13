import numpy as np
import pytest

from hic_anomaly.detect import bin_features, detect_anomalies, extract_bin_features, run_ai
from hic_anomaly.stats_utils import bh_fdr, empirical_pvalue

N = 200


def expected(n=N, scale=60.0):
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return scale / (1.0 + d) ** 1.0 + 0.3


def simulate(seed, planted=None, n=N):
    """Poisson map with distance decay; log2 obs/exp using the *estimated* diagonal mean."""
    rng = np.random.default_rng(seed)
    E = expected(n)
    mu = E.copy()
    if planted == "deletion":  # new contacts between flanks (a, b), bright block
        mu[60:66, 140:146] *= 6
        mu[140:146, 60:66] *= 6
    if planted == "inversion":  # reverse segment [70, 130): remap bins then re-draw
        perm = np.arange(n)
        perm[70:130] = perm[70:130][::-1]
        mu = E[np.ix_(perm, perm)]
    obs = rng.poisson(mu).astype(float)
    obs = np.triu(obs) + np.triu(obs, 1).T
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    sums = np.bincount(d.ravel(), weights=obs.ravel())
    cnt = np.bincount(d.ravel())
    exp_hat = (sums / cnt)[d]
    return np.log2((obs + 1) / (exp_hat + 1))


def near(df, i, j, tol=2):
    if df.empty:
        return False
    return bool(((abs(df.bin_i - i) <= tol) & (abs(df.bin_j - j) <= tol)).any())


def test_deletion_block_recovered():
    df = detect_anomalies(simulate(1, "deletion"), 100_000, seed=1)
    assert near(df, 64, 141, 6)  # compact block: breakpoint refined to a bow-tie corner
    assert (df.q_value <= 0.05).all()
    assert df.iloc[0].type == "deletion/dup"


def test_inversion_recovered_near_breakpoints():
    df = detect_anomalies(simulate(2, "inversion"), 100_000, seed=2)
    assert near(df, 70, 130, 3)
    assert df.iloc[0].type == "inversion"


@pytest.mark.parametrize("seed", range(6))
def test_clean_maps_few_calls(seed):
    df = detect_anomalies(simulate(100 + seed), 100_000, seed=seed)
    assert len(df) == 0


def test_seed_reproducible():
    L = simulate(5, "deletion")
    a = detect_anomalies(L, 1000, seed=3)
    b = detect_anomalies(L, 1000, seed=3)
    assert a.equals(b)


def test_null_pvalues_roughly_uniform():
    """Clean maps: p-values of tested pairs should be ~uniform."""
    from hic_anomaly import detect as D

    L = simulate(7)
    rng = np.random.default_rng(0)
    nulls = [D._permute_within_diagonals(L, rng) for _ in range(2)]
    stats_ = []
    for M in [L] + nulls:
        ws = (5, 10)
        edges = D._band_edges(N, 11)
        iu, ju, names, Z, _ = D._pair_stat_table(M, ws, 11, np.zeros((N, 4)), edges)
        stats_.append(D._zmax(Z, names))
    pool = np.concatenate(stats_[1:])
    p = empirical_pvalue(stats_[0], pool)
    p = p[np.isfinite(p)]
    assert 0.4 < p.mean() < 0.6
    assert 0.02 < (p < 0.05).mean() < 0.09


def test_cnv_called_with_coverage():
    L = simulate(9)
    cov = np.full(N, 100.0)
    cov[30:50] *= 2.0
    cov = cov * np.random.default_rng(0).normal(1, 0.03, N)
    df = detect_anomalies(L, 1000, coverage=cov, seed=0)
    c = df[df.type == "cnv"]
    assert len(c) == 1 and abs(c.bin_i.iloc[0] - 30) <= 2 and abs(c.bin_j.iloc[0] - 49) <= 2


def test_accepts_sparse_and_nan_mask():
    import scipy.sparse as sp

    L = simulate(1, "deletion")
    L[20:23, :] = np.nan
    L[:, 20:23] = np.nan
    df = detect_anomalies(L, 1000, seed=1)
    assert not df.empty
    detect_anomalies(sp.csr_matrix(np.nan_to_num(L)), 1000, seed=1)


def test_legacy_method_selectable(toy_matrix):
    df = detect_anomalies(toy_matrix, 1000, method="legacy")
    assert (df.type == "legacy").all() and len(df) >= 1
    assert extract_bin_features(toy_matrix).shape[1] == 8
    assert run_ai(toy_matrix)[0].shape == (40,)


def test_bin_features_columns():
    f = bin_features(simulate(1))
    assert {"insulation_w5", "insulation_delta_w10", "asymmetry", "coverage_z"} <= set(f.columns)


def test_bh_matches_manual():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    q = bh_fdr(p)
    assert np.isnan(q[3]) and np.allclose(q[[0, 2, 1, 4]], [0.04, 0.05333333, 0.05333333, 0.5])


def test_return_scores_backward_compatible():
    L = simulate(1, "deletion")
    calls = detect_anomalies(L, 1000, seed=1)
    calls2, scores = detect_anomalies(L, 1000, seed=1, return_scores=True)
    assert calls.equals(calls2)
    assert list(scores.columns) == ["bin", "score", "threshold", "called"]
    assert len(scores) == N and np.isfinite(scores.threshold.iloc[0])
    assert scores.called.sum() >= 1
    # legacy and tiny inputs also honour return_scores
    c, s = detect_anomalies(np.ones((5, 5)), 1000, return_scores=True)
    assert c.empty and len(s) == 5


def test_one_call_per_event_not_per_pixel():
    df = detect_anomalies(simulate(1, "deletion"), 100_000, seed=1)
    assert len(df[df.type != "cnv"]) <= 2


def test_residual_row_col_bias_is_polished_out():
    """A smooth multiplicative per-bin bias left in log2 obs/exp must not create calls."""
    rng = np.random.default_rng(0)
    bad = 0
    for s in range(8):
        L = simulate(300 + s)
        b = np.sin(np.linspace(0, 6, N) + s) * 0.4 + rng.normal(0, 0.1, N)
        bad += len(detect_anomalies(L + b[:, None] + b[None, :], 1000, seed=s)) > 0
    assert bad <= 1


def test_cnv_is_separate_track_and_large_segments_found():
    L = simulate(9)
    cov = np.full(N, 100.0) * np.random.default_rng(1).normal(1, 0.03, N)
    cov[60:120] *= 1.6  # 30% of the map: beyond a 51-bin rolling median
    df = detect_anomalies(L, 1000, coverage=cov, seed=0)
    c = df[df.type == "cnv"]
    assert len(c) == 1 and abs(c.bin_i.iloc[0] - 60) <= 2 and abs(c.bin_j.iloc[0] - 119) <= 2
    assert (df[df.type != "cnv"].shape[0]) == 0


def test_types_cover_inversion_and_single_corner():
    assert detect_anomalies(simulate(2, "inversion"), 100_000, seed=2).iloc[0].type == "inversion"
    assert detect_anomalies(simulate(1, "deletion"), 100_000, seed=1).iloc[0].type == "deletion/dup"


# ---- CNV / masked-block track and data-driven peak null ----
from hic_anomaly.detect import _cnv_calls, _peak_null_pvalues  # noqa: E402


def _cov(seed, n=300, sigma=0.5):
    rng = np.random.default_rng(seed)
    return 1000.0 * 2 ** rng.normal(0, sigma, n)


def _flat_L(n):
    return np.zeros((n, n))


def test_cnv_isolated_low_bins_are_not_calls():
    cov = _cov(0)
    cov[[60, 61, 140, 200, 201, 202]] *= 0.1  # dropout-like bins, runs shorter than min_len
    rows = _cnv_calls(None, _flat_L(300), cov, 0.05)
    assert [r for r in rows if r[2] == "cnv"] == []


def test_cnv_step_found_and_localised():
    cov = _cov(1)
    cov[100:140] *= 2.0
    rows = [r for r in _cnv_calls(None, _flat_L(300), cov, 0.05) if r[2] == "cnv"]
    assert any(abs(r[0] - 100) <= 2 and abs(r[1] - 139) <= 2 for r in rows)


def test_cnv_null_calls_rare():
    hits = sum(len(_cnv_calls(None, _flat_L(300), _cov(s), 0.05)) > 0 for s in range(20))
    assert hits <= 3


def test_masked_block_separate_type_and_isolated_masks_ignored():
    n = 300
    cov = _cov(2, n)
    cov[[30, 90, 200]] = 0  # isolated masked bins: not a block
    cov[120:150] = 0  # bounded run of 30 masked bins
    L = _flat_L(n)
    L[120:150, :] = np.nan
    L[:, 120:150] = np.nan
    rows = _cnv_calls(None, L, cov, 0.05)
    blocks = [r for r in rows if r[2] == "homozygous-deletion-candidate"]
    assert len(blocks) == 1 and (blocks[0][0], blocks[0][1]) == (120, 149)
    assert all(r[2] in ("cnv", "homozygous-deletion-candidate") for r in rows)


def test_masked_block_not_called_at_edge():
    n = 300
    cov = _cov(3, n)
    cov[:20] = 0
    assert not [r for r in _cnv_calls(None, _flat_L(n), cov, 0.05) if r[2] == "homozygous-deletion-candidate"]


def test_peak_null_scale_free_and_calibrated():
    rng = np.random.default_rng(5)
    t = rng.gumbel(0, 1, 400)  # null-like maxima
    p1, _, _ = _peak_null_pvalues(t)
    p2, _, _ = _peak_null_pvalues(10 * t + 3)  # overdispersion / shift must not matter
    assert np.allclose(p1, p2, rtol=1e-6, atol=1e-12)
    assert (p1 < 0.05).mean() < 0.09  # not anti-conservative on a null sample
    t[:3] += 12  # planted events do not inflate the fitted scale
    pe, _, _ = _peak_null_pvalues(t)
    assert (pe[:3] < 1e-3).all()


def test_default_windows_include_small_scale():
    import inspect

    assert inspect.signature(detect_anomalies).parameters["ws"].default[0] <= 3


def test_offdiag_blob_is_pair_call_not_cnv():
    n = 200
    rng = np.random.default_rng(3)
    L = rng.normal(0, 0.3, (n, n))
    L = (L + L.T) / 2
    L[60:66, 140:146] += 2.0
    L[140:146, 60:66] += 2.0
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    L[d < 2] = np.nan
    cov = np.nansum(2.0 ** L, axis=1) * 10
    df = detect_anomalies(L, 1e5, coverage=cov)
    assert len(df) >= 1
    assert not (df["type"] == "cnv").any()
    top = df.iloc[0]
    assert abs(top.bin_i - 62) <= 6 and abs(top.bin_j - 142) <= 6


def test_small_matrix_warns():
    import pytest

    with pytest.warns(UserWarning, match="too small/coarse"):
        detect_anomalies(np.zeros((80, 80)), 250_000)
