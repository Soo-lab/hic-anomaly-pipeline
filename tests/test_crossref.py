import numpy as np
import pandas as pd

from hic_anomaly.crossref import cross_reference
from hic_anomaly.stats_utils import (
    betabinom_two_sided,
    estimate_rho_mom,
    log2_ratio_ci,
)


def make_imbalance(seed, nbins=600, rho=0.02, planted=(), depth=200):
    rng = np.random.default_rng(seed)
    n = rng.poisson(depth, nbins) + 20
    mu = np.full(nbins, 0.5)
    for a, b, m in planted:
        mu[a:b] = m
    s = 1 / rho - 1
    pr = rng.beta(mu * s, (1 - mu) * s)
    k = rng.binomial(n, pr)
    return pd.DataFrame(
        {
            "chrom": "chr1",
            "start": np.arange(nbins) * 100_000,
            "end": (np.arange(nbins) + 1) * 100_000,
            "hap1_count": k,
            "hap2_count": n - k,
        }
    )


def run(tmp_path, imb, regions, **kw):
    ip, ap, op = tmp_path / "i.tsv", tmp_path / "a.tsv", tmp_path / "o.tsv"
    imb.to_csv(ip, sep="\t", index=False)
    pd.DataFrame(regions, columns=["chrom", "start", "end"]).to_csv(ap, sep="\t", index=False)
    return cross_reference(str(ap), str(ip), str(op), buffer_bp=0, **kw)


def test_rho_recovered_and_robust_to_signal():
    imb = make_imbalance(0, nbins=3000, rho=0.02, planted=[(0, 150, 0.7)])
    k = imb.hap1_count.to_numpy()
    n = k + imb.hap2_count.to_numpy()
    assert abs(estimate_rho_mom(k, n) - 0.02) < 0.006


def test_betabinom_p_uniform_under_null():
    rng = np.random.default_rng(1)
    rho, n = 0.03, np.full(2000, 150)
    pr = rng.beta(0.5 * (1 / rho - 1), 0.5 * (1 / rho - 1), 2000)
    k = rng.binomial(n, pr)
    p = betabinom_two_sided(k, n, 0.5, rho)
    assert 0.45 < p.mean() < 0.55 and 0.03 < (p < 0.05).mean() < 0.07
    # binomial would be badly anti-conservative on the same data
    assert (betabinom_two_sided(k, n, 0.5, 0.0) < 0.05).mean() > 0.2


def test_planted_region_significant_nulls_not(tmp_path):
    imb = make_imbalance(3, planted=[(100, 110, 0.7)])
    regions = [("chr1", 100 * 100_000, 110 * 100_000)] + [
        ("chr1", s * 100_000, (s + 10) * 100_000) for s in range(200, 560, 30)
    ]
    out = run(tmp_path, imb, regions)
    hit = out[out.start == 100 * 100_000].iloc[0]
    assert hit.q_value < 0.01 and hit.cross_validated and hit.log2_effect > 0.5
    assert hit.ci_low < hit.log2_effect < hit.ci_high
    assert (out[out.start != 100 * 100_000].q_value > 0.05).all()
    assert {"p_value", "q_value", "z_score", "mean_abs_imbalance", "cross_validated"} <= set(
        out.columns
    )


def test_missing_overlap_and_alias_columns(tmp_path):
    imb = make_imbalance(4, nbins=50)
    ip, ap, op = tmp_path / "i.tsv", tmp_path / "a.tsv", tmp_path / "o.tsv"
    imb.to_csv(ip, sep="\t", index=False)
    pd.DataFrame({"chrom": ["1", "chr9"], "start_bp": [0, 0], "end_bp": [1_000_000, 10]}).to_csv(
        ap, sep="\t", index=False
    )
    out = cross_reference(str(ap), str(ip), str(op), buffer_bp=0)
    assert len(out) == 2 and out.p_value.isna().sum() == 1


def test_log2_ci_contains_estimate():
    eff, lo, hi = log2_ratio_ci(np.array([70.0]), np.array([100.0]))
    assert lo[0] < eff[0] < hi[0] and eff[0] > 0


def test_estimate_rho_median_robust_to_many_imbalanced_windows():
    from hic_anomaly.stats_utils import estimate_rho_median

    imb = make_imbalance(0, nbins=3000, rho=0.02, planted=[(0, 240, 0.7)])  # 8% imbalanced
    k = imb.hap1_count.to_numpy()
    n = k + imb.hap2_count.to_numpy()
    assert abs(estimate_rho_median(k, n) - 0.02) < 0.01
    assert estimate_rho_median(np.full(100, 50), np.full(100, 100)) == 0.0


def block_correlated(seed, nbins=600, block=5, rho=0.02, depth=200):
    rng = np.random.default_rng(seed)
    n = rng.poisson(depth, nbins) + 5
    s = 1 / rho - 1
    pr = np.repeat(rng.beta(0.5 * s, 0.5 * s, nbins // block + 1), block)[:nbins]
    k = rng.binomial(n, pr)
    return pd.DataFrame(
        {
            "chrom": "chr1",
            "start": np.arange(nbins) * 100_000,
            "end": (np.arange(nbins) + 1) * 100_000,
            "hap1_count": k,
            "hap2_count": n - k,
        }
    )


def test_robust_null_controls_type1_under_block_correlation(tmp_path):
    regions = [("chr1", s * 100_000, (s + 10) * 100_000) for s in range(0, 600, 10)]
    rej = {True: 0, False: 0}
    tot = 0
    for seed in range(6):
        imb = block_correlated(seed)
        for robust in (True, False):
            out = run(tmp_path, imb, regions, robust=robust)
            rej[robust] += int((out.p_value < 0.05).sum())
        tot += len(regions)
    assert rej[True] / tot < 0.08
    assert rej[False] / tot > 0.15  # independent-bin model is anti-conservative here


def test_center_auto_absorbs_reference_bias(tmp_path):
    imb = make_imbalance(5, nbins=600, rho=1e-4)
    tot = imb.hap1_count + imb.hap2_count
    k = np.random.default_rng(5).binomial(tot, 0.53)
    imb["hap2_count"] = tot - k
    imb["hap1_count"] = k
    regions = [("chr1", s * 100_000, (s + 10) * 100_000) for s in range(0, 600, 10)]
    auto = run(tmp_path, imb, regions)
    fixed = run(tmp_path, imb, regions, robust=False)
    assert (auto.p_value < 0.05).mean() < 0.1 < (fixed.p_value < 0.05).mean()


def test_imbalance_only_warns_and_not_validated(tmp_path):
    import pytest

    imb = make_imbalance(0)[["chrom", "start", "end"]].copy()
    imb["imbalance"] = 0.6
    with pytest.warns(UserWarning, match="hap1_count"):
        out = run(tmp_path, imb, [("chr1", 1_000_000, 2_000_000)])
    assert out["p_value"].isna().all()
    assert not out["cross_validated"].any()


def test_independent_bins_warns_center_ignored(tmp_path):
    import pytest

    imb = make_imbalance(0)
    with pytest.warns(UserWarning, match="center"):
        run(tmp_path, imb, [("chr1", 1_000_000, 2_000_000)], robust=False)


def test_library_default_min_imbalance_matches_cli():
    import inspect

    sig = inspect.signature(cross_reference)
    assert sig.parameters["min_imbalance_thresh"].default == 0.15
