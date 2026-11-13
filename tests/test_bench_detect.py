"""Smoke tests for the ground-truth benchmark harness (benchmarks/bench_detect.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import bench_detect as B  # noqa: E402


def _mu_ratio(kind, n=120, depth=60.0, seed=0):
    """Noise-free check: planted junction pixel is enriched over the reference decay."""
    rng = np.random.default_rng(seed)
    obs, tr = B.simulate(n, depth * 1e4, kind, rng)  # huge depth ~ expectation
    ref = B.efun(B._dist(n), depth * 1e4)
    return obs / ref, tr


@pytest.mark.parametrize("kind", B.SV_KINDS)
def test_planted_junction_is_enriched(kind):
    ratio, tr = _mu_ratio(kind)
    for i, j in tr["pixels"]:
        assert ratio[i, j] > 3.0
    assert tr["b"] - tr["a"] >= 30


def test_deletion_and_dup_change_coverage():
    for kind, lo, hi in (("deletion", 0.4, 0.65), ("dup", 1.3, 1.7)):
        rng = np.random.default_rng(3)
        obs, tr = B.simulate(160, 6e5, kind, rng)
        a, b = tr["cnv"]
        inside = obs.sum(1)[a + 12 : b - 11].mean()
        clean = B.simulate(160, 6e5, "clean", np.random.default_rng(3))[0].sum(1)
        rel = inside / clean[a + 12 : b - 11].mean()
        assert lo < rel < hi


def test_simulation_is_seed_deterministic():
    t = B.mk("main", 200, "mid", "inversion", 0)
    a = B.simulate(200, 60.0, "inversion", np.random.default_rng(B.task_seed(t)))
    b = B.simulate(200, 60.0, "inversion", np.random.default_rng(B.task_seed(t)))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert np.allclose(a[0], a[0].T)


def test_pr_curve_and_matching():
    c = B.pr_curve([(3.0, 1), (2.0, 0), (1.0, 1)], n_pos=2)
    assert c["recall"] == [0.5, 0.5, 1.0]
    assert np.isclose(c["ap"], 0.5 * 1.0 + 0.5 * (2 / 3))
    assert B.cheb(10, 20, [(12, 19)]) == 2
    det, fp, loc = B.eval_1d([[(8, 9)], [(50, 50)]], [11, 30])
    assert det and fp == 1 and loc == 2
    labs = B.pr_labels_1d([(5, [(11, 11)]), (4, [(12, 12)]), (3, [(90, 90)])], [11])
    assert labs == [(5.0, 1), (3.0, 0)]  # duplicate hit dropped


def test_run_map_clean_has_no_v2_calls_but_legacy_forces_hits():
    row, pr = B.run_map(B.mk("main", 200, "mid", "clean", 0))
    assert row["v2_n_pair"] == 0 and row["v2_n_cnv"] == 0
    assert row["leg_n_bins"] >= 1  # fixed 1% contamination always flags bins
    assert row["v2_subset_ok"] == 1 and 0.03 < row["p_lt_0.05"] < 0.07
