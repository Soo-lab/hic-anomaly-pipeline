"""Held-out validation of detect v2 vs legacy on a simulator NOT used for tuning.

Differences from bench_detect.py (the tuning simulator):
  * negative-binomial counts (gamma-Poisson, size r) instead of Poisson
  * power-law decay (d+1)^-a, a in {0.8, 1.3}, and a plateau (floor) variant
  * smooth sinusoidal-eigenvector compartment background (checkerboard) on most maps
  * per-bin iid lognormal bias + low-coverage / dropout bins; maps are ICE-balanced
    (pipeline-realistic) before v2 sees them
  * N in {150, 300, 500}; event sizes 10..0.25N, allele fraction in {0.35, 0.7, 1.0}
  * mixed maps with 2-4 disjoint events, mixed kinds
  * clean maps (>=60) incl. stress nulls (r=2, strong bias)
  * seeds from a different SeedSequence root, detector seed 7

Usage: python benchmarks/bench_heldout.py [out_markdown] [--quick] [--workers N]
Outputs benchmarks/results/heldout_*.csv, heldout_pr_curves.json, heldout_bench.md
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hic_anomaly.detect import detect_anomalies, merge_adjacent_bins, run_ai  # noqa: E402
from hic_anomaly.matrix import expected_by_diagonal, ice_balance  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
TOL = 2
BIN = 100_000
DSEED = 7
ROOT_SEED = 90210771
KINDS = ("deletion", "dup", "inversion", "butterfly")
DECAYS = {"pl0.8": (0.8, 0.0), "pl1.3": (1.3, 0.0), "plateau": (1.0, 0.02)}


def efun(d, depth, decay):
    a, floor = DECAYS[decay]
    return depth * np.maximum((1.0 + d) ** (-a), floor) + depth * 0.002


def layout_mu(n, ref, depth, decay):
    ref = np.asarray(ref)
    pos = np.arange(len(ref))
    dd = efun(np.abs(pos[:, None] - pos[None, :]), depth, decay)
    mu = np.zeros((n, n))
    np.add.at(mu, (ref[:, None], ref[None, :]), dd)
    return mu


def compartments(n, rng):
    """Smooth eigenvector-like signal; contact factor exp(amp * e_i * e_j)."""
    x = np.arange(n)
    e = np.zeros(n)
    for _ in range(3):
        per = rng.uniform(25, 80)
        e += rng.normal() * np.sin(2 * np.pi * (x / per + rng.random()))
    e = np.tanh(1.5 * e / e.std())
    return np.exp(0.35 * np.outer(e, e))


def bias_vec(n, rng, sigma, low_frac):
    b = np.exp(rng.normal(0, sigma, n))
    k = int(low_frac * n)
    if k:
        b[rng.choice(n, k, replace=False)] *= rng.uniform(0.05, 0.3, k)
    return b


def make_events(n, rng, kinds, sizes_hi):
    """Disjoint segments [a,b); returns list of dict(kind,a,b,af)."""
    ev, cur = [], 30
    for k in kinds:
        s = int(rng.integers(10, max(11, sizes_hi)))
        gap = int(rng.integers(15, 40))
        a = cur + gap
        if a + s > n - 30:
            break
        ev.append(dict(kind=k, a=a, b=a + s, af=float(rng.choice([0.35, 0.7, 1.0]))))
        cur = a + s
    return ev


def simulate(n, depth, decay, rng, events, comp=True, r=5.0, sigma=0.4, low=0.03):
    ident = np.arange(n)
    mu_ref = layout_mu(n, ident, depth, decay)
    # build derivative layout for layout-type events (mixing weight = event af, mean over events)
    pat = np.zeros((n, n))
    mu_s = mu_ref.copy()
    for e in events:
        a, b, af, k = e["a"], e["b"], e["af"], e["kind"]
        if k == "deletion":
            der = np.r_[ident[:a], ident[b:]]
            e["pixels"], e["cnv"] = [(a - 1, b)], (a, b - 1)
        elif k == "dup":
            der = np.r_[ident[:b], ident[a:b], ident[b:]]
            e["pixels"], e["cnv"] = [(a, b - 1)], (a, b - 1)
        elif k == "inversion":
            der = np.r_[ident[:a], ident[a:b][::-1], ident[b:]]
            e["pixels"], e["cnv"] = [(a - 1, b - 1), (a, b)], None
        else:
            ii, jj = np.meshgrid(ident, ident, indexing="ij")
            p = np.zeros((n, n))
            q3 = (ii < a) & (jj >= b)
            p[q3] = efun((a - 1 - ii[q3]) + (jj[q3] - b) + 1, depth, decay)
            q4 = (ii >= a) & (jj < b) & (ii < jj)
            p[q4] = efun((ii[q4] - a) + (b - 1 - jj[q4]) + 1, depth, decay)
            p = np.triu(p) + np.triu(p, 1).T
            pat += af * p
            e["pixels"], e["cnv"] = [(a - 1, b), (a, b - 1)], None
            e["bps"] = sorted({c for px in e["pixels"] for c in px})
            continue
        # local replacement: delta between derivative and reference contacts
        mu_s += af * (layout_mu(n, der, depth, decay) - mu_ref)
        e["bps"] = sorted({c for px in e["pixels"] for c in px})
    mu = np.maximum(mu_s + pat, 0.0)
    if comp:
        mu = mu * compartments(n, rng)
    bv = bias_vec(n, rng, sigma, low)
    mu = mu * np.outer(bv, bv)
    lam = np.triu(mu)
    g = rng.gamma(r, 1.0 / r, size=lam.shape)  # mean 1, var 1/r
    obs = rng.poisson(lam * g).astype(float)
    obs = np.triu(obs) + np.triu(obs, 1).T
    return obs


def dist(n):
    return np.abs(np.subtract.outer(np.arange(n), np.arange(n)))


def logoe_ice(obs):
    n = obs.shape[0]
    bal, bias = ice_balance(sp.csr_matrix(obs))
    mask = ~np.isfinite(bias)
    scale = obs.sum() / max(bal.sum(), 1e-12)
    exp = expected_by_diagonal(bal, mask) * scale
    d = dist(n)
    L = np.log2((bal.toarray() * scale + 1) / (exp[d] + 1))
    L[d < 2] = np.nan
    L[mask, :] = np.nan
    L[:, mask] = np.nan
    return L


# ---------------------------------------------------------------- matching / PR
def d1(lo, hi, bps):
    return min(max(lo - b, b - hi, 0) for b in bps) if bps else np.inf


def cheb(i, j, pixels):
    return min(max(abs(i - a), abs(j - b)) for a, b in pixels) if pixels else np.inf


def pr_curve(cands, n_pos):
    if not cands or n_pos == 0:
        return {"precision": [1.0], "recall": [0.0], "ap": 0.0}
    s = np.array([c[0] for c in cands])
    y = np.array([c[1] for c in cands])
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]
    P = tp[last] / (tp[last] + fp[last])
    R = tp[last] / n_pos
    ap = float(np.sum(np.diff(np.r_[0.0, R]) * P))
    return {"precision": P.tolist(), "recall": R.tolist(), "ap": ap}


def pr_labels(cands, events, mode):
    """cands (score, key). First hit per event = 1; repeat hits dropped; other = 0."""
    out, seen = [], set()
    for sc, key in sorted(cands, key=lambda c: -c[0]):
        hit = None
        for k, e in enumerate(events):
            if mode == "2d":
                ok = cheb(key[0], key[1], e["pixels"]) <= TOL
            else:
                ok = min(d1(lo, hi, e["bps"]) for lo, hi in key) <= TOL
            if ok:
                hit = k
                break
        if hit is not None and hit in seen:
            continue
        if hit is not None:
            seen.add(hit)
        out.append((float(sc), int(hit is not None)))
    return out


def run_map(t):
    rng = np.random.default_rng(np.random.SeedSequence(t["seed"]))
    n = t["n"]
    sp_ = t["events_spec"]
    if sp_ is None:
        events = []
    elif sp_[0] == "single":
        events = make_events(n, rng, [sp_[1]], int(0.25 * n))
    else:
        k = int(rng.integers(2, 5))
        events = make_events(n, rng, list(rng.choice(KINDS, k)), int(0.18 * n))
    obs = simulate(n, t["depth"], t["decay"], rng, events, comp=t["comp"], r=t["r"],
                   sigma=t["sigma"], low=t["low"])
    L = logoe_ice(obs)
    cov = obs.sum(1)
    row = {k: t[k] for k in ("scenario", "n", "decay", "depth", "r", "comp", "rep")}
    row["n_events"] = len(events)
    row["kinds"] = ",".join(e["kind"] for e in events)
    row["map_id"] = t["map_id"]
    allpix = [p for e in events for p in e["pixels"]]
    allbps = sorted({b for e in events for b in e["bps"]})
    t0 = time.perf_counter()
    df = detect_anomalies(L, BIN, seed=DSEED, coverage=cov)
    row["v2_sec"] = time.perf_counter() - t0
    pair, cn = df[df.type != "cnv"], df[df.type == "cnv"]
    pc = list(zip(pair.bin_i, pair.bin_j))
    ok_pair = [cheb(i, j, allpix) <= TOL for i, j in pc]
    row["v2_calls"] = len(pc)
    row["v2_pair_fp"] = int(len(pc) - sum(ok_pair))
    cnvs = [e["cnv"] for e in events if e["cnv"]]
    cok = [any(abs(i - c[0]) <= TOL and abs(j - c[1]) <= TOL for c in cnvs)
           for i, j in zip(cn.bin_i, cn.bin_j)]
    row["v2_cnv_calls"], row["v2_cnv_fp"] = len(cok), int(len(cok) - sum(cok))
    ev_rows = []
    for e in events:
        ds = [cheb(i, j, e["pixels"]) for i, j in pc]
        near = [x for x in ds if x <= 10]
        c1 = [min(d1(i, i, e["bps"]), d1(j, j, e["bps"])) <= TOL
              for i, j in zip(df.bin_i, df.bin_j)]
        ev_rows.append(dict(kind=e["kind"], size=e["b"] - e["a"], af=e["af"],
                            v2_2d=int(any(x <= TOL for x in ds)),
                            v2_1d=int(any(c1)),
                            v2_loc=min(near) if near else np.nan,
                            v2_cnv=int(bool(e["cnv"]) and any(
                                abs(i - e["cnv"][0]) <= TOL and abs(j - e["cnv"][1]) <= TOL
                                for i, j in zip(cn.bin_i, cn.bin_j)))))
    # 1D fp for v2 (calls with no bin near any breakpoint)
    row["v2_1d_fp"] = int(sum(min(d1(i, i, allbps), d1(j, j, allbps)) > TOL
                              for i, j in zip(df.bin_i, df.bin_j)))
    dfa = detect_anomalies(L, BIN, seed=DSEED, fdr=1.0)
    pa = dfa[dfa.type != "cnv"]
    pr2 = pr_labels([(s, (i, j)) for s, i, j in zip(pa.score, pa.bin_i, pa.bin_j)], events, "2d")
    pr1 = pr_labels([(s, [(i, i), (j, j)]) for s, i, j in zip(pa.score, pa.bin_i, pa.bin_j)],
                    events, "1d")
    t0 = time.perf_counter()
    flag, sc = run_ai(obs.astype(np.int32), 0.01, DSEED)
    row["leg_sec"] = time.perf_counter() - t0
    regs = merge_adjacent_bins(np.flatnonzero(flag), sc, BIN, "chr1", n * BIN)
    row["leg_calls"] = len(regs)
    row["leg_1d_fp"] = int(sum(min(d1(r["start_bin"], r["end_bin"], [b]) for b in allbps) > TOL
                               if allbps else True for r in regs))
    for e, er in zip(events, ev_rows):
        ds = [d1(r["start_bin"], r["end_bin"], e["bps"]) for r in regs]
        er["leg_1d"] = int(any(x <= TOL for x in ds))
        near = [x for x in ds if x <= 10]
        er["leg_loc"] = min(near) if near else np.nan
    prl = pr_labels([(sc[k], [(k, k)]) for k in range(n)], events, "1d")
    # calibration: thinned pair p-values only available via private spy; skip. Use sig count.
    for er in ev_rows:
        er.update({k: row[k] for k in ("scenario", "n", "decay", "r", "map_id")})
    return row, ev_rows, {"v2_2d": pr2, "v2_1d": pr1, "legacy": prl}


def build_tasks(quick):
    T, mid = [], 0

    def add(scenario, n, decay, rep, spec, depth=60.0, r=5.0, comp=True, sigma=0.4, low=0.03):
        nonlocal mid
        mid += 1
        T.append(dict(scenario=scenario, n=n, decay=decay, depth=depth, r=r, comp=comp, rep=rep,
                      sigma=sigma, low=low, events_spec=spec, map_id=mid,
                      seed=[ROOT_SEED, mid, n, rep]))

    single_reps = 1 if quick else 5
    nclean = 1 if quick else 6
    for n in (150, 300, 500):
        for dec in DECAYS:
            for dep in (25.0, 100.0):
                for kind in KINDS:
                    for rep in range(single_reps):
                        add(f"single_d{int(dep)}", n, dec, rep, ("single", kind), depth=dep)
            for rep in range(nclean):
                add("clean", n, dec, rep, None)
            for rep in range(nclean):  # stress null: heavy overdispersion, strong bias, no comp
                add("clean_stress", n, dec, rep, None, r=2.0, sigma=0.7, low=0.08)
        mreps = 3 if quick else 12
        for rep in range(mreps):
            add("mixed", n, ["pl0.8", "pl1.3", "plateau"][rep % 3], rep, ("mixed",), depth=60.0)
    return T


def prf(tp, fp, npos):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / npos if npos else np.nan
    f = 2 * p * r / (p + r) if p and r and (p + r) > 0 and np.isfinite(p) else 0.0
    return p, r, f


def summarize(maps, evs, prs, keys, cols_extra=True):
    out = []
    for gk, g in maps.groupby(keys, sort=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        ids = set(g.map_id)
        e = evs[evs.map_id.isin(ids)]
        ev_n = len(e)
        r = dict(zip(keys, gk), maps=len(g), events=ev_n)
        if ev_n:
            pm = g[g.n_events > 0]
            for name, tp, fp in (("leg", e.leg_1d.sum(), g.leg_1d_fp.sum()),
                                 ("v2_1d", e.v2_1d.sum(), g.v2_1d_fp.sum()),
                                 ("v2_2d", e.v2_2d.sum(), (g.v2_pair_fp + g.v2_cnv_fp).sum())):
                p, rc, f = prf(tp, fp, ev_n)
                r.update({f"{name}_P": p, f"{name}_R": rc, f"{name}_F1": f})
            r["leg_loc"] = e.leg_loc.mean()
            r["v2_loc"] = e.v2_loc.mean()
            calls = pm.v2_calls.sum() + pm.v2_cnv_calls.sum()
            r["v2_FDP"] = (pm.v2_pair_fp.sum() + pm.v2_cnv_fp.sum()) / calls if calls else 0.0
            for nm, col in (("leg", "legacy"), ("v2_1d", "v2_1d"), ("v2_2d", "v2_2d")):
                c = [x for i in g.index for x in prs[i][col]]
                r[f"{nm}_AP"] = pr_curve(c, ev_n)["ap"]
        r["v2_s_per_map"] = g.v2_sec.mean()
        r["leg_s_per_map"] = g.leg_sec.mean()
        out.append(r)
    return pd.DataFrame(out)


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    o = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(
            ("nan" if not np.isfinite(v) else fmt.format(v)) if isinstance(v, (float, np.floating))
            else str(v) for v in (r[c] for c in cols)) + " |")
    return "\n".join(o)


def platform_info():
    import os
    import platform

    import scipy
    import sklearn

    cpu = platform.processor() or platform.machine()
    try:
        import subprocess

        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, check=False).stdout.strip() or cpu
    except Exception:  # noqa: BLE001
        pass
    return (f"platform: {platform.platform()}; cpu: {cpu}; cores: {os.cpu_count()}; "
            f"python {platform.python_version()}; numpy {np.__version__}; "
            f"scipy {scipy.__version__}; sklearn {sklearn.__version__}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("out_markdown", nargs="?", default=None, help="append markdown summary here")
    ap.add_argument("--quick", action="store_true", help="reduced replicate counts")
    ap.add_argument("--workers", type=int, default=8, help="worker processes (default 8)")
    ns = ap.parse_args()
    quick, workers, out_md = ns.quick, ns.workers, ns.out_markdown
    RES.mkdir(exist_ok=True)
    pre = "heldout_quick_" if quick else "heldout_"
    tasks = build_tasks(quick)
    print(len(tasks), "maps", flush=True)
    t0 = time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(run_map, tasks, chunksize=1)
    else:
        res = [run_map(t) for t in tasks]
    print(f"done {time.time() - t0:.0f}s", flush=True)
    maps = pd.DataFrame([r[0] for r in res])
    evs = pd.DataFrame([x for r in res for x in r[1]])
    prs = [r[2] for r in res]
    maps.to_csv(RES / f"{pre}per_map.csv", index=False)
    evs.to_csv(RES / f"{pre}per_event.csv", index=False)

    ev_maps = maps[maps.n_events > 0]
    A = ev_maps.assign(grp="all")
    single = ev_maps[ev_maps.scenario.str.startswith("single")]
    summarize(A, evs, prs, ["grp"])
    S_scen = summarize(ev_maps, evs, prs, ["scenario", "n"])
    S_decay = summarize(ev_maps, evs, prs, ["scenario", "decay"])
    clean = maps[maps.scenario.str.startswith("clean")]
    # add clean maps' false calls into pooled FP/AP
    pooled = maps.assign(grp="all")
    S_pool = summarize(pooled, evs, prs, ["grp"])
    S_n = summarize(maps, evs, prs, ["n"])
    kind = (evs.groupby("kind").agg(events=("kind", "size"), leg_1d_R=("leg_1d", "mean"),
                                    v2_1d_R=("v2_1d", "mean"), v2_2d_R=("v2_2d", "mean"),
                                    v2_cnv_R=("v2_cnv", "mean"), v2_loc=("v2_loc", "mean"),
                                    leg_loc=("leg_loc", "mean")).reset_index())
    evs["size_bin"] = pd.cut(evs["size"], [0, 15, 30, 60, 200])
    sz = (evs.groupby("size_bin", observed=True).agg(events=("kind", "size"),
          v2_2d_R=("v2_2d", "mean"), v2_1d_R=("v2_1d", "mean"), leg_1d_R=("leg_1d", "mean"))
          .reset_index().astype({"size_bin": str}))
    af = (evs.groupby("af").agg(events=("kind", "size"), v2_2d_R=("v2_2d", "mean"),
          leg_1d_R=("leg_1d", "mean")).reset_index())
    cl = (clean.groupby(["scenario", "n", "decay"]).agg(
        maps=("rep", "size"), v2_calls_per_map=("v2_calls", "mean"),
        v2_cnv_per_map=("v2_cnv_calls", "mean"),
        v2_maps_with_call=("v2_calls", lambda x: (x > 0).mean()),
        leg_regions_per_map=("leg_calls", "mean")).reset_index())
    clean_tot = dict(
        maps=len(clean),
        v2_false_calls_per_map=float((clean.v2_calls + clean.v2_cnv_calls).mean()),
        v2_maps_with_any_call=float(((clean.v2_calls + clean.v2_cnv_calls) > 0).mean()),
        v2_pair_calls_per_map=float(clean.v2_calls.mean()),
        leg_regions_per_map=float(clean.leg_calls.mean()))
    mixed = ev_maps[ev_maps.scenario == "mixed"]
    # curves
    idx = maps.index
    npos = len(evs)
    curves = {
        "legacy IF (per-bin, 1D)": pr_curve([x for i in idx for x in prs[i]["legacy"]], npos),
        "v2 (1D-projected)": pr_curve([x for i in idx for x in prs[i]["v2_1d"]], npos),
        "v2 (pair, 2D)": pr_curve([x for i in idx for x in prs[i]["v2_2d"]], npos),
    }
    for k, c in curves.items():
        m = len(c["precision"])
        if m > 400:
            ii = np.unique(np.round(np.linspace(0, m - 1, 400)).astype(int))
            curves[k] = {"precision": [c["precision"][i] for i in ii],
                         "recall": [c["recall"][i] for i in ii], "ap": c["ap"]}
    (RES / f"{pre}pr_curves.json").write_text(json.dumps(curves))
    for nm, d in (("summary", S_pool), ("by_n", S_n), ("by_scenario", S_scen),
                  ("by_decay", S_decay), ("by_kind", kind), ("clean", cl)):
        d.to_csv(RES / f"{pre}{nm}.csv", index=False)

    P = S_pool.iloc[0]
    sing = summarize(single.assign(grp="s"), evs, prs, ["grp"]).iloc[0]
    mx = summarize(mixed.assign(grp="m"), evs, prs, ["grp"]).iloc[0]
    # q<=0.05 call-level FDP on event maps only
    fdp = P.v2_FDP
    cleanfrac = clean_tot["v2_maps_with_any_call"]
    slack = 0.05 + 2 * np.sqrt(0.05 * 0.95 / max(len(clean), 1))
    crit = [
        ("H1 clean: maps with any v2 call <= %.3f" % slack,
         f"{cleanfrac:.3f} of {len(clean)} maps; {clean_tot['v2_false_calls_per_map']:.3f} calls/map; "
         f"legacy {clean_tot['leg_regions_per_map']:.2f} regions/map",
         "PASS" if cleanfrac <= slack else "FAIL"),
        ("H2 call-level FDP <= 0.10 (q=0.05, pooled event maps)",
         f"{fdp:.3f} (single {sing.v2_FDP:.3f}, mixed {mx.v2_FDP:.3f})",
         "PASS" if fdp <= 0.10 else "FAIL"),
        ("H3 v2 2D F1 >= 0.80 (pooled incl. clean FPs)", f"{P.v2_2d_F1:.3f}",
         "PASS" if P.v2_2d_F1 >= 0.8 else "FAIL"),
        ("H4 v2 1D F1 > legacy F1 and PR-AUC > legacy",
         f"F1 {P.v2_1d_F1:.3f} vs {P.leg_F1:.3f}; AP {P.v2_1d_AP:.3f} vs {P.leg_AP:.3f}",
         "PASS" if P.v2_1d_F1 > P.leg_F1 and P.v2_1d_AP > P.leg_AP else "FAIL"),
        ("H5 v2 2D PR-AUC >= 0.90", f"{P.v2_2d_AP:.3f}", "PASS" if P.v2_2d_AP >= 0.9 else "FAIL"),
        ("H6 v2 localization mean <= 2 bins", f"{P.v2_loc:.2f} (legacy {P.leg_loc:.2f})",
         "PASS" if P.v2_loc <= 2 else "FAIL"),
        ("H7 mixed maps: recall >= 0.7", f"R {mx.v2_2d_R:.3f} P {mx.v2_2d_P:.3f} F1 {mx.v2_2d_F1:.3f}",
         "PASS" if mx.v2_2d_R >= 0.7 else "FAIL"),
    ]
    cdf = pd.DataFrame(crit, columns=["criterion", "measured", "verdict"])
    cdf.to_csv(RES / f"{pre}criteria.csv", index=False)
    cols = ["leg_P", "v2_1d_P", "v2_2d_P", "leg_R", "v2_1d_R", "v2_2d_R", "leg_F1", "v2_1d_F1",
            "v2_2d_F1", "leg_AP", "v2_1d_AP", "v2_2d_AP", "leg_loc", "v2_loc", "v2_FDP",
            "leg_s_per_map", "v2_s_per_map"]
    text = "\n".join([
        f"\n### bench_heldout {date.today()}{' (QUICK)' if quick else ''}",
        f"\n{platform_info()}\n",
        "",
        f"Script `benchmarks/bench_heldout.py`; CSV/JSON `benchmarks/results/{pre}*`. "
        f"{len(maps)} maps ({len(clean)} clean, {len(ev_maps)} with events, {len(evs)} events). "
        "NB noise, power-law/plateau decay, compartments, per-bin bias+dropout, ICE before v2, "
        f"tol +-{TOL}, detector seed {DSEED}, different seed root than bench_detect.",
        "",
        "#### Criteria", "", md(cdf), "",
        "#### Pooled (event maps + clean maps; FPs from clean maps counted)", "",
        md(S_pool[["maps", "events"] + cols]), "",
        "#### By N", "", md(S_n[["n", "maps", "events"] + cols]), "",
        "#### By scenario x N", "", md(S_scen[["scenario", "n", "maps", "events"] + cols]), "",
        "#### By scenario x decay", "",
        md(S_decay[["scenario", "decay", "maps", "events"] + cols]), "",
        "#### Recall by kind", "", md(kind), "",
        "#### Recall by event size (bins)", "", md(sz), "",
        "#### Recall by allele fraction", "", md(af), "",
        "#### Clean maps", "", md(cl), "",
        f"Clean totals: {clean_tot}", "",
    ])
    (RES / f"{pre}bench.md").write_text(text)
    print(text)
    if out_md:
        with open(out_md, "a") as f:
            f.write(text)


if __name__ == "__main__":
    main()
