from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.ensemble import IsolationForest

from hic_anomaly.stats_utils import bh_fdr, empirical_pvalue, mad_sigma

# Feature Extraction & Anomaly Detection


def robust_z(x):
    """Robust z: 0.6745 * (x - median) / MAD (0.6745 = 1 / 1.4826; zeros if MAD ~ 0)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def extract_bin_features(matrix):
    M = matrix.astype(np.float64)
    n = M.shape[0]
    total = M.sum(axis=1)

    local5 = np.zeros(n)
    local10 = np.zeros(n)
    diagonal = np.diag(M).copy()

    for i in range(n):
        a5, b5 = max(0, i - 5), min(n, i + 6)
        a10, b10 = max(0, i - 10), min(n, i + 11)
        local5[i] = M[i, a5:b5].sum()
        local10[i] = M[i, a10:b10].sum()

    distal = np.maximum(total - local10, 0)
    local_distal_ratio = local10 / (distal + 1.0)

    features = np.column_stack([
        np.log1p(total),
        np.log1p(local5),
        np.log1p(local10),
        np.log1p(distal),
        np.log1p(diagonal),
        np.log1p(local_distal_ratio),
        robust_z(total),
        robust_z(local10),
    ])
    return features


def run_ai(matrix, contamination=0.01, random_state=42):
    # Mask unmappable bins (zero contacts) from training to prevent false positives
    total_contacts = matrix.sum(axis=1)
    mappable_indices = np.where(total_contacts > 0)[0]

    features = extract_bin_features(matrix)
    labels = np.ones(matrix.shape[0], dtype=int)
    scores = np.zeros(matrix.shape[0], dtype=float)

    if len(mappable_indices) == 0:
        return labels == -1, scores

    train_features = features[mappable_indices]
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_features)

    raw_scores = -model.decision_function(train_features)
    lo, hi = np.percentile(raw_scores, 1), np.percentile(raw_scores, 99)

    if hi <= lo:
        norm_scores = np.zeros_like(raw_scores)
    else:
        norm_scores = np.clip((raw_scores - lo) / (hi - lo), 0, 1)

    train_preds = model.predict(train_features)
    labels[mappable_indices] = train_preds
    scores[mappable_indices] = norm_scores

    anomaly = labels == -1
    return anomaly, scores


def merge_adjacent_bins(
    anomaly_indices,
    scores,
    bin_size,
    chrom,
    chrom_len,
    merge_gap_bins=1,
):
    if len(anomaly_indices) == 0:
        return []

    anomaly_indices = np.sort(np.asarray(anomaly_indices))
    regions = []

    def make_region(start, end, max_score):
        end_bp = min((end + 1) * bin_size, chrom_len)
        start_bp = start * bin_size
        return {
            "chrom": chrom,
            "start_bin": start,
            "end_bin": end,
            "start_bp": start_bp,
            "end_bp": end_bp,
            "start_Mb": start_bp / 1e6,
            "end_Mb": end_bp / 1e6,
            "size_Mb": (end_bp - start_bp) / 1e6,
            "max_ai_score": max_score,
        }

    start = int(anomaly_indices[0])
    previous = int(anomaly_indices[0])
    max_score = float(scores[previous])

    for idx in anomaly_indices[1:]:
        idx = int(idx)
        if idx - previous <= merge_gap_bins + 1:
            previous = idx
            max_score = max(max_score, float(scores[idx]))
        else:
            regions.append(make_region(start, previous, max_score))
            start = previous = idx
            max_score = float(scores[idx])

    regions.append(make_region(start, previous, max_score))
    return regions


# ---------------------------------------------------------------------------
# v2: calibrated pair-level detection on a log2 obs/exp map
# ---------------------------------------------------------------------------
#
# Assumptions (no causal claims): the input is a (mostly) null log2 obs/exp map,
# i.e. the expected model captures distance decay and bias, and most pixels are
# unaffected. Calls are statistical outliers in local contact structure that are
# *consistent with* a structural rearrangement or CNV; they are not validated SVs.
# Null: values permuted within each diagonal (keeps the per-distance marginal,
# destroys spatial structure). Spatially correlated residual bias in real data
# is not modelled, so p-values can be anti-conservative on poorly balanced maps.

TYPES = ("deletion/dup", "inversion", "translocation-like/butterfly", "cnv")


def _as_dense_logoe(logoe):
    """Dense float64 array with NaN for masked pixels. Sparse: missing entries are 0."""
    if hasattr(logoe, "toarray"):
        logoe = logoe.toarray()
    L = np.array(logoe, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("logoe must be a square matrix")
    return np.where(np.isfinite(L), L, np.nan)


class _Sat:
    """Summed-area tables of values and validity for O(1) box means."""

    def __init__(self, L):
        self.n = L.shape[0]
        V = np.isfinite(L)
        L0 = np.where(V, L, 0.0)
        pad = ((1, 0), (1, 0))
        self.S = np.pad(L0.cumsum(0).cumsum(1), pad)
        self.C = np.pad(V.astype(np.float64).cumsum(0).cumsum(1), pad)

    def box(self, r0, r1, c0, c1):
        """Mean and valid fraction over rows [r0,r1) cols [c0,c1); arrays broadcast."""
        n = self.n
        area = (np.asarray(r1) - r0) * (np.asarray(c1) - c0)
        r0, r1, c0, c1 = (np.clip(a, 0, n) for a in (r0, r1, c0, c1))

        def tot(T):
            return T[r1, c1] - T[r0, c1] - T[r1, c0] + T[r0, c0]

        cnt = tot(self.C)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, tot(self.S) / np.maximum(cnt, 1), np.nan)
        return mean, cnt / np.maximum(area, 1)


def bin_features(logoe, ws=(5, 10), coverage=None):
    """Per-bin features: insulation score + delta, up/down asymmetry, coverage z.

    Returns a DataFrame (one row per bin). Insulation IS_w(i) is the mean log2 obs/exp
    over rows [i-w, i) x cols (i, i+w]; delta is IS minus its rolling median (51 bins).
    """
    L = _as_dense_logoe(logoe)
    n = L.shape[0]
    sat = _Sat(L)
    i = np.arange(n)
    out = {}
    for w in ws:
        IS, vf = sat.box(i - w, i, i + 1, i + w + 1)
        IS = np.where(vf >= 0.5, IS, np.nan)
        filled = np.where(np.isfinite(IS), IS, np.nanmedian(IS) if np.isfinite(IS).any() else 0)
        out[f"insulation_w{w}"] = IS
        out[f"insulation_delta_w{w}"] = np.where(
            np.isfinite(IS), IS - ndimage.median_filter(filled, size=51, mode="nearest"), np.nan
        )
    w = max(ws)
    up, v1 = sat.box(i - 1, i + 2, i - w - 1, i - 1)
    dn, v2 = sat.box(i - 1, i + 2, i + 2, i + w + 2)
    out["asymmetry"] = np.where((v1 >= 0.5) & (v2 >= 0.5), up - dn, np.nan)
    out["coverage_z"] = _coverage_z(L, coverage)
    return pd.DataFrame(out)


def _coverage_z(L, coverage):
    """Robust z of log2 coverage relative to a rolling median (CNV signal)."""
    if coverage is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = np.nanmean(L, axis=1)  # weak proxy only; pass real coverage if available
    else:
        cov = np.asarray(coverage, dtype=float)
        with np.errstate(all="ignore"):
            sig = np.where(cov > 0, np.log2(np.where(cov > 0, cov, 1.0)), np.nan)
    ok = np.isfinite(sig)
    if ok.sum() < 5:
        return np.zeros(len(sig))
    filled = np.where(ok, sig, np.median(sig[ok]))
    resid = sig - ndimage.median_filter(filled, size=51, mode="nearest")
    z = (resid - np.nanmedian(resid)) / mad_sigma(resid)
    return np.where(ok, z, np.nan)


def _diag_kernels(w):
    k = np.eye(2 * w + 1)
    return k, k[::-1]


def _pair_maps(L, ws):
    """Full-matrix feature maps: dict name -> NxN (NaN where box mostly masked)."""
    n = L.shape[0]
    sat = _Sat(L)
    R, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    V = np.isfinite(L).astype(float)
    L0 = np.where(V > 0, L, 0.0)
    maps = {}
    for w in ws:
        K, vf = sat.box(R - w, R + w + 1, J - w, J + w + 1)
        maps[f"block_w{w}"] = np.where(vf >= 0.5, K, np.nan)
        q1, f1 = sat.box(R - w, R, J - w, J)
        q2, f2 = sat.box(R, R + w, J, J + w)
        q3, f3 = sat.box(R - w, R, J, J + w)
        q4, f4 = sat.box(R, R + w, J - w, J)
        good = (np.minimum.reduce([f1, f2, f3, f4])) >= 0.5
        maps[f"bowtie_w{w}"] = np.where(good, (q1 + q2) - (q3 + q4), np.nan)
        main, anti = _diag_kernels(w)
        dm = ndimage.correlate(L0, main, mode="constant") / np.maximum(
            ndimage.correlate(V, main, mode="constant"), 1
        )
        da = ndimage.correlate(L0, anti, mode="constant") / np.maximum(
            ndimage.correlate(V, anti, mode="constant"), 1
        )
        maps[f"antidiag_w{w}"] = np.where(vf >= 0.5, da - dm, np.nan)
    return maps


def _band_edges(n, kmin):
    return np.unique(np.round(np.geomspace(kmin, max(n, kmin + 1), 13)).astype(int))


def _standardize(F, d, edges, ref=None):
    """Robust z per distance band. Returns standardized F and (median, sigma) per band."""
    band = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    if ref is None:
        med = np.zeros((nb, F.shape[1]))
        sig = np.ones((nb, F.shape[1]))
        for b in range(nb):  # <= 12 distance bands, not per-bin
            sel = band == b
            if sel.sum() < 10:
                continue
            for c in range(F.shape[1]):
                col = F[sel, c]
                col = col[np.isfinite(col)]
                if col.size >= 10:
                    med[b, c] = np.median(col)
                    sig[b, c] = mad_sigma(col)
        ref = (med, sig)
    med, sig = ref
    return (F - med[band]) / sig[band], ref


def _pair_stat_table(L, ws, kmin, bf_z, edges, ref=None):
    """Feature matrix over upper-triangle pairs plus component z columns."""
    n = L.shape[0]
    maps = _pair_maps(L, ws)
    iu, ju = np.triu_indices(n, k=kmin)
    names = list(maps)
    F = np.column_stack([maps[k][iu, ju] for k in names])
    Z, ref = _standardize(F, ju - iu, edges, ref)
    return iu, ju, names, Z, ref


def _build(Z, names, iu, ju, bf_z):
    """Model matrix: pair z features + bin-level features of both endpoints."""
    X = np.column_stack([Z, bf_z[iu], bf_z[ju]])
    return np.nan_to_num(X, nan=0.0), np.nan_to_num(Z, nan=-np.inf)


def _zmax(Z, names):
    sel = [k for k, nm in enumerate(names) if nm.startswith(("block", "bowtie"))]
    A = Z[:, sel].copy()
    for j, k in enumerate(sel):
        if names[k].startswith("bowtie"):
            A[:, j] = np.abs(A[:, j])
    with np.errstate(all="ignore"):
        m = np.nanmax(np.where(np.isfinite(A), A, -np.inf), axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def _permute_within_diagonals(L, rng):
    """Shuffle valid upper-triangle values within each diagonal; return symmetric map."""
    n = L.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = L[iu, ju]
    ok = np.isfinite(vals)
    iu, ju, vals = iu[ok], ju[ok], vals[ok]
    d = ju - iu
    base = np.lexsort((np.arange(d.size), d))
    shuf = np.lexsort((rng.random(d.size), d))
    new = np.full_like(vals, np.nan)
    new[base] = vals[shuf]
    P = np.full((n, n), np.nan)
    P[iu, ju] = new
    P[ju, iu] = new
    diag = np.diag(L)
    P[np.diag_indices(n)] = diag
    return P


def _band_median(A, d, edges, kmin):
    """Per-distance-band median of A (upper-triangle values, d >= kmin) broadcast to a map."""
    band = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2)
    out = np.zeros(d.shape)
    for b in range(len(edges) - 1):  # <= 12 bands
        sel = (band == b) & (d >= kmin) & np.isfinite(A)
        if sel.sum() >= 10:
            out[band == b] = np.median(A[sel])
    return out


def _median_polish(L, edges, kmin, iters=4):
    """Residual row/column effect r (symmetric two-way median polish on band-centred L).

    Models L ~ band(d) + r_i + r_j (+ local structure). Removing r kills residual
    per-bin bias (imperfect ICE/KR, copy number) that would otherwise light up whole
    rows/columns. Medians are robust to a few bright event pixels. Because r absorbs any
    per-bin signal that is constant along a row, copy-number-like events are NOT
    visible to the pair track by design; they are called from the coverage track.
    """
    n = L.shape[0]
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    A = L - _band_median(L, d, edges, kmin)
    A = np.where(d >= kmin, A, np.nan)
    r = np.zeros(n)
    for _ in range(iters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rm = np.nanmedian(A, axis=1)
        rm = np.where(np.isfinite(rm), rm, 0.0)
        rm = rm - np.median(rm)
        A = A - rm[:, None] - rm[None, :]
        r += rm
    return r


def _exp_tail_fit(t, q_lo, cens=0.1):
    """Type-II-censored exponential fit of exceedances above the q_lo quantile.

    The top `cens` fraction of exceedances (where true events live) is treated as
    censored at the largest uncensored value, so a few strong events cannot inflate or
    deflate the scale. Returns (u, theta, frac_above_u).
    """
    u = float(np.quantile(t, q_lo))
    ex = np.sort(t[t > u] - u)
    m = ex.size
    r = max(min(int(np.floor((1 - cens) * m)), m - 4), 3)  # censor >= 4 largest (events)
    if m < 8:
        return u, max(float(np.std(t)), 1e-6), m / t.size
    theta = (ex[:r].sum() + (m - r) * ex[r - 1]) / r
    return u, max(float(theta), 1e-9), m / t.size


def _peak_null_pvalues(t, infl=1.0):
    """Right-tail p-values of local-peak heights from a data-driven null.

    Peak heights are maxima of a smooth field, so the null is Gumbel-like with
    map-specific location and scale (overdispersion, compartments and residual bias all
    move them). Two estimators, both robust to a few true events, are combined by taking
    the pointwise LARGEST p (most conservative):

    1. Gumbel fitted by quantile matching on the interquartile range of all peaks
       (beta = IQR / 1.5725, mu = median + 0.3665 beta); the top 25% never enters.
    2. If there are >= 60 peaks: type-II-censored exponential tails (top 10% and at
       least the 4 largest treated as events) on exceedances over the 60th, 75th and
       88th percentiles, which lets the data widen the null when the far tail is heavier
       than the Gumbel body implies.

    No simulator-specific constant is involved; `infl` (default 1) is an optional extra
    user multiplier on the scales. Within the bulk the empirical rank p is used when it
    is larger than the model p.
    """
    t = np.asarray(t, dtype=float)
    srt = np.sort(t)
    emp = (1.0 + (t.size - np.searchsorted(srt, t, side="left"))) / (t.size + 1.0)
    q25, q50, q75 = np.quantile(t, [0.25, 0.5, 0.75])
    beta = max((q75 - q25) / 1.5725, 1e-9) * infl
    mu = q50 + 0.3665 * beta  # Gumbel median = mu - beta ln ln 2
    p = -np.expm1(-np.exp(-(t - mu) / beta))
    u, theta = mu, beta
    if t.size >= 60:
        for q in (0.6, 0.75, 0.88):
            u_, th_, f_ = _exp_tail_fit(t, q)
            pe = np.where(t > u_, f_ * np.exp(-(t - u_) / (th_ * infl)), 1.0)
            p = np.maximum(p, pe)
            if th_ > theta:
                u, theta = u_, th_
    p = np.where(t > q50, np.maximum(p, 0.0), np.maximum(p, emp))
    return np.clip(p, 1e-300, 1.0), u, theta


def _refine_and_type(R, Zb, Tm, t_floor, pi, pj, ws, reach=None, min_gain=1.0):
    """Refine each peak to the nearest strong bow-tie corner; type it by quadrants.

    The breakpoint is the pixel with the largest |bow-tie z| (largest window) within
    Chebyshev `reach` bins (default 2*ws[-1]) of the peak: a junction is a corner of
    the enriched region, whereas a block interior or edge is flat in the bow-tie
    contrast, so the box-score peak can sit deep inside a block while the corner is the
    breakpoint. Candidates must keep T >= t_floor[k] (a fraction of the peak's excess over
    the bulk) so compact blocks are not moved onto unrelated noise corners. Peaks of one
    event therefore collapse onto one corner (deduplicated by the caller). Depends only
    on (R, Zb, Tm, peak), not on fdr. The peak moves only if the corner's |z| exceeds the
    peak's by `min_gain` (so compact blocks keep their box-score peak).

    Quadrants q1..q4 are mean log2 obs/exp in the boxes upper-left, lower-right,
    upper-right, lower-left of the corner. Expected patterns (simulation-derived):
    inversion = q1 and q2 bright (two same-sense junctions); butterfly = q3 and q4
    bright; deletion/dup = a single bright q3 (deletion) or q4 (dup). Not validated on
    real data.
    """
    n = R.shape[0]
    reach = 2 * ws[-1] if reach is None else reach
    ri, rj = np.array(pi, dtype=int), np.array(pj, dtype=int)
    for k in range(len(pi)):
        i, j = int(pi[k]), int(pj[k])
        rr = reach
        r0, r1, c0, c1 = max(0, i - rr), min(n, i + rr + 1), max(0, j - rr), min(n, j + rr + 1)
        g = Zb[r0:r1, c0:c1].copy()
        g = np.where(np.isfinite(g) & (Tm[r0:r1, c0:c1] >= t_floor[k]), g, -np.inf)
        if np.isfinite(g).any():
            a_, b_ = np.unravel_index(np.argmax(g), g.shape)
            z0 = Zb[i, j] if np.isfinite(Zb[i, j]) else 0.0
            if g[a_, b_] >= z0 + min_gain:  # move only for a clearly stronger corner
                ri[k], rj[k] = r0 + a_, c0 + b_
    sat = _Sat(R)
    q = np.zeros((4, len(pi)))
    for w in (ws[0], ws[-1]):
        for k, (r0, r1, c0, c1) in enumerate(
            [(ri - w, ri, rj - w, rj), (ri, ri + w, rj, rj + w),
             (ri - w, ri, rj, rj + w), (ri, ri + w, rj - w, rj)]
        ):
            m, _ = sat.box(r0, r1, c0, c1)
            q[k] += np.nan_to_num(m) / 2
    qmax = q.max(axis=0)
    inv = np.minimum(q[0], q[1])
    bf = np.minimum(q[2], q[3])
    typ = np.full(len(pi), TYPES[0], dtype=object)
    typ[(bf >= 0.5 * qmax) & (bf > 0)] = TYPES[2]
    typ[(inv >= 0.5 * qmax) & (inv > 0) & (inv >= bf)] = TYPES[1]
    return ri, rj, typ


def detect_anomalies(
    logoe,
    bin_size,
    method="v2",
    seed=42,
    fdr=0.05,
    ws=(3, 5, 10),
    n_null=5,
    coverage=None,
    min_sep=None,
    return_scores=False,
    null_inflation=1.0,
    edge_bins=None,
):
    """Call-level anomaly calls from a log2 obs/exp map.

    Pipeline: two-way median polish (removes residual per-bin bias) -> box / bow-tie
    features standardised per distance band (robust median/MAD) -> combined score T
    (mean of max feature z and an Isolation-Forest score z) -> non-maximum suppression
    -> one candidate per local peak. Call-level p-values come from an empirical null
    fitted on the population of ALL peak heights in the same map (censored exponential
    tail with data-driven scale, see `_peak_null_pvalues`), then Benjamini-Hochberg across peaks. Each call is one
    event (not one pixel); its breakpoint is the bow-tie corner within 3 bins of the
    peak. Pixel-level empirical p-values (self-null + exponential tail) are computed
    for diagnostics and calibration checks only.

    Parameters
    ----------
    logoe : (N, N) dense array or scipy sparse; NaN marks masked bins. Sparse input
        treats absent entries as 0 (neutral log2 obs/exp).
    method : "v2" (default) or "legacy" (Isolation Forest on raw-count bin features,
        contamination fixed at 0.01; expects raw counts, returns per-bin rows).
    seed : seeds Isolation Forest subsampling.
    fdr : Benjamini-Hochberg level over candidate peaks (call level).
    n_null : retained for CLI compatibility; unused by the empirical null.
    coverage : optional per-bin coverage vector for CNV calls (separate "cnv" type,
        tested on the coverage track, never on pairs).
    return_scores : also return a per-bin DataFrame (bin, score, threshold, called).
    null_inflation : optional extra multiplier on the data-fitted tail scale (default 1.0 =
        none; the scale is estimated from the map, see `_peak_null_pvalues`).
    edge_bins : pixels with row < edge_bins or column >= N - edge_bins are not tested
        (default ws[-1]). Balancing (ICE/KR) is unreliable at matrix ends and produced
        spurious calls there in simulation; real events within edge_bins of an end are
        missed by design.

    Returns a DataFrame with columns bin_i, bin_j, start_bp_i, start_bp_j, type, score,
    p_value, q_value, sorted by q_value (or (df, scores) if return_scores). Zero rows is
    valid. `type` is a pair type, "cnv" (coverage shift) or "homozygous-deletion-candidate" (>= 5 masked
    bins bounded by unmasked ones: homozygous-deletion candidate / gap; bin_i/bin_j = first/
    last masked bin). Remaining limitations: dup junctions are almost invisible in obs/exp
    pairs (rows are polished away); dups are only found via the cnv track. the peak null assumes most peaks are null and an at
    most exponential upper tail; maps dominated by real structure (many events) or very
    few peaks (small N) are less reliable; spatially structured residual bias that
    survives the polish raises the null scale and costs power rather than error, but
    this is only checked on simulation.

    Sensitivity limit (resolution): the pair track needs enough local peaks (>= 5 after
    non-maximum suppression with a (2*ws[-1]+1)-bin window) to fit the peak null, and the
    CNV scan needs windows of >= 6 bins (`min_len`). With few bins (roughly < 150) a
    ~1 Mb event spans only a few bins, the interior valid area (edge_bins excluded at
    both ends) holds too few peaks, and the call set is empty or unreliable. Example:
    a 20 Mb chromosome at 250 kb (80 bins) gives 0 calls for a 1 Mb translocation-like
    blob and a 1.25 Mb CNV even though both are visible in the scores; at 100 kb (200
    bins) they are found. A warning is emitted when n_bins < 150. This is a documented
    limit, not a bug. Conversely, very sparse maps (many bins, few contacts per pixel)
    have noise-dominated peaks: at 10 kb on 100k pairs several long-span
    "deletion/dup" calls with scores ~8x below the real events and q 0.01-0.05 appear
    next to the real calls; interpret low-score long-span calls with caution (not
    filtered: a span cap would also remove genuine deletion junctions).

    Statistical details (for METHODS):
    * Robust sigma is `mad_sigma` = 1.4826 * median|x - median(x)| (consistent for a
      normal); `robust_z` uses its inverse, 0.6745 * (x - med) / MAD.
    * Pixel-level "bulk" empirical p-value (diagnostics only): inside the null bulk
      p = (1 + #{null >= t}) / (1 + n) (rank p, never 0); beyond the 99th percentile an
      exponential tail extends below 1/n. The same rank p enters the peak null as a floor
      (the larger of model and rank p is used).
    * CNV scan Bonferroni floor: for the coverage-shift track each peak z gets
      p_scan = min(1, 2 * sf(z) * m_eff) where m_eff = sum over scales of
      (#valid windows / max(2, w / 2)), the approximate number of independent
      (window, scale) tests. With >= 12 peaks p = max(p_peak_null, p_scan); with fewer
      p = p_scan alone. BH is applied afterwards (family: all CNV peaks).
    * Masked-block ("homozygous-deletion-candidate") null p0 = fraction of masked bins
      that lie in runs shorter than `min_masked` (floor 1/n, cap 0.5), i.e. the
      per-bin independent masking rate; p = (n - m + 1)(1 - p0) p0^m for a run of m
      bins. Optimistic when masks are clustered; candidates only.
    """
    cols = ["bin_i", "bin_j", "start_bp_i", "start_bp_j", "type", "score", "p_value", "q_value"]
    n0 = logoe.shape[0]
    empty_scores = pd.DataFrame(
        {"bin": np.arange(n0), "score": np.nan, "threshold": np.nan, "called": False}
    )

    def ret(df, sc=None):
        return (df, empty_scores if sc is None else sc) if return_scores else df

    if method == "legacy":
        M = np.asarray(logoe.toarray() if hasattr(logoe, "toarray") else logoe)
        flag, sc = run_ai(M, 0.01, seed)
        idx = np.flatnonzero(flag)
        df = pd.DataFrame(
            {
                "bin_i": idx, "bin_j": idx, "start_bp_i": idx * bin_size,
                "start_bp_j": idx * bin_size, "type": "legacy", "score": sc[idx],
                "p_value": np.nan, "q_value": np.nan,
            },
            columns=cols,
        )
        return ret(df, pd.DataFrame({"bin": np.arange(len(sc)), "score": sc,
                                     "threshold": np.nan, "called": flag}))
    if method != "v2":
        raise ValueError("method must be 'v2' or 'legacy'")

    L = _as_dense_logoe(logoe)
    n = L.shape[0]
    ws = tuple(sorted(ws))
    if n < 150:
        warnings.warn(
            f"detect_anomalies: only {n} bins (< 150): matrix too small/coarse, calls "
            "unreliable (events of ~1 Mb span few bins and too few peaks exist to fit the "
            "null); use a finer bin size.",
            UserWarning,
            stacklevel=2,
        )
    kmin = min_sep if min_sep is not None else 2 * ws[0] + 1
    if n < kmin + 2 * ws[-1] + 2:
        return ret(pd.DataFrame(columns=cols))
    rng = np.random.default_rng(seed)

    edges = _band_edges(n, kmin)
    r = _median_polish(L, edges, kmin)
    R = L - r[:, None] - r[None, :]

    bf = bin_features(R, ws, None)
    bf["coverage_z"] = _coverage_z(L, coverage)
    bf_mat = bf[[c for c in bf.columns if c.startswith(("insulation_delta", "asym"))]].to_numpy()
    bf_z = np.nan_to_num(
        np.column_stack([(c - np.nanmedian(c)) / mad_sigma(c) for c in bf_mat.T])
    )

    iu, ju, names, Z, ref = _pair_stat_table(R, ws, kmin, bf_z, edges)
    zm_obs = _zmax(Z, names)
    edge = ws[-1] if edge_bins is None else int(edge_bins)
    valid = np.isfinite(zm_obs) & (iu >= edge) & (ju < n - edge)
    if valid.sum() < 50:
        return ret(pd.DataFrame(columns=cols))
    X, _ = _build(Z, names, iu, ju, bf_z)

    sub = rng.choice(np.flatnonzero(valid), size=min(20000, valid.sum()), replace=False)
    iforest = IsolationForest(
        n_estimators=100, max_samples=256, contamination="auto", random_state=seed, n_jobs=1
    )
    iforest.fit(X[sub])
    s_obs = -iforest.score_samples(X)
    s_med, s_sig = np.median(s_obs[valid]), mad_sigma(s_obs[valid])
    T = np.where(valid, 0.5 * (zm_obs + (s_obs - s_med) / s_sig), np.nan)

    # Pixel-level empirical p (self-null + tail); diagnostics / calibration only.
    p_pix = empirical_pvalue(T, T[valid])
    q_pix = bh_fdr(p_pix)

    # Peaks (NMS) and the call-level empirical null on the peak population.
    Tm = np.full((n, n), -np.inf)
    Tm[iu, ju] = np.where(valid, T, -np.inf)
    size = 2 * ws[-1] + 1
    is_peak = (Tm == ndimage.maximum_filter(Tm, size=size, mode="constant", cval=-np.inf))
    pk = np.flatnonzero(is_peak[iu, ju] & valid)
    rows = []
    thr = np.nan
    pi_, pj_ = iu[pk], ju[pk]
    if pk.size >= 5:
        tp = T[pk]
        pp, _, _ = _peak_null_pvalues(tp, null_inflation)
        qq = bh_fdr(pp)
        hit = qq <= fdr
        if hit.any():
            thr = float(tp[hit].min())
        sel = np.flatnonzero(hit)
        if sel.size:
            Zb = np.full((n, n), np.nan)
            cb = [c for c, nm in enumerate(names) if nm.startswith("bowtie")]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zbm = np.nanmax(np.abs(Z[:, cb]), axis=1) if cb else np.full(len(iu), np.nan)
            Zb[iu, ju] = np.where(valid, zbm, np.nan)
            Zb = np.where(np.isfinite(Zb) & (np.arange(n)[:, None] < np.arange(n)[None, :]),
                          Zb, np.nan)
            tmed = float(np.median(T[valid]))
            floor = tmed + 0.1 * (tp[sel] - tmed)
            ri, rj, typ = _refine_and_type(R, Zb, Tm, floor, pi_[sel], pj_[sel], ws)
            best = {}
            for k, s_ in enumerate(sel):
                key = (int(ri[k]), int(rj[k]))
                if key[0] < key[1] and (key not in best or pp[s_] < best[key][4]):
                    best[key] = (key[0], key[1], typ[k], float(tp[s_]), float(pp[s_]),
                                 float(qq[s_]))
            rows = list(best.values())

    rows.extend(_cnv_calls(None, L, coverage, fdr))
    df = pd.DataFrame(rows, columns=["bin_i", "bin_j", "type", "score", "p_value", "q_value"])
    df["start_bp_i"] = df["bin_i"] * bin_size
    df["start_bp_j"] = df["bin_j"] * bin_size
    df = df[cols].sort_values("q_value", kind="stable").reset_index(drop=True)
    df.attrs["n_sig_pixels"] = int((np.nan_to_num(q_pix, nan=1.0) <= fdr).sum())
    df.attrs["peak_threshold"] = thr
    if not return_scores:
        return df
    with np.errstate(all="ignore"):
        Tsym = np.full((n, n), -np.inf)
        Tsym[iu, ju] = np.where(valid, T, -np.inf)
        bs = np.maximum(Tsym.max(axis=0), Tsym.max(axis=1))
    bs = np.where(np.isfinite(bs), bs, np.nan)
    called = np.zeros(n, dtype=bool)
    for a, b in zip(df.bin_i, df.bin_j):
        called[a] = called[min(b, n - 1)] = True
    scores = pd.DataFrame({"bin": np.arange(n), "score": bs, "threshold": thr, "called": called})
    return df, scores


def _masked_bins(L, coverage):
    """Bins with no usable data: zero/NaN coverage or (almost) fully NaN row in the map."""
    n = L.shape[0]
    masked = np.isfinite(L).sum(axis=1) < 0.2 * n
    if coverage is not None:
        cov = np.asarray(coverage, dtype=float)
        masked |= ~np.isfinite(cov) | (cov <= 0)
    return masked


def _scan_stat(c1, k1, s, e):
    """Sum and valid count of bins [s, e] from prefix sums (arrays broadcast)."""
    return c1[e + 1] - c1[s], k1[e + 1] - k1[s]


def _cnv_calls(cz, L, coverage, fdr, min_len=6, min_shift=0.3, edge=None, min_masked=5,
               max_scale=256, flank=16):
    """Segment calls from the coverage track, separate from pair calls.

    Two call types, each tested on its own family (BH within family at `fdr`):

    "cnv": copy-number-like coverage shift. Signal = log2 coverage on *unmasked* bins
    (zero coverage / ICE-masked bins are missing, never low values; bins beyond 4 robust
    sigma are also treated as missing, the rest winsorised at 2.5 sigma, so isolated
    dropout/low bins carry bounded or no weight). sigma = MAD of first differences / sqrt 2.
    A stride-1 scan over window scales (6 .. ~n bins, geometric) compares the window with a flank of
    <= `flank` bins on EACH side; both edges must be steps in the same direction
    (intersection-union: z = signed min of the two edge z), so a smooth compartment-scale
    trend or the shadow of a neighbouring event is not a call. Per-scale z is scaled by a
    robust empirical null (max(1, 1.4826 median|z|) over that scale's windows; the centre of
    a flank-step statistic is 0 by construction), absorbing overdispersion and autocorrelation (capped at the <=16-bin scale's value times
    sqrt(w/16), the perfectly-correlated bound, so a large event cannot inflate its own null). Calls need BH q <= fdr over
    per-scale local maxima of |z| (p from `_peak_null_pvalues`), |shift| >= min_shift log2 at both edges, length >= min_len
    and >= 70% valid bins. Overlaps resolved by greedy best-z; boundaries refined by
    maximising |sum - n*flank_mean|/sqrt(n) near the window ends (p stays the scan
    window's). Assumptions: most bins copy-neutral; slow coverage bias not modelled (pass
    bias-corrected coverage); events < ~`flank` bins apart share flanks; segments covering
    more than ~1/3 of the map inflate the per-scale empirical null and are under-called. First/last `edge`
    bins are not scanned.

    "homozygous-deletion-candidate" (homozygous-deletion candidate / assembly gap / centromere): a run of
    >= min_masked masked bins bounded on both sides by >= 5 unmasked bins. Null: each bin
    is masked independently with p0 = fraction of masked bins in short runs (floor 1/n);
    p = (n - m + 1)(1 - p0) p0^m for a run of m. Independence is optimistic for real
    clustered masks (repeats), so these are candidates only. Never reported as pair calls.
    """
    n = L.shape[0]
    masked = _masked_bins(L, coverage)
    rows = []
    edge = max(5, min(15, n // 10)) if edge is None else int(edge)

    # --- masked blocks ---
    fill = masked.copy()
    fill[1:-1] |= masked[:-2] & masked[2:]  # bridge single unmasked bins inside a block
    lab, m = ndimage.label(fill)
    cand = []
    if m:
        lab0, m0 = ndimage.label(masked)
        sizes = np.bincount(lab0.ravel(), minlength=m0 + 1)[1:]
        short = masked & np.isin(lab0, 1 + np.flatnonzero(sizes < min_masked))
        p0 = min(max(short.sum() / n, 1.0 / n), 0.5)
        for r in range(1, m + 1):
            idx = np.flatnonzero(lab == r)
            a, b = int(idx[0]), int(idx[-1])
            ln = int(masked[a:b + 1].sum())
            lo, hi = max(0, a - 5), min(n, b + 6)
            if (ln >= min_masked and ln >= 0.9 * (b - a + 1) and a >= 5 and b <= n - 6
                    and masked[lo:a].mean() <= 0.4 and masked[b + 1:hi].mean() <= 0.4):
                pv = min(1.0, (n - ln + 1) * (1 - p0) * p0 ** ln)
                cand.append((a, b, pv))
    if cand:
        pv = np.array([c[2] for c in cand])
        qv = bh_fdr(pv)
        for (a, b, p_), q_ in zip(cand, qv):
            if q_ <= fdr:
                rows.append((a, b, "homozygous-deletion-candidate", float(-np.log10(max(p_, 1e-300))),
                             float(p_), float(q_)))

    # --- coverage shifts ---
    if coverage is None:
        return rows
    cov = np.asarray(coverage, dtype=float)
    ok = ~masked
    if ok.sum() < max(20, 2 * min_len):
        return rows
    with np.errstate(all="ignore"):
        sig = np.where(ok, np.log2(np.where(cov > 0, cov, 1.0)), np.nan)
    med = np.nanmedian(sig)
    dif = np.diff(sig)
    dif = dif[np.isfinite(dif)]
    sigma = max(mad_sigma(dif) / np.sqrt(2), 1e-3) if dif.size >= 10 else 0.0
    if sigma <= 0:
        return rows
    # winsorise relative to a LOCAL level (9-bin running median keeps steps of >= 5 bins),
    # so isolated dropouts are clipped but a genuine large shift is not
    base = ndimage.median_filter(np.where(ok, sig, med), size=9, mode="nearest")
    resid = np.where(ok, sig - base, 0.0)
    ok = ok & (np.abs(resid) <= 4.0 * sigma)  # extreme low/high bins: treated as missing
    x = np.where(ok, base - med + np.clip(resid, -2.5 * sigma, 2.5 * sigma), 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    k1 = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    scales = [w for w in (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
              if min_len <= w <= min(max_scale, n - 2 * edge - 2 * flank - 10)]
    W = []  # per scale: s, e, z (NaN where invalid), step d, flank mean fm
    for w in scales:
        s = np.arange(0, n - w + 1)
        e = s + w - 1
        f = min(w, flank)  # narrow flanks: a CNV has sharp edges, compartments do not
        ls, le = np.maximum(s - f, 0), s - 1  # left flank [ls, le]
        rs, re_ = e + 1, np.minimum(e + f, n - 1)  # right flank [rs, re_]
        tin, cin = _scan_stat(c1, k1, s, e)
        tl, cl = _scan_stat(c1, k1, ls, np.maximum(le, ls - 1))
        tr, cr = _scan_stat(c1, k1, np.minimum(rs, n), re_)
        cl = np.where(le >= ls, cl, 0.0)
        tl = np.where(le >= ls, tl, 0.0)
        cr = np.where(re_ >= rs, cr, 0.0)
        tr = np.where(re_ >= rs, tr, 0.0)
        cfl = cl + cr
        good = (cin >= 0.7 * w) & (cl >= 0.6 * f) & (cr >= 0.6 * f) & (s >= edge) & (e < n - edge)
        fm = (tl + tr) / np.maximum(cfl, 1)
        mi = tin / np.maximum(cin, 1)
        dl, dr = mi - tl / np.maximum(cl, 1), mi - tr / np.maximum(cr, 1)
        zl = dl / (sigma * np.sqrt(1 / np.maximum(cin, 1) + 1 / np.maximum(cl, 1)))
        zr = dr / (sigma * np.sqrt(1 / np.maximum(cin, 1) + 1 / np.maximum(cr, 1)))
        # intersection-union: BOTH edges must be steps in the same direction
        both = np.sign(zl) == np.sign(zr)
        z = np.where(good & both, np.sign(zl) * np.minimum(np.abs(zl), np.abs(zr)), np.nan)
        good = good & both
        d = np.where(good, np.sign(zl) * np.minimum(np.abs(dl), np.abs(dr)), 0.0)
        W.append((s, e, z, d, fm, good))
    if not W:
        return rows

    Wn = []
    ref = None  # (scale, null sigma) at the largest scale <= 16 (little event contamination)
    for s_, e_, z, d, fm, good in W:
        zc = z[good]
        if zc.size < 10:
            continue
        w_ = int(e_[0] - s_[0] + 1)
        z0, zs = 0.0, max(1.4826 * np.median(np.abs(zc)), 1.0)  # step stat: null centre is 0
        if w_ <= 16:
            ref = (w_, zs)
        elif ref is not None:  # perfectly correlated bins would scale z by sqrt(w/w0): hard cap
            zs = min(zs, ref[1] * np.sqrt(w_ / ref[0]))
        Wn.append((s_, e_, (z - z0) / zs, d, fm, good))
    if not Wn:
        return rows
    # Call-level null: local maxima (along position, per scale) of |z| form the peak
    # population; p from the same data-driven Gumbel/censored-tail null as the pair track,
    # BH over all peaks of all scales.
    allp = []
    for k, (s_, e_, zn, d, fm, good) in enumerate(Wn):
        w_ = int(e_[0] - s_[0] + 1)
        za = np.where(good & np.isfinite(zn), np.abs(zn), 0.0)
        pk = (za == ndimage.maximum_filter1d(za, size=max(5, w_ // 2 * 2 + 1), mode="constant")) \
            & (za > 0)
        allp.extend((float(za[j]), k, int(j)) for j in np.flatnonzero(pk))
    allp.sort(reverse=True)
    used = np.zeros(n, dtype=bool)
    peaks = []  # non-overlapping segment candidates: one representative per event
    for h, k, j in allp:
        a_, b_ = int(Wn[k][0][j]), int(Wn[k][1][j])
        if used[a_:b_ + 1].any():
            continue
        used[a_:b_ + 1] = True
        peaks.append((k, j, h))
    hts = np.array([p_[2] for p_ in peaks])
    # Scan multiplicity: z is the max over ~m_eff roughly independent (window, scale)
    # tests, so the per-peak Gaussian tail is Bonferroni-adjusted by m_eff (windows at
    # scale w overlap ~w/2-fold). With too few peaks to fit a null this is the only p;
    # otherwise the fitted peak null is used and the larger p wins (conservative).
    m_eff = sum(float(g_.sum()) / max(2.0, (e_[0] - s_[0] + 1) / 2.0)
                for s_, e_, _z, _d, _fm, g_ in Wn)
    p_scan = np.minimum(1.0, 2 * stats.norm.sf(hts) * max(m_eff, 1.0))
    if len(peaks) >= 12:
        pp, _, _ = _peak_null_pvalues(hts)
        pp = np.maximum(pp, p_scan)
    else:
        pp = p_scan
    qq = bh_fdr(pp)
    cands = []
    for (k, j, _h), p_, q_ in zip(peaks, pp, qq):
        s_, e_, zn, d, fm, good = Wn[k]
        if abs(d[j]) >= min_shift and np.sign(zn[j]) == np.sign(d[j]):
            cands.append((abs(zn[j]), int(s_[j]), int(e_[j]), float(p_), float(q_),
                          float(np.sign(d[j])), float(fm[j])))
    cands.sort(key=lambda t: -t[0])
    taken = np.zeros(n, dtype=bool)
    for zabs, s, e, p_, q_, sgn_, fm_ in cands:
        if q_ > fdr:
            break
        if taken[s:e + 1].any():
            continue
        R = max(3, (e - s + 1) // 4)
        ss = np.arange(max(edge, s - R), min(n - edge, s + R + 1))
        ee = np.arange(max(edge, e - R), min(n - edge, e + R + 1))
        S_, E_ = np.meshgrid(ss, ee, indexing="ij")
        valid = E_ - S_ + 1 >= min_len
        tot, cnt = _scan_stat(c1, k1, S_, E_)
        stat = np.where(valid & (cnt > 0),
                        sgn_ * (tot - cnt * fm_) / np.sqrt(np.maximum(cnt, 1)), -np.inf)
        a_, b_ = np.unravel_index(np.argmax(stat), stat.shape)
        s2, e2 = int(S_[a_, b_]), int(E_[a_, b_])
        if taken[s2:e2 + 1].any():
            s2, e2 = s, e
        taken[s2:e2 + 1] = True
        rows.append((s2, e2, "cnv", float(zabs), float(p_), float(q_)))
    return rows
