"""Statistical helpers: BH-FDR, empirical/tail p-values, beta-binomial tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mad_sigma(x):
    """Robust sigma (1.4826 * MAD) ignoring NaN; falls back to std if MAD is 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    s = 1.4826 * np.median(np.abs(x - np.median(x)))
    if s < 1e-12:
        s = float(np.std(x))
    return s if s > 1e-12 else 1.0


def bh_fdr(p):
    """Benjamini-Hochberg q-values. NaN p-values stay NaN and are not counted in m."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(np.clip(p[ok], 0.0, 1.0), method="bh")
    return q


def empirical_pvalue(obs, null, tail_frac=0.01):
    """Right-tail p-value of `obs` against a pooled null sample.

    Inside the null bulk: (1 + #{null >= t}) / (1 + n). Beyond the (1 - tail_frac)
    null quantile an exponential tail is fitted to the exceedances (peaks-over-
    threshold, GPD with shape 0, conservative for lighter tails) so p can go
    below 1/n and still be used with BH over many tests.
    """
    obs = np.asarray(obs, dtype=float)
    null = np.sort(np.asarray(null, dtype=float))
    null = null[np.isfinite(null)]
    n = null.size
    p = np.full(obs.shape, np.nan)
    ok = np.isfinite(obs)
    if n == 0:
        return p
    t = obs[ok]
    p_emp = (1.0 + (n - np.searchsorted(null, t, side="left"))) / (n + 1.0)
    u = null[int((1 - tail_frac) * (n - 1))]
    exc = null[null > u] - u
    if exc.size >= 10:
        theta = max(float(exc.mean()), 1e-12)
        tail = tail_frac * np.exp(-(t - u) / theta)
        p_emp = np.where(t > u, np.minimum(tail, p_emp), p_emp)
    p[ok] = np.clip(p_emp, 1e-300, 1.0)
    return p


def estimate_rho_mom(k, n, mu=0.5, trim=0.05, min_n=10):
    """Beta-binomial overdispersion rho by method of moments on the trimmed bulk.

    Var(k/n) = mu(1-mu)/n * (1 + (n-1) rho). With X2 = n (f-mu)^2 / (mu(1-mu)),
    E[X2] = 1 + (n-1) rho, so rho = sum(X2-1) / sum(n-1). The largest `trim`
    fraction of X2 is dropped so true imbalance does not inflate rho; the
    truncation bias is corrected assuming a normal bulk. Returns rho in [0, 0.5].
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = n >= min_n
    if ok.sum() < 20:
        return 0.0
    f = k[ok] / n[ok]
    nn = n[ok]
    x2 = nn * (f - mu) ** 2 / (mu * (1 - mu))
    keep = x2 <= np.quantile(x2, 1 - trim)
    c = stats.norm.isf(trim / 2)
    # Variance of a standard normal truncated at |z|<c, to undo the trimming bias.
    shrink = 1 - 2 * c * stats.norm.pdf(c) / (1 - trim)
    excess = (x2[keep].mean() / shrink) - 1.0
    denom = (nn[keep] - 1).mean()
    return float(np.clip(excess / denom, 0.0, 0.5))


def estimate_rho_median(k, n, mu=0.5, min_n=10):
    """Beta-binomial rho by median matching (genomic-control style), robust to contamination.

    X2_b = n_b (f_b - mu)^2 / (mu (1-mu)) has median 0.4549 (chi-square, 1 df) when
    X2_b / (1 + (n_b - 1) rho) is chi-square(1). Solves for rho in [0, 0.5] by interpolating
    the monotone median over a log grid. Unlike a trimmed mean, up to ~50% of
    windows can be truly imbalanced without moving it much; but the median is a less
    efficient estimator (~1.5x the variance of the mean for normal data).
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = n >= min_n
    if ok.sum() < 10:
        return 0.0
    x2 = n[ok] * (k[ok] / n[ok] - mu) ** 2 / (mu * (1 - mu))
    nn = n[ok] - 1
    target = float(stats.chi2.median(1))
    if np.median(x2) <= target:
        return 0.0
    grid = np.concatenate([[0.0], np.geomspace(1e-6, 0.5, 400)])
    med = np.array([np.median(x2 / (1 + nn * g)) for g in grid])
    # med decreases with rho; first grid point where it drops to the target
    idx = int(np.argmax(med <= target))
    if idx == 0:
        return 0.0
    f = (med[idx - 1] - target) / max(med[idx - 1] - med[idx], 1e-12)
    return float(grid[idx - 1] + f * (grid[idx] - grid[idx - 1]))


def window_sums(x, width, stride=1):
    """Sums of `x` over sliding windows of `width` (vectorised, via cumulative sums)."""
    x = np.asarray(x, dtype=float)
    if x.size < width:
        return np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[width:] - c[:-width])[::stride]


def block_overdispersion(k_by_chrom, n_by_chrom, width, mu=None, min_blocks=20):
    """Region-scale beta-binomial overdispersion measured on windows of `width` bins.

    Pools sliding windows (stride 1) of summed counts from each chromosome and
    applies `estimate_rho_median` (median matching, robust to imbalanced regions) to the window sums, so rho is the *effective* rho of a
    region of that size: correlated neighbouring bins (shared mapping/reference bias,
    block-correlated allelic ratios) are included automatically, which the bin-level
    rho rescaled by independent-bin variance is not. If `mu` is None the centre is the
    median window fraction (empirical null centre, robust to a minority of imbalanced
    regions; it also absorbs any genome-wide reference bias). Returns
    (mu_hat, rho_eff, n_windows); rho_eff is NaN when fewer than `min_blocks` windows
    exist. Windows overlap, so their count overstates the information; the estimate is
    noisier than for independent blocks, and a minority of imbalanced regions still
    has limited influence (median).
    """
    stride = 1
    ks, ns = [], []
    for k, n in zip(k_by_chrom, n_by_chrom):
        ks.append(window_sums(k, width, stride))
        ns.append(window_sums(n, width, stride))
    k = np.concatenate(ks) if ks else np.zeros(0)
    n = np.concatenate(ns) if ns else np.zeros(0)
    ok = n >= 10
    if ok.sum() < min_blocks:
        return (0.5 if mu is None else mu), np.nan, int(ok.sum())
    if mu is None:
        mu = float(np.median(k[ok] / n[ok]))
        mu = min(max(mu, 0.01), 0.99)
    return mu, estimate_rho_median(k, n, mu=mu), int(ok.sum())


def betabinom_two_sided(k, n, mu=0.5, rho=0.0):
    """Exact two-sided beta-binomial p (sum of pmf <= pmf(k)); binomial if rho ~ 0.

    Vectorised over arrays k, n.
    """
    k = np.atleast_1d(np.asarray(k, dtype=np.int64))
    n = np.atleast_1d(np.asarray(n, dtype=np.int64))
    out = np.full(k.shape, np.nan)
    for idx in np.flatnonzero(n > 0):  # loop over tests with distinct n only
        kk, nn = int(k[idx]), int(n[idx])
        if rho < 1e-6:
            dist = stats.binom(nn, mu)
        else:
            s = 1.0 / rho - 1.0
            dist = stats.betabinom(nn, mu * s, (1 - mu) * s)
        # Large n: two-sided via doubled smaller tail (pmf enumeration is O(n)).
        lo, hi = dist.cdf(kk), dist.sf(kk - 1)
        p_tail = 2.0 * min(lo, hi)
        if nn <= 20000:
            pm = dist.pmf(np.arange(nn + 1))
            p_tail = pm[pm <= pm[kk] * (1 + 1e-9)].sum()
        out[idx] = min(1.0, p_tail)
    return out


def log2_ratio_ci(k, n, conf=0.95):
    """log2 effect size log2((k+.5)/(n-k+.5)) with a Jeffreys-interval CI on k/n."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    eff = np.log2((k + 0.5) / (n - k + 0.5))
    a = (1 - conf) / 2
    lo = stats.beta.ppf(a, k + 0.5, n - k + 0.5)
    hi = stats.beta.ppf(1 - a, k + 0.5, n - k + 0.5)
    odds = lambda q: np.log2(np.clip(q, 1e-12, 1 - 1e-12) / np.clip(1 - q, 1e-12, 1))  # noqa: E731
    return eff, odds(lo), odds(hi)
