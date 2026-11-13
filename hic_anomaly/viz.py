"""Figures for the Hi-C anomaly pipeline.

Everything here is display-only: input matrices and scores are never modified.
Large matrices are block-averaged for drawing (see ``_downsample``); the title
states the factor when that happens.

Two looks are available where the original pipeline had a figure:

* ``style="legacy"`` reproduces the original ``plot_matrix`` exactly
  (``Reds``, bin-index axes, 10x10 in, 300 dpi).
* ``style="modern"`` draws the publication figure (``plot_heatmap``): colour
  scale in contacts, genomic axes in Mb/kb, called regions marked on the axes.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pysam  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from mpl_toolkits.axes_grid1.axes_size import Fixed  # noqa: E402

# Okabe-Ito colour-blind-safe palette.
OKABE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}
MAX_DISPLAY = 1500  # display-only downsample target (bins per side)
STYLES = ("legacy", "modern")

# White -> yellow -> red -> black ("hic_fall", the default contact scale).
# Lightness falls roughly monotonically, but hue changes along the way, so it is
# NOT perceptually uniform. For uniform steps pass cmap="cividis" (CVD-safe) or
# cmap="viridis" to plot_heatmap; both are matplotlib built-ins.
_FALL = LinearSegmentedColormap.from_list(
    "hic_fall",
    [
        "#FFFFFF",
        "#FFFFCC",
        "#FFEDA0",
        "#FED976",
        "#FEB24C",
        "#FD8D3C",
        "#FC4E2A",
        "#E31A1C",
        "#BD0026",
        "#800026",
        "#000000",
    ],
)
_DIVERGING = "RdBu_r"
_MASKED = "#D9D9D9"  # NaN (masked / unmappable) bins
_CALL = OKABE["blue"]  # the one colour for called regions (maps, strips, tracks)
_SCORE = "#4D4D4D"  # per-bin score lines: neutral, so they never read as calls
_THRESHOLD = "#000000"  # threshold: black dashed, distinct from every call marker
_GRID = "#B0B0B0"

# One look for every modern figure. DejaVu Sans ships with matplotlib, so the
# output does not depend on fonts installed on the machine.
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}
_DPI = 200


def _modern():
    return plt.rc_context(_RC)


def _check_style(style):
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")


# Helpers (display only)


def _downsample(matrix, max_size=MAX_DISPLAY):
    """Block-mean for DISPLAY only. Returns (float array, factor). Input untouched.

    Accepts dense arrays or scipy sparse matrices (sparse input is binned
    without ever being densified at full resolution). A trailing partial block
    is averaged over the bins it does contain; NaN (masked) bins are ignored
    within a block.
    """
    n = matrix.shape[0]
    f = max(int(np.ceil(n / max_size)), 1)
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        if f == 1:
            return coo.toarray().astype(float), 1
        m = int(np.ceil(n / f))
        tot = sp.coo_matrix(
            (coo.data.astype(float), (coo.row // f, coo.col // f)), shape=(m, m)
        ).toarray()
        sizes = np.diff(np.append(np.arange(0, n, f), n)).astype(float)
        return tot / np.outer(sizes, sizes), f
    m = np.asarray(matrix)
    if f == 1:
        return np.array(m, dtype=float), 1
    edges = np.arange(0, n, f)

    def block_sum(a):
        rows = np.add.reduceat(a, edges, axis=0, dtype=np.float64)
        return np.add.reduceat(rows, edges, axis=1)

    if np.issubdtype(m.dtype, np.floating) and not np.isfinite(m).all():
        finite = np.isfinite(m)
        tot = block_sum(np.where(finite, m, 0.0))
        cnt = block_sum(finite)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = tot / cnt
        out[cnt == 0] = np.nan
        return out, f
    sizes = np.diff(np.append(edges, n)).astype(float)
    return block_sum(m) / np.outer(sizes, sizes), f


def _call_intervals_bp(calls):
    """Normalise calls to a list of (start_bp, end_bp).

    Accepts None, a DataFrame, a list of dicts (``start_bp``/``end_bp`` or
    ``start``/``end``) or a list of (start_bp, end_bp) pairs.
    """
    if calls is None:
        return []
    if hasattr(calls, "to_dict"):
        calls = calls.to_dict("records")
    out = []
    for c in calls:
        if isinstance(c, dict):
            a = c["start_bp"] if "start_bp" in c else c["start"]
            b = c["end_bp"] if "end_bp" in c else c["end"]
        else:
            a, b = c[0], c[1]
        a, b = float(a), float(b)
        if np.isfinite(a) and np.isfinite(b):
            out.append((min(a, b), max(a, b)))
    return out


def _call_intervals_mb(calls):
    """Same as ``_call_intervals_bp`` but in Mb."""
    return [(a / 1e6, b / 1e6) for a, b in _call_intervals_bp(calls)]


def _unit(end_bp):
    """(divisor, label) for an axis ending at ``end_bp``: Mb from 1 Mb up, kb below."""
    return (1e6, "Mb") if end_bp >= 1e6 else (1e3, "kb")


def _ticks(nbins=7):
    """Round genomic ticks (1, 2, 2.5, 5 steps) that include the axis ends when round."""
    return MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)


def _halo():
    """White outline under a line so it reads on both light and dark pixels."""
    return [patheffects.Stroke(linewidth=2.6, foreground="white"), patheffects.Normal()]


def _bin_label(bin_size):
    if bin_size >= 1e6:
        return f"{bin_size / 1e6:g} Mb"
    if bin_size >= 1e3:
        return f"{bin_size / 1e3:g} kb"
    return f"{bin_size:g} bp"


def _log_ticks(lo, hi):
    """Colour-bar ticks at 1, 2, 5 x 10^k inside [lo, hi]; decades only when crowded.

    Always returns at least two values so a narrow range never shows a lone tick.
    """
    if not (lo > 0 and hi > lo):
        return [lo, hi]

    def gen(mults):
        out = []
        for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
            for m in mults:
                v = m * 10.0**e
                if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                    out.append(v)
        return out

    vals = gen((1, 2, 5))
    if len(vals) > 6:
        vals = gen((1,))
    return vals if len(vals) >= 2 else [lo, hi]


def _pos_label(chrom, unit):
    return f"{chrom} position ({unit})".strip()


def _save(fig, out, dpi=_DPI):
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def _despine(ax, keep=("left", "bottom")):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _color_scale(disp, log, diverging, vmin, vmax, cmap):
    """Return (drawable array, norm, cmap, colorbar extend). ``disp`` untouched."""
    finite = disp[np.isfinite(disp)]
    if diverging:
        cm = plt.get_cmap(cmap or _DIVERGING).with_extremes(bad=_MASKED)
        lim = vmax
        if lim is None:
            lim = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
        lim = lim if lim > 0 else 1.0
        lo = -lim if vmin is None else vmin
        over = finite.size and (finite.max() > lim or finite.min() < lo)
        return np.ma.masked_invalid(disp), Normalize(lo, lim), cm, "both" if over else "neither"

    base = _FALL if cmap in (None, "hic_fall") else plt.get_cmap(cmap)
    pos = finite[finite > 0]
    if log:
        hi = vmax if vmax is not None else (float(np.percentile(pos, 99.9)) if pos.size else 10.0)
        lo = vmin if vmin is not None else (float(pos.min()) if pos.size else 1.0)
        lo = max(lo, hi / 1e5)
        if hi <= lo:
            hi = lo * 10.0
        # Empty pixels are drawn at the bottom of the scale (not masked); LogNorm
        # cannot take 0, so they are lifted to ``lo`` which maps to colour 0.
        cm = base.with_extremes(bad=_MASKED)
        data = np.where(disp <= 0, lo, disp)  # NaN stays NaN
        extend = "max" if pos.size and pos.max() > hi * (1 + 1e-9) else "neither"
        return np.ma.masked_invalid(data), LogNorm(lo, hi), cm, extend
    cm = base.with_extremes(bad=_MASKED)
    hi = vmax if vmax is not None else (float(np.percentile(pos, 99)) if pos.size else 1.0)
    lo = 0.0 if vmin is None else vmin
    if hi <= lo:
        hi = lo + 1.0
    extend = "max" if finite.size and finite.max() > hi * (1 + 1e-9) else "neither"
    return np.ma.masked_invalid(disp), Normalize(lo, hi), cm, extend


def _clean_threshold(threshold, n):
    """None, a finite float, or a per-bin float array of length n."""
    if threshold is None:
        return None
    try:
        t = np.asarray(threshold, dtype=float)
    except (TypeError, ValueError):
        return None
    if t.ndim == 0:
        return float(t) if np.isfinite(t) else None
    t = t.ravel()
    if t.size == 1:
        return float(t[0]) if np.isfinite(t[0]) else None
    if t.size == n and np.isfinite(t).any():
        return t
    return None


def _draw_track(ax, scores, threshold, bin_size, start_bp, div, direction, ylabel, spans):
    """One per-bin score panel. Returns legend handles."""
    if direction not in ("above", "below", "both"):
        raise ValueError("direction must be 'above', 'below' or 'both'")
    handles = []
    s = np.zeros(0) if scores is None else np.asarray(scores, dtype=float).ravel()
    x0 = start_bp / div
    x1 = (start_bp + max(s.size, 1) * bin_size) / div
    for a, b in spans:
        ax.axvspan(a / div, b / div, color=_CALL, alpha=0.22, lw=0)
    if spans:
        handles.append(Patch(color=_CALL, alpha=0.22, label="called region"))
    _despine(ax)
    ax.set_xlim(x0, x1)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(_ticks(8))
    if s.size == 0 or not np.isfinite(s).any():
        ax.text(
            0.5,
            0.5,
            "no per-bin scores available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#666666",
            fontsize=9,
        )
        ax.set_yticks([])
        return handles
    x = (start_bp + (np.arange(s.size) + 0.5) * bin_size) / div
    finite = s[np.isfinite(s)]
    if finite.min() < 0 < finite.max():
        ax.axhline(0, color=_GRID, lw=0.6, zorder=1)
    ax.plot(x, s, color=_SCORE, lw=0.9, zorder=2)
    thr = _clean_threshold(threshold, s.size)
    if thr is not None:
        kw = dict(color=_THRESHOLD, ls=(0, (4, 2)), lw=1, zorder=3)
        with np.errstate(invalid="ignore"):
            if direction == "both":
                hit = np.abs(s) > np.abs(thr)
            else:
                hit = s > thr if direction == "above" else s < thr
        if np.ndim(thr):
            (line,) = ax.plot(x, thr, **kw)
            if direction == "both":
                ax.plot(x, -np.abs(thr), **kw)
            label = "threshold (per bin)"
        else:
            line = ax.axhline(thr, **kw)
            if direction == "both" and thr != 0:
                ax.axhline(-thr, **kw)
            if direction == "both":
                label = f"threshold = ±{abs(thr):.3g}"
            else:
                label = f"threshold = {thr:.3g}"
        line.set_label(label)
        handles.append(line)
        if hit.any():
            (pts,) = ax.plot(
                x[hit],
                s[hit],
                ls="none",
                marker="o",
                ms=3,
                mfc=_CALL,
                mec="white",
                mew=0.4,
                zorder=4,
                label=f"beyond threshold (n = {int(hit.sum())})",
            )
            handles.append(pts)
    ax.margins(y=0.08)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    return handles


def _mark_strip(ax, spans, div, vertical=False):
    """Thin lane next to a heatmap showing called intervals (true width plus a
    hairline at the midpoint so sub-pixel calls stay visible)."""
    ax.set_facecolor("#F2F2F2")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(
        left=False, bottom=False, top=False, right=False, labelleft=False, labelbottom=False
    )
    for a, b in spans:
        a, b = a / div, b / div
        mid = (a + b) / 2
        if vertical:
            ax.axhspan(a, b, color=_CALL, lw=0)
            ax.axhline(mid, color=_CALL, lw=0.8)
        else:
            ax.axvspan(a, b, color=_CALL, lw=0)
            ax.axvline(mid, color=_CALL, lw=0.8)


def _render_heatmap(
    fig,
    ax,
    matrix,
    chrom,
    bin_size,
    calls=None,
    log=True,
    triangular=False,
    title="Hi-C contact map",
    start_bp=0,
    cmap=None,
    vmin=None,
    vmax=None,
    diverging=False,
    cbar_label=None,
    max_distance_bp=None,
    scores=None,
    threshold=None,
    score_label="anomaly score",
    score_direction="above",
    unit=None,
    n_calls=None,
    header_detail=None,
):
    """Draw one heatmap (plus strips/colorbar/optional score track) around ``ax``."""
    n = matrix.shape[0]
    disp, f = _downsample(matrix)
    data, norm, cm, extend = _color_scale(disp, log, diverging, vmin, vmax, cmap)
    span_bp = n * bin_size
    div, unit_name = unit or _unit(start_bp + span_bp)
    x0 = start_bp / div
    x1 = (start_bp + span_bp) / div
    x1d = (start_bp + disp.shape[0] * f * bin_size) / div  # extent of drawn blocks
    spans = [
        (max(a, start_bp), min(b, start_bp + span_bp))
        for a, b in _call_intervals_bp(calls)
        if b > start_bp and a < start_bp + span_bp
    ]
    interp = "nearest" if disp.shape[0] <= 800 else "antialiased"
    divider = make_axes_locatable(ax)
    has_track = scores is not None

    if triangular:
        depth = span_bp if not max_distance_bp else min(max_distance_bp, span_bp)
        im = ax.imshow(
            data,
            cmap=cm,
            norm=norm,
            origin="lower",
            extent=(x0, x1d, x0, x1d),
            interpolation=interp,
        )
        # (column pos x, row pos y) -> (midpoint, half separation), then data coords.
        rot = Affine2D.from_values(0.5, 0.5, 0.5, -0.5, 0, 0)
        im.set_transform(rot + ax.transData)
        # Trim to the triangle so partial edge blocks never poke outside it.
        tri = Polygon(
            [(x0, 0), ((x0 + x1) / 2, (x1 - x0) / 2), (x1, 0)], closed=True, transform=ax.transData
        )
        im.set_clip_path(tri)
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, depth / div / 2)
        # Axis runs in half-separation; label it with round separation values.
        sep = _ticks(4).tick_values(0, depth / div)
        ax.set_yticks([t / 2 for t in sep if 0 <= t <= depth / div * (1 + 1e-9)])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{2 * v:g}"))
        ax.set_ylabel(f"Separation ({unit_name})")
        _despine(ax)
        for a, b in spans:
            a, b = a / div, b / div
            ax.add_patch(
                Polygon(
                    [(a, 0), ((a + b) / 2, (b - a) / 2), (b, 0)],
                    closed=True,
                    fill=False,
                    ec=_CALL,
                    lw=1.1,
                    path_effects=_halo(),
                )
            )
        bottom = ax
        if spans:
            strip = divider.append_axes("bottom", size=Fixed(0.09), pad=0.03, sharex=ax)
            _mark_strip(strip, spans, div)
            ax.tick_params(labelbottom=False, bottom=False)
            strip.tick_params(bottom=True, labelbottom=True)
            strip.spines["bottom"].set_visible(True)
            bottom = strip
        cax = divider.append_axes("right", size=Fixed(0.13), pad=0.12)
    else:
        im = ax.imshow(
            data,
            cmap=cm,
            norm=norm,
            origin="upper",
            extent=(x0, x1d, x1d, x0),
            interpolation=interp,
            aspect="equal",
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(x1, x0)
        ax.yaxis.set_major_locator(_ticks())
        ax.set_ylabel(_pos_label(chrom, unit_name))
        for a, b in spans:
            a, b = a / div, b / div
            ax.add_patch(
                Rectangle((a, a), b - a, b - a, fill=False, ec=_CALL, lw=1.1, path_effects=_halo())
            )
        bottom = ax
        if spans:
            top = divider.append_axes("top", size=Fixed(0.09), pad=0.03, sharex=ax)
            _mark_strip(top, spans, div)
            right = divider.append_axes("right", size=Fixed(0.09), pad=0.03, sharey=ax)
            _mark_strip(right, spans, div, vertical=True)
        cax = divider.append_axes("right", size=Fixed(0.13), pad=0.12)

    cbar = fig.colorbar(im, cax=cax, extend=extend, extendfrac=0.03)
    if cbar_label is None:
        cbar_label = "log2(observed / expected)" if diverging else "Contacts per bin pair"
        if f > 1:
            cbar_label += " (block mean)"
        if triangular and depth < 0.6 * span_bp:  # short bar: wrap the label
            cbar_label = cbar_label.replace(" per ", "\nper ", 1).replace(" (block", "\n(block")
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.6)
    if log and not diverging:
        # Plain numbers (1, 10, 100) rather than 10^k: reads better at small ranges.
        cbar.set_ticks(_log_ticks(im.norm.vmin, im.norm.vmax))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        cbar.ax.yaxis.set_minor_locator(NullLocator())

    if has_track:
        track = divider.append_axes("bottom", size=Fixed(0.95), pad=0.36, sharex=ax)
        bottom.tick_params(labelbottom=False)
        handles = _draw_track(
            track, scores, threshold, bin_size, start_bp, div, score_direction, score_label, []
        )
        if handles:
            track.legend(
                handles=handles,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                ncol=len(handles),
                borderaxespad=0.1,
                columnspacing=1.2,
                handlelength=1.6,
                fontsize=7,
            )
        bottom = track
    bottom.xaxis.set_major_locator(_ticks())
    bottom.set_xlim(x0, x1)
    bottom.set_xlabel(_pos_label(chrom, unit_name))

    header = top if spans and not triangular else ax
    if not title:
        label = str(chrom)
    elif str(chrom) in title:
        label = title
    else:
        label = f"{title}: {chrom}"
    header.set_title(label, loc="left")
    # Detail sits on its own line above the title, so the two never collide.
    parts = [header_detail or f"{_bin_label(bin_size)} bins"]
    if f > 1:
        parts.append(f"shown at {_bin_label(bin_size * f)} ({f}x block mean)")
    if spans:
        n_lab = len(spans) if n_calls is None else int(n_calls)
        parts.append(f"blue: called region (n = {n_lab})")
    header.annotate(
        " \u00b7 ".join(parts),
        xy=(1.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 20),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
        annotation_clip=False,
    )
    return im


# Modern figures


def plot_heatmap(
    matrix,
    chrom,
    bin_size,
    out,
    calls=None,
    log=True,
    triangular=False,
    title="Hi-C contact map",
    *,
    start_bp=0,
    cmap=None,
    vmin=None,
    vmax=None,
    diverging=False,
    cbar_label=None,
    max_distance_bp=None,
    scores=None,
    threshold=None,
    score_label="anomaly score",
    score_direction="above",
    n_calls=None,
    header_detail=None,
):
    """Contact heatmap with genomic axes, colorbar and called regions marked.

    matrix      dense array or scipy sparse matrix (never modified; block-mean
                downsampled for display above ``MAX_DISPLAY`` bins per side).
    calls       regions to mark: DataFrame / list of dicts / (start_bp, end_bp).
    log         logarithmic colour scale (colorbar still reads in contacts).
    triangular  rotate the upper triangle 45 degrees (genome-browser view);
                ``max_distance_bp`` limits how far from the diagonal it reaches.
    start_bp    genomic coordinate of the first bin (for zoomed windows).
    diverging   symmetric diverging scale for signed maps such as log2(obs/exp).
    scores      optional per-bin array drawn as an aligned track underneath,
                with ``threshold`` (see ``plot_score_track``).
    cmap        matplotlib colormap name. Default ``"hic_fall"`` (not perceptually
                uniform); ``"cividis"`` (CVD-safe) or ``"viridis"`` give uniform steps.
    n_calls     number of calls for the "called region (n = ...)" label. Defaults to
                the number of drawn intervals; pass the true call count when
                ``calls`` holds one interval per anchor.
    """
    n = matrix.shape[0]
    with _modern():
        if triangular:
            depth = n * bin_size
            if max_distance_bp:
                depth = min(max_distance_bp, depth)
            ratio = depth / (2.0 * n * bin_size) if n else 0.5
            fig, ax = plt.subplots(
                figsize=(7.2, 7.2 * ratio + (2.1 if scores is not None else 1.0))
            )
        else:
            fig, ax = plt.subplots(figsize=(6.2, 6.6 if scores is not None else 5.6))
        _render_heatmap(
            fig,
            ax,
            matrix,
            chrom,
            bin_size,
            calls=calls,
            log=log,
            triangular=triangular,
            title=title,
            start_bp=start_bp,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            diverging=diverging,
            cbar_label=cbar_label,
            max_distance_bp=max_distance_bp,
            scores=scores,
            threshold=threshold,
            score_label=score_label,
            score_direction=score_direction,
            n_calls=n_calls,
            header_detail=header_detail,
        )
        return _save(fig, out)


def plot_score_track(
    scores,
    threshold=None,
    bin_size=1,
    out=None,
    calls=None,
    chrom="",
    ylabel="anomaly score",
    *,
    start_bp=0,
    direction="above",
    title=None,
):
    """Per-bin score track(s) on a genomic axis.

    scores     per-bin array, or a dict ``{label: array}`` for stacked panels
               sharing the x axis. ``None``, empty or all-NaN input draws an
               annotated empty panel instead of failing; NaN bins leave gaps.
    threshold  ``None`` (no line), a number, a per-bin array, or a dict keyed
               like ``scores``. Non-finite thresholds are ignored.
    direction  which side of the threshold is flagged: ``"above"``,
               ``"below"`` (e.g. Isolation Forest scores) or ``"both"`` (|score|,
               draws +/- threshold; e.g. z-scores). May be a dict keyed like
               ``scores``.
    Returns ``out`` when given, else the open Figure.
    """
    if isinstance(scores, dict):
        panels = list(scores.items()) or [(ylabel, None)]
    else:
        panels = [(ylabel, scores)]
    n = max((0 if s is None else np.size(s)) for _, s in panels)
    spans = _call_intervals_bp(calls)
    span_bp = max([n * bin_size] + [b - start_bp for _, b in spans])
    div, unit_name = _unit(start_bp + span_bp)
    with _modern():
        fig, axes = plt.subplots(
            len(panels), 1, sharex=True, squeeze=False, figsize=(8.0, 1.75 * len(panels) + 0.75)
        )
        seen = {}
        for ax, (label, s) in zip(axes[:, 0], panels):
            thr = threshold.get(label) if isinstance(threshold, dict) else threshold
            way = direction.get(label, "above") if isinstance(direction, dict) else direction
            for h in _draw_track(ax, s, thr, bin_size, start_bp, div, way, label, spans):
                seen.setdefault(
                    h.get_label()
                    if len(panels) == 1
                    else h.get_label().split(" =")[0].split(" (")[0],
                    h,
                )
            ax.set_xlim(start_bp / div, (start_bp + max(span_bp, bin_size)) / div)
        if seen:
            handles = list(seen.values())
            labels = list(seen.keys())
            axes[0, 0].legend(
                handles,
                labels,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.02),
                ncol=len(handles),
                borderaxespad=0,
                columnspacing=1.4,
                handlelength=1.6,
            )
        if title:
            axes[0, 0].set_title(title)
        axes[-1, 0].set_xlabel(_pos_label(chrom, unit_name))
        fig.align_ylabels(axes[:, 0])
        fig.subplots_adjust(hspace=0.18)
        return _save(fig, out) if out else fig


def plot_comparison(
    legacy_calls,
    new_calls,
    chrom_len=None,
    out=None,
    chrom="",
    labels=("legacy", "new"),
    *,
    truth=None,
    matrix=None,
    bin_size=None,
    log=True,
    title=None,
):
    """Legacy vs new calls on a shared genomic axis.

    Without ``matrix``: interval lanes (optionally a ``truth`` lane on top).
    With ``matrix`` and ``bin_size``: the same heatmap twice, side by side,
    the left marked with the legacy calls and the right with the new ones.
    """
    lanes = [
        (labels[0], _call_intervals_bp(legacy_calls), OKABE["orange"]),
        (labels[1], _call_intervals_bp(new_calls), OKABE["blue"]),
    ]
    if truth is not None:
        lanes.insert(0, ("truth", _call_intervals_bp(truth), OKABE["black"]))
    with _modern():
        if matrix is not None:
            if not bin_size:
                raise ValueError("bin_size is required when matrix is given")
            fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
            for ax, (name, spans, _c) in zip(axes, lanes[-2:]):
                _render_heatmap(
                    fig, ax, matrix, chrom, bin_size, calls=spans, log=log, title=f"{name} calls"
                )
            if title:
                fig.suptitle(title, fontsize=11, fontweight="bold", x=0.02, ha="left")
            fig.subplots_adjust(wspace=0.32)
            return _save(fig, out) if out else fig

        end = max([b for _, s, _c in lanes for _, b in s] + [chrom_len or 0])
        div, unit_name = _unit(end)
        fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(lanes) + 0.9))
        k = len(lanes)
        for i, (_name, spans, color) in enumerate(lanes):
            y = k - 1 - i  # first lane on top
            ax.axhspan(y + 0.12, y + 0.88, color="#F2F2F2", lw=0, zorder=0)
            for a, b in spans:
                ax.add_patch(
                    Rectangle((a / div, y + 0.12), (b - a) / div, 0.76, color=color, lw=0, zorder=2)
                )
                # hairline keeps calls narrower than a pixel visible
                ax.plot(
                    [(a + b) / 2 / div] * 2,
                    [y + 0.12, y + 0.88],
                    color=color,
                    lw=0.8,
                    zorder=2,
                    solid_capstyle="butt",
                )
        ax.set_xlim(0, end / div if end > 0 else 1)
        ax.set_ylim(0, k)
        ax.set_yticks(
            [k - 1 - i + 0.5 for i in range(k)], [f"{name} (n = {len(s)})" for name, s, _c in lanes]
        )
        ax.tick_params(left=False)
        ax.xaxis.set_major_locator(_ticks(8))
        _despine(ax, keep=("bottom",))
        ax.set_xlabel(_pos_label(chrom, unit_name))
        if title:
            ax.set_title(title)
        return _save(fig, out) if out else fig


def plot_pr_curves(results_dict, out, title="Precision-recall"):
    """Precision-recall curves.

    results_dict: name -> {"precision": [...], "recall": [...]} with optional
    "ap" (average precision, shown in the legend), "baseline" (prevalence,
    drawn as a dotted chance line) and "point" ((recall, precision) of the
    operating point, drawn as a marker).
    """
    colors = [OKABE[k] for k in ("blue", "vermillion", "green", "purple", "orange", "sky", "black")]
    styles = ["-", "--", "-.", (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1))]
    with _modern():
        fig, ax = plt.subplots(figsize=(4.6, 4.9))
        results = results_dict or {}
        for i, (name, r) in enumerate(results.items()):
            color = colors[i % len(colors)]
            lab = name + (f" (AP = {r['ap']:.2f})" if r.get("ap") is not None else "")
            ax.plot(
                r["recall"],
                r["precision"],
                color=color,
                ls=styles[i % len(styles)],
                lw=1.8,
                label=lab,
            )
            if r.get("point") is not None:
                ax.plot(
                    *r["point"],
                    marker="o",
                    ms=6,
                    mfc=color,
                    mec="white",
                    mew=1.0,
                    ls="none",
                    zorder=5,
                )
            if r.get("baseline") is not None:
                ax.axhline(r["baseline"], color=color, ls=":", lw=0.8, alpha=0.8)
        if not results:
            ax.text(0.5, 0.5, "no curves to show", ha="center", va="center", color="#666666")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ticks = np.linspace(0, 1, 6)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.grid(color=_GRID, lw=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        _despine(ax)
        if results:
            ax.legend(
                loc="upper left", bbox_to_anchor=(0.0, -0.16), borderaxespad=0, handlelength=2.6
            )
        return _save(fig, out)


# Original-pipeline API (legacy look by default)


def plot_matrix(
    matrix, chrom, bin_size, output_path, title="Hi-C Heatmap", style="legacy", **kwargs
):
    """Heatmap with the original pipeline's signature.

    style="legacy" (default) is the original figure, unchanged: ``Reds``,
    bin-index axes, 10x10 in, 300 dpi. style="modern" forwards to
    ``plot_heatmap`` (extra keyword arguments are passed through).
    """
    _check_style(style)
    if style == "modern":
        return plot_heatmap(matrix, chrom, bin_size, output_path, title=title, **kwargs)
    if kwargs:
        raise TypeError(f"style='legacy' takes no extra arguments: {sorted(kwargs)}")
    plt.figure(figsize=(10, 10))
    contact_map = np.log1p(matrix)
    plt.imshow(
        contact_map,
        cmap="Reds",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    plt.title(f"{title}: {chrom} ({bin_size // 1000}kb bins)")
    plt.xlabel("Genomic Bin")
    plt.ylabel("Genomic Bin")
    plt.colorbar(label="log(1 + contacts)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


_ZOOM_CHUNK = 1_000_000


def zoom_matrix_from_bam(bam_path, chrom, start_bp, end_bp, zoom_bin_size, min_mapq=0):
    """Dense int32 contact matrix of [start_bp, end_bp) with bounded memory.

    Same pairing rule as ``matrix.build_contact_matrix_sparse``: each pair is
    counted once, from its leftmost mate (RNEXT/PNEXT), so no per-read-name
    dictionary is kept. Both mates must start inside the window. ``min_mapq``
    applies to the read and, when the MQ tag is present, to its mate.
    Returns (matrix, pairs_used).
    """
    num_bins = int(np.ceil((end_bp - start_bp) / zoom_bin_size))
    flat = np.zeros(num_bins * num_bins, dtype=np.int64)
    buf = np.empty(_ZOOM_CHUNK, dtype=np.int64)
    n = used = 0

    def flush(count):
        keys, cnt = np.unique(buf[:count], return_counts=True)
        flat[keys] += cnt

    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        for r in bam.fetch(chrom, start_bp, end_bp):
            if r.is_unmapped or r.mate_is_unmapped:
                continue
            if r.is_secondary or r.is_supplementary or r.is_duplicate:
                continue
            if not r.is_paired or r.next_reference_id != r.reference_id:
                continue
            if r.mapping_quality < min_mapq or (
                min_mapq > 0 and r.has_tag("MQ") and r.get_tag("MQ") < min_mapq
            ):
                continue
            p1, p2 = r.reference_start, r.next_reference_start
            if p1 > p2 or (p1 == p2 and not r.is_read1):
                continue  # counted from the leftmost mate
            if p1 < start_bp or p2 >= end_bp:
                continue
            buf[n] = ((p1 - start_bp) // zoom_bin_size) * num_bins + (
                (p2 - start_bp) // zoom_bin_size
            )
            n += 1
            used += 1
            if n == _ZOOM_CHUNK:
                flush(n)
                n = 0
    finally:
        bam.close()
    if n:
        flush(n)
    upper = flat.reshape(num_bins, num_bins)
    matrix = upper + upper.T
    matrix[np.diag_indices(num_bins)] = np.diag(upper)
    return matrix.astype(np.int32), used


def _zoom_matrix_legacy(bam_path, chrom, start_bp, end_bp, zoom_bin_size):
    """Original zoom pairing (per-read-name dict, no MAPQ filter). Unchanged."""
    zoom_len = end_bp - start_bp
    num_bins = int(np.ceil(zoom_len / zoom_bin_size))
    matrix = np.zeros((num_bins, num_bins), dtype=np.int32)

    bamfile = pysam.AlignmentFile(bam_path, "rb")
    read_pairs = {}

    try:
        for read in bamfile.fetch(chrom, start_bp, end_bp):
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_duplicate:
                continue
            if read.next_reference_id != read.reference_id:
                continue

            # Skip pairs whose mate starts outside the zoom region
            if not (start_bp <= read.next_reference_start < end_bp):
                continue

            qname = read.query_name
            if qname not in read_pairs:
                read_pairs[qname] = read
                continue

            read1 = read_pairs.pop(qname)
            pos1, pos2 = read1.reference_start, read.reference_start

            bin1 = (pos1 - start_bp) // zoom_bin_size
            bin2 = (pos2 - start_bp) // zoom_bin_size

            if bin1 >= num_bins or bin2 >= num_bins:
                continue

            if bin1 == bin2:
                matrix[bin1, bin2] += 1
            else:
                matrix[bin1, bin2] += 1
                matrix[bin2, bin1] += 1
    finally:
        bamfile.close()
    return matrix


def zoom_region_from_bam(
    bam_path,
    chrom,
    chrom_len,
    start_bp,
    end_bp,
    zoom_bin_size,
    padding_bp,
    outdir,
    region_id,
    style="legacy",
    min_mapq=0,
):
    """Zoomed heatmap of a region (plus padding) straight from the BAM.

    style="legacy" (default): original pairing and original figure, unchanged;
    ``min_mapq`` must stay 0 because the original applied no MAPQ filter.
    style="modern": bounded-memory leftmost-mate pairing with a ``min_mapq``
    filter (``zoom_matrix_from_bam``), drawn with ``plot_heatmap`` on genomic
    axes with the unpadded region marked.
    Returns the PNG path.
    """
    _check_style(style)
    region = (start_bp, end_bp)
    start_bp = max(0, start_bp - padding_bp)
    end_bp = min(chrom_len, end_bp + padding_bp)

    out_img = os.path.join(outdir, f"zoom_{region_id}_{zoom_bin_size // 1000}kb.png")
    if style == "modern":
        matrix, _used = zoom_matrix_from_bam(
            bam_path, chrom, start_bp, end_bp, zoom_bin_size, min_mapq=min_mapq
        )
        a, b = region
        # Region in the title (unpadded, the called interval); the padded window
        # goes on the detail line. Ranges under 50 kb get three decimals.
        dec = 3 if b - a < 5e4 else 2
        title = (
            f"Hi-C zoom: {chrom}:{a / 1e6:.{dec}f}-{b / 1e6:.{dec}f} Mb, "
            f"{_bin_label(zoom_bin_size)} bins"
        )
        plot_heatmap(
            matrix,
            chrom,
            zoom_bin_size,
            out_img,
            calls=[region],
            title=title,
            start_bp=start_bp,
            header_detail=f"window {start_bp / 1e6:.2f}-{end_bp / 1e6:.2f} Mb",
        )
        return out_img

    if min_mapq:
        raise ValueError("min_mapq requires style='modern' (the legacy zoom has no MAPQ filter)")
    matrix = _zoom_matrix_legacy(bam_path, chrom, start_bp, end_bp, zoom_bin_size)
    # No style argument here: the original call, so the output stays identical.
    plot_matrix(
        matrix,
        f"{chrom}:{start_bp:,}-{end_bp:,}",
        zoom_bin_size,
        out_img,
        title="Hi-C Zoomed",
    )
    return out_img
