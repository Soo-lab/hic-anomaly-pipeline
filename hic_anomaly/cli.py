"""Command-line orchestrator for the Hi-C anomaly pipeline.

Two flows share one entry point:

* ``--method v2`` (default): MAPQ-filtered sparse matrix -> ICE balancing ->
  distance-decay expected -> log2 obs/exp -> calibrated pair-level calls with
  BH q-values -> figures, optional HTML report.
* ``--method legacy``: the original single-pass pipeline (dense raw counts,
  Isolation Forest with a fixed contamination fraction), kept so earlier
  results can be reproduced.
"""

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from hic_anomaly import __version__, viz
from hic_anomaly.detect import bin_features, detect_anomalies, merge_adjacent_bins, run_ai
from hic_anomaly.io_utils import standardize_input_to_bam
from hic_anomaly.matrix import (
    build_matrices_parallel,
    create_contact_matrix,
    expected_by_diagonal,
    ice_balance,
    load_matrix_npz,
    obs_over_exp,
    resolve_chrom_and_length,
    save_matrix_npz,
)
from hic_anomaly.report import write_html_report

logger = logging.getLogger(__name__)

# detect_anomalies works on dense N x N feature maps, so N is capped.
MAX_DETECT_BINS = 3000
# Screening matrices below this many bins give limited detection sensitivity.
SMALL_MATRIX_BINS = 150
ZOOM_PADDING_BP = 500_000

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

LEGACY_FIELDS = [
    "chrom",
    "start_bin",
    "end_bin",
    "start_bp",
    "end_bp",
    "start_Mb",
    "end_Mb",
    "size_Mb",
    "max_ai_score",
]
DETECT_FIELDS = [
    "bin_i",
    "bin_j",
    "start_bp_i",
    "start_bp_j",
    "type",
    "score",
    "p_value",
    "q_value",
]
V2_FIELDS = [
    "chrom",
    "start",
    "end",
    "bin_i",
    "bin_j",
    "start_bp_i",
    "start_bp_j",
    "type",
    "score",
    "p_value",
    "q_value",
]


class PipelineError(Exception):
    """A user-facing failure carrying the process exit code."""

    def __init__(self, message, exit_code=EXIT_ERROR):
        super().__init__(message)
        self.exit_code = exit_code


# Logging


def _configure_logging(verbose=False, quiet=False):
    """Route package log records to stdout (info) and stderr (warnings and up)."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if verbose:
        fmt = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    else:
        fmt = logging.Formatter("%(message)s")

    pkg_logger = logging.getLogger("hic_anomaly")
    for handler in list(pkg_logger.handlers):
        pkg_logger.removeHandler(handler)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(level)
    out_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    out_handler.setFormatter(fmt)

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(max(level, logging.WARNING))
    err_handler.setFormatter(fmt)

    pkg_logger.addHandler(out_handler)
    pkg_logger.addHandler(err_handler)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False


# Argument parsing


def _parse_windows(text):
    try:
        ws = tuple(sorted({int(t) for t in text.split(",") if t.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got '{text}'")
    if not ws or min(ws) < 1:
        raise argparse.ArgumentTypeError("window sizes must be positive integers")
    return ws


def _positive_int(text):
    value = _int_arg(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be an integer >= 1, got {text}")
    return value


def _nonnegative_int(text):
    value = _int_arg(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be an integer >= 0, got {text}")
    return value


def _int_arg(text):
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got '{text}'")


def _float_in(low, high, low_open, high_open):
    """argparse type for a float inside an interval (open or closed at each end)."""

    def parse(text):
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number, got '{text}'")
        below = value <= low if low_open else value < low
        above = value >= high if high_open else value > high
        if not np.isfinite(value) or below or above:
            lo_b = "(" if low_open else "["
            hi_b = ")" if high_open else "]"
            raise argparse.ArgumentTypeError(
                f"must be in {lo_b}{low:g}, {high:g}{hi_b}, got {text}"
            )
        return value

    parse.__name__ = "float"
    return parse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hic-anomaly",
        description="Universal Hi-C Analysis Pipeline: Accepts BAM, SAM, FASTQ, or FASTA.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    io = parser.add_argument_group("input / output")
    io.add_argument(
        "--input",
        required=True,
        help="Input file: BAM, SAM, or R1 FASTQ/FASTA (.gz supported)",
    )
    io.add_argument(
        "--input2",
        default=None,
        help="Input R2 mate file (Required for FASTQ/FASTA)",
    )
    io.add_argument(
        "--ref-fasta",
        default=None,
        help="Reference genome FASTA file (Required if aligning raw FASTQ/FASTA)",
    )
    io.add_argument(
        "--chrom",
        required=True,
        action="append",
        help="Target chromosome (e.g. chr1 or 1). Repeat the flag or give a "
        "comma-separated list for several; 'all' runs every contig in the BAM header",
    )
    io.add_argument(
        "--outdir",
        default="universal_hic_output",
        help="Output directory",
    )
    io.add_argument(
        "--threads",
        type=_positive_int,
        default=4,
        help="Threads for BWA / samtools sorting (default: 4)",
    )

    core = parser.add_argument_group("method")
    core.add_argument(
        "--method",
        choices=["v2", "legacy"],
        default="v2",
        help="v2: balanced log2 obs/exp map, pair-level calls with q-values. "
        "legacy: original raw-count Isolation Forest pipeline (default: v2)",
    )
    core.add_argument(
        "--screen-bin",
        type=_positive_int,
        default=1_000_000,
        help="Screening bin size (default: 1 Mb)",
    )
    core.add_argument(
        "--zoom-bin",
        type=_positive_int,
        default=250_000,
        help="Zoom bin size (default: 250 kb)",
    )
    core.add_argument(
        "--max-zoom",
        type=_nonnegative_int,
        default=5,
        help="Zoom into at most this many top calls per chromosome; 0 disables (default: 5). "
        "v2 zoom uses modern pairing: only pairs with both mates inside the zoom window "
        "(call plus 500 kb padding) are counted",
    )
    core.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=42,
        help="Random seed. Legacy: Isolation Forest. v2: Isolation Forest and "
        "its pair subsample (default: 42)",
    )

    v2 = parser.add_argument_group("v2 options (ignored by --method legacy)")
    v2.add_argument(
        "--min-mapq",
        type=_nonnegative_int,
        default=30,
        help="Drop pairs with MAPQ below this value (default: 30)",
    )
    v2.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Processes for building per-chromosome matrices (default: 4)",
    )
    v2.add_argument(
        "--strict-pairs",
        action="store_true",
        help="Join mates by read name and filter both (exact, matches the original "
        "pipeline; more memory). Default: single-pass pairing from mate coordinates, "
        "which checks the mate's MAPQ only via the MQ tag (default: off)",
    )
    v2.add_argument(
        "--bam-threads",
        type=_positive_int,
        default=None,
        help="htslib decompression threads per BAM reader (default: 1 per worker process, "
        "up to 4 when running serially)",
    )
    v2.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for cached sparse matrices (.npz), keyed by input, "
        "chromosome, bin size and MAPQ",
    )
    v2.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ICE-balance the matrix before computing obs/exp (default: on)",
    )
    v2.add_argument(
        "--ignore-diags",
        type=_nonnegative_int,
        default=2,
        help="Diagonals excluded from ICE balancing (default: 2)",
    )
    v2.add_argument(
        "--oe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise by the per-diagonal expected (log2 obs/exp) before detection "
        "(default: on; p-values are not calibrated with --no-oe)",
    )
    v2.add_argument(
        "--fdr",
        type=_float_in(0.0, 1.0, True, True),
        default=0.05,
        help="Benjamini-Hochberg FDR level for calls (default: 0.05)",
    )
    v2.add_argument(
        "--n-null",
        type=_nonnegative_int,
        default=5,
        help="Accepted for backward compatibility; unused by the current detector "
        "(default: 5)",
    )
    v2.add_argument(
        "--windows",
        type=_parse_windows,
        default=(5, 10),
        metavar="W1,W2",
        help="Feature window half-widths in bins (default: 5,10)",
    )
    v2.add_argument(
        "--triangular",
        action="store_true",
        help="Draw the screening heatmap as a rotated upper triangle",
    )
    v2.add_argument(
        "--no-log-heatmap",
        action="store_true",
        help="Plot raw contact counts instead of log(1 + contacts)",
    )

    legacy = parser.add_argument_group("legacy options (ignored by --method v2)")
    legacy.add_argument(
        "--contamination",
        type=_float_in(0.0, 0.5, True, False),
        default=0.01,
        help="AI contamination fraction (default: 0.01)",
    )

    out = parser.add_argument_group("reporting")
    out.add_argument(
        "--report",
        action="store_true",
        help="Write a self-contained HTML report (report.html) to the output directory",
    )
    verbosity = out.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show warnings and errors",
    )
    return parser


# Shared helpers


def _bam_references(bam_file):
    import pysam

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        return list(bam.references)


def _resolve_chroms(bam_file, chrom_args):
    """Expand repeated / comma-separated --chrom values. Returns ([(name, length)], is_all)."""
    tokens = [t.strip() for value in chrom_args for t in value.split(",") if t.strip()]
    if not tokens:
        raise PipelineError("--chrom needs at least one chromosome name", EXIT_USAGE)
    is_all = any(t.lower() == "all" for t in tokens)
    if is_all:
        tokens = _bam_references(bam_file)
    resolved, seen = [], set()
    for token in tokens:
        try:
            name, length = resolve_chrom_and_length(bam_file, token)
        except ValueError as exc:
            raise PipelineError(str(exc), EXIT_USAGE) from exc
        if name not in seen:
            seen.add(name)
            resolved.append((name, length))
    return resolved, is_all


def _versions():
    versions = {
        "hic_anomaly": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "pysam"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "n/a"
    return versions


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_run_info(outdir, info):
    path = os.path.join(outdir, "run_info.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=_json_default)
        fh.write("\n")
    return path


def _report_summary(info):
    """Flat key/value view of run_info for the HTML report's parameter table."""
    params = info["params"]
    summary = {
        "title": f"Hi-C anomaly report ({info['method']})",
        "input": params["input"],
        "method": info["method"],
        "chromosomes": ", ".join(info["chromosomes"]),
        "screen_bin": params["screen_bin"],
        "zoom_bin": params["zoom_bin"],
        "seed": info["seed"],
    }
    if info["method"] == "v2":
        for key in ("min_mapq", "balance", "ignore_diags", "oe", "fdr", "n_null", "windows"):
            summary[key] = params[key]
    else:
        summary["contamination"] = params["contamination"]
    for key, value in info["stats"]["total"].items():
        summary[f"reads_{key}"] = value
    summary["calls"] = info["n_calls"]
    return summary


# Legacy flow


@contextlib.contextmanager
def _quiet_logger(name, enabled=True):
    """Hide a module's info lines so legacy console output matches the original."""
    target = logging.getLogger(name)
    previous = target.level
    if enabled:
        target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        target.setLevel(previous)


def _run_legacy(args, bam_file, chroms):
    """Original pipeline, one chromosome at a time. Returns (calls_df, figures, stats, outputs)."""
    multi = len(chroms) > 1
    all_regions, figures, outputs, stats = [], [], [], {}

    for chrom_name, chrom_len in chroms:
        logger.info(f"[✓] Target: {chrom_name} ({chrom_len:,} bp)")

        # Build low-res screening matrix
        logger.info(f"--> Building whole-chromosome matrix at {args.screen_bin // 1000}kb...")
        with _quiet_logger("hic_anomaly.matrix", enabled=not args.verbose):
            matrix, total_reads, used_pairs = create_contact_matrix(
                bam_file, chrom_name, chrom_len, args.screen_bin
            )
        logger.info(f"    Total reads: {total_reads:,} | Valid pairs: {used_pairs:,}")
        stats[chrom_name] = {"total": int(total_reads), "used": int(used_pairs)}

        coarse_png = os.path.join(args.outdir, f"hic_{chrom_name}_screening.png")
        viz.plot_matrix(
            matrix,
            chrom_name,
            args.screen_bin,
            coarse_png,
            title="Whole-Chromosome Screening",
            style="legacy",
        )
        figures.append((f"{chrom_name} screening heatmap", coarse_png))
        outputs.append(coarse_png)

        # Anomaly detection
        logger.info(f"--> Running Isolation Forest (contamination={args.contamination})...")
        anomalies, scores = run_ai(matrix, contamination=args.contamination, random_state=args.seed)
        anomaly_bins = np.where(anomalies)[0]
        regions = merge_adjacent_bins(anomaly_bins, scores, args.screen_bin, chrom_name, chrom_len)
        logger.info(f"    Flagged candidate regions: {len(regions)}")
        stats[chrom_name]["calls"] = len(regions)

        calls_tsv = os.path.join(args.outdir, f"anomaly_candidates_{chrom_name}.tsv")
        with open(calls_tsv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEGACY_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(regions)
        outputs.append(calls_tsv)
        all_regions.extend(regions)

        # Fine zoom
        zoom_dir = os.path.join(args.outdir, "zoomed_regions")
        os.makedirs(zoom_dir, exist_ok=True)
        for i, r in enumerate(regions[: args.max_zoom], 1):
            region_id = f"{i:02d}_{int(r['start_Mb'])}Mb"
            if multi:  # avoid file-name collisions between chromosomes
                region_id = f"{chrom_name}_{region_id}"
            png = viz.zoom_region_from_bam(
                bam_file,
                chrom_name,
                chrom_len,
                r["start_bp"],
                r["end_bp"],
                args.zoom_bin,
                padding_bp=ZOOM_PADDING_BP,
                outdir=zoom_dir,
                region_id=region_id,
                style="legacy",
            )
            figures.append((f"{chrom_name} zoom {i}", png))
            outputs.append(png)

    return pd.DataFrame(all_regions, columns=LEGACY_FIELDS), figures, stats, outputs


# v2 flow


def _min_detect_bins(windows):
    """Smallest matrix detect_anomalies will score (mirrors its own size guard)."""
    return 2 * min(windows) + 1 + 2 * max(windows) + 2


def _suggest_bin(chrom_len):
    """Smallest 'round' bin size that keeps a chromosome within MAX_DETECT_BINS."""
    need = int(np.ceil(chrom_len / MAX_DETECT_BINS))
    for step in (1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000):
        if step >= need:
            return step
    return int(np.ceil(need / 1_000_000)) * 1_000_000


def _check_bin_limit(chroms, bin_size):
    too_big = [(c, int(np.ceil(n / bin_size))) for c, n in chroms]
    too_big = [(c, nb) for c, nb in too_big if nb > MAX_DETECT_BINS]
    if not too_big:
        return
    longest = max(n for c, n in chroms if c in dict(too_big))
    listed = ", ".join(f"{c} ({nb:,} bins)" for c, nb in too_big[:5])
    more = f" and {len(too_big) - 5} more" if len(too_big) > 5 else ""
    raise PipelineError(
        f"--screen-bin {bin_size} gives too many bins for dense detection: {listed}{more}. "
        f"The limit is {MAX_DETECT_BINS:,} bins per chromosome (memory grows with bins squared). "
        f"Use a larger --screen-bin, e.g. --screen-bin {_suggest_bin(longest)}.",
        EXIT_USAGE,
    )


def _cache_path(cache_dir, bam_file, chrom, bin_size, min_mapq, strict_pairs=False):
    """One .npz per input + chromosome + bin size + MAPQ (+ pairing mode).

    A changed BAM gets a new key. Strict pairing gets its own file, so the
    default filenames are unchanged.
    """
    st = os.stat(bam_file)
    ident = f"{os.path.abspath(bam_file)}|{st.st_size}|{st.st_mtime_ns}"
    tag = hashlib.sha1(ident.encode()).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(bam_file))[0]
    safe_chrom = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in chrom)
    mode = ".strict" if strict_pairs else ""
    name = f"{stem}.{tag}.{safe_chrom}.bin{bin_size}.mapq{min_mapq}{mode}.npz"
    return os.path.join(cache_dir, name)


def _load_or_build_matrices(args, bam_file, chroms):
    """Return {chrom: (csr, stats, cache_hit)} using the npz cache when enabled."""
    result, missing = {}, []
    for name, _ in chroms:
        cached = None
        if args.cache_dir:
            path = _cache_path(
                args.cache_dir, bam_file, name, args.screen_bin, args.min_mapq, args.strict_pairs
            )
            cached = load_matrix_npz(path)
        if cached is not None:
            logger.info(f"    {name}: loaded cached matrix {path}")
            result[name] = (cached[0], cached[1], True)
        else:
            missing.append(name)

    if missing:
        built = build_matrices_parallel(
            bam_file,
            missing,
            args.screen_bin,
            workers=args.workers,
            min_mapq=args.min_mapq,
            bam_threads=args.bam_threads,
            strict_pairs=args.strict_pairs,
        )
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
        for name in missing:
            mat, stats = built[name]
            result[name] = (mat, stats, False)
            if args.cache_dir:
                path = _cache_path(
                    args.cache_dir,
                    bam_file,
                    name,
                    args.screen_bin,
                    args.min_mapq,
                    args.strict_pairs,
                )
                save_matrix_npz(path, mat, stats)
                logger.debug("cached %s matrix at %s", name, path)
    return result


def _log_oe_map(args, mat, chrom_name):
    """Balance + distance-normalise.

    Returns (dense map with NaN for masked bins, bool mask of bad bins, raw coverage).
    """
    n = mat.shape[0]
    coverage = np.asarray(mat.sum(axis=1)).ravel().astype(float)
    if args.balance:
        balanced, bias = ice_balance(mat, ignore_diags=args.ignore_diags)
        mask = ~np.isfinite(bias)
    else:
        balanced = mat.astype(np.float64)
        mask = coverage <= 0

    if args.oe:
        expected = expected_by_diagonal(balanced, mask=mask)
        # Sparse in, dense out: unobserved pixels get their exact value (not 0)
        # because detect_anomalies densifies this map anyway.
        dense = obs_over_exp(balanced.tocsr(), expected, dense_output=True)
    else:
        dense = np.log2(1.0 + balanced.toarray())

    dense = np.asarray(dense, dtype=np.float64)
    dense[mask, :] = np.nan
    dense[:, mask] = np.nan
    if n and mask.mean() > 0.5:
        logger.warning(
            f"{chrom_name}: {int(mask.sum())} of {n} bins masked as low coverage; "
            "calls will be sparse. Consider a larger --screen-bin or --no-balance."
        )
    return dense, mask, coverage


# Transitional: viz.plot_heatmap gains an ``n_calls`` keyword (label count of calls,
# not anchors). Remove this guard once it is always present.
_HEATMAP_TAKES_N_CALLS = "n_calls" in inspect.signature(viz.plot_heatmap).parameters


def _calls_to_spans(df, bin_size, chrom_len):
    """Intervals to mark on figures: each anchor of a pair call, the full run of a CNV call."""
    spans = []
    for row in df.itertuples(index=False):
        if row.type == "cnv":
            spans.append((int(row.start_bp_i), min(int(row.start_bp_j) + bin_size, chrom_len)))
        else:
            for start in (int(row.start_bp_i), int(row.start_bp_j)):
                spans.append((start, min(start + bin_size, chrom_len)))
    return [{"start_bp": a, "end_bp": b} for a, b in dict.fromkeys(spans)]


def _zoom_calls(bam_file, chrom_name, chrom_len, calls, zoom_bin, zoom_dir, max_zoom, min_mapq=0):
    """Zoom heatmaps for the top calls of one chromosome. Returns [(caption, png)]."""
    figures = []
    if max_zoom <= 0 or len(calls) == 0:
        return figures
    os.makedirs(zoom_dir, exist_ok=True)
    for i, row in enumerate(calls.head(max_zoom).itertuples(index=False), 1):
        start, end = int(row.start), int(row.end)
        lo = max(0, start - ZOOM_PADDING_BP)
        hi = min(chrom_len, end + ZOOM_PADDING_BP)
        zoom_bins = int(np.ceil((hi - lo) / zoom_bin))
        if zoom_bins > MAX_DETECT_BINS:
            logger.warning(
                f"{chrom_name}: skipping zoom {i} ({start:,}-{end:,}); it needs {zoom_bins:,} "
                f"bins at --zoom-bin {zoom_bin}. Use a larger --zoom-bin."
            )
            continue
        png = viz.zoom_region_from_bam(
            bam_file,
            chrom_name,
            chrom_len,
            start,
            end,
            zoom_bin,
            padding_bp=ZOOM_PADDING_BP,
            outdir=zoom_dir,
            region_id=f"{chrom_name}_{i:02d}_{int(start / 1e6)}Mb",
            style="modern",
            min_mapq=min_mapq,
        )
        figures.append((f"{chrom_name} zoom {i}: {row.type} at {start:,}-{end:,}", png))
    return figures


def _run_v2(args, bam_file, chroms, skip_small=False):
    """Calibrated flow. Returns (calls_df, figures, stats, outputs)."""
    min_bins = _min_detect_bins(args.windows)
    if skip_small:
        kept = [(c, n) for c, n in chroms if int(np.ceil(n / args.screen_bin)) >= min_bins]
        if len(kept) < len(chroms):
            logger.info(
                f"    Skipping {len(chroms) - len(kept)} contigs shorter than {min_bins} bins "
                f"at --screen-bin {args.screen_bin}"
            )
        chroms = kept
        if not chroms:
            raise PipelineError(
                f"No contig has at least {min_bins} bins at --screen-bin {args.screen_bin}. "
                "Use a smaller --screen-bin.",
                EXIT_USAGE,
            )
    _check_bin_limit(chroms, args.screen_bin)
    if not args.oe:
        logger.warning(
            "--no-oe: detection assumes a distance-normalised map; p-values and q-values "
            "are not calibrated without it."
        )

    logger.info(
        f"--> Building sparse matrices at {args.screen_bin // 1000}kb "
        f"(min MAPQ {args.min_mapq}) for {len(chroms)} chromosome(s)..."
    )
    matrices = _load_or_build_matrices(args, bam_file, chroms)

    frames, figures, outputs, stats = [], [], [], {}
    zoom_dir = os.path.join(args.outdir, "zoomed_regions")
    for chrom_name, chrom_len in chroms:
        mat, build_stats, cache_hit = matrices[chrom_name]
        n = mat.shape[0]
        logger.info(f"[✓] Target: {chrom_name} ({chrom_len:,} bp, {n:,} bins)")
        logger.info(
            f"    Total reads: {build_stats.get('total', 0):,} | "
            f"Valid pairs: {build_stats.get('used', 0):,}"
        )
        if n < SMALL_MATRIX_BINS:
            logger.warning(
                f"{chrom_name}: {n} bins at --screen-bin {args.screen_bin}. "
                "coarse/small matrix: detection sensitivity is limited; "
                "use a smaller --screen-bin."
            )
        if n < min_bins:
            # Too few bins to score (and matrix.expected_by_diagonal cannot
            # smooth fewer than 5 diagonals), so report no calls for this contig.
            logger.warning(
                f"{chrom_name}: only {n} bins at --screen-bin {args.screen_bin}; detection "
                f"needs at least {min_bins}. Use a smaller --screen-bin. No calls made."
            )
            logoe = None
            coverage = np.asarray(mat.sum(axis=1)).ravel().astype(float)
            mask = coverage <= 0
            calls = pd.DataFrame(columns=DETECT_FIELDS)
            scores_df = None
        else:
            logoe, mask, coverage = _log_oe_map(args, mat, chrom_name)
            logger.info(
                f"--> Detecting anomalies (seed={args.seed}, FDR={args.fdr}, "
                f"{int(mask.sum())} masked bins)..."
            )
            calls, scores_df = detect_anomalies(
                logoe,
                args.screen_bin,
                method="v2",
                seed=args.seed,
                fdr=args.fdr,
                ws=args.windows,
                n_null=args.n_null,
                coverage=coverage,
                return_scores=True,
            )
        calls.insert(0, "chrom", chrom_name)
        calls["start"] = calls["start_bp_i"]
        calls["end"] = np.minimum(calls["start_bp_j"] + args.screen_bin, chrom_len)
        calls = calls[V2_FIELDS]
        logger.info(f"    Calls at q <= {args.fdr}: {len(calls)}")
        frames.append(calls)
        stats[chrom_name] = {
            **{k: int(v) for k, v in build_stats.items()},
            "bins": int(n),
            "masked_bins": int(mask.sum()),
            "cache_hit": bool(cache_hit),
            "calls": int(len(calls)),
        }

        spans = _calls_to_spans(calls, args.screen_bin, chrom_len)
        heat_png = os.path.join(args.outdir, f"hic_{chrom_name}_screening.png")
        heat_kwargs = {"n_calls": len(calls)} if _HEATMAP_TAKES_N_CALLS else {}
        viz.plot_heatmap(
            mat.toarray(),
            chrom_name,
            args.screen_bin,
            heat_png,
            calls=spans,
            log=not args.no_log_heatmap,
            triangular=args.triangular,
            title="Whole-Chromosome Screening",
            **heat_kwargs,
        )
        figures.append((f"{chrom_name} screening heatmap with calls", heat_png))

        # Top panel: per-bin detection score and its call threshold. Bottom panel:
        # the coverage z-score that drives detect_anomalies' CNV calls.
        if scores_df is None:
            score_arr, threshold = None, None
            track = np.zeros(n)
        else:
            score_arr = scores_df["score"].to_numpy(dtype=float)
            threshold = float(scores_df["threshold"].iloc[0]) if len(scores_df) else np.nan
            track = bin_features(logoe, args.windows, coverage)["coverage_z"].to_numpy()
        track_png = os.path.join(args.outdir, f"hic_{chrom_name}_scores.png")
        viz.plot_score_track(
            {"anomaly score": score_arr, "coverage z": np.nan_to_num(track)},
            {"anomaly score": threshold},
            args.screen_bin,
            track_png,
            calls=spans,
            chrom=chrom_name,
            direction={"anomaly score": "above", "coverage z": "both"},
        )
        figures.append((f"{chrom_name} per-bin anomaly score and coverage z", track_png))
        outputs.extend([heat_png, track_png])

        zooms = _zoom_calls(
            bam_file,
            chrom_name,
            chrom_len,
            calls,
            args.zoom_bin,
            zoom_dir,
            args.max_zoom,
            min_mapq=args.min_mapq,
        )
        figures.extend(zooms)
        outputs.extend(png for _, png in zooms)

    non_empty = [f for f in frames if len(f)]
    if non_empty:
        all_calls = pd.concat(non_empty, ignore_index=True)
    else:
        all_calls = pd.DataFrame(columns=V2_FIELDS)
    calls_csv = os.path.join(args.outdir, "anomaly_calls.csv")
    all_calls.to_csv(calls_csv, index=False)
    outputs.append(calls_csv)
    return all_calls, figures, stats, outputs


# Main Orchestrator


def _run(args):
    started = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Standardize any format to sorted, indexed BAM
    bam_file = standardize_input_to_bam(args)

    # 2. Extract reference specs
    chroms, is_all = _resolve_chroms(bam_file, args.chrom)

    # 3-5. Matrix, detection, figures
    if args.method == "legacy":
        calls, figures, stats, outputs = _run_legacy(args, bam_file, chroms)
    else:
        calls, figures, stats, outputs = _run_v2(args, bam_file, chroms, skip_small=is_all)

    totals = {}
    for per_chrom in stats.values():
        for key, value in per_chrom.items():
            if key not in ("bins", "masked_bins", "cache_hit", "calls"):
                totals[key] = totals.get(key, 0) + value

    info = {
        "tool": "hic-anomaly",
        "method": args.method,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - started, 3),
        "command": sys.argv,
        "versions": _versions(),
        "seed": args.seed,
        "params": {**vars(args), "windows": list(args.windows)},
        "bam": bam_file,
        "chromosomes": sorted(stats, key=[c for c, _ in chroms].index),
        "stats": {"total": totals, "per_chromosome": stats},
        "n_calls": int(len(calls)),
        "outputs": [os.path.relpath(p, args.outdir) for p in outputs],
    }
    if args.method == "v2":
        info["notes"] = [
            "q-values are Benjamini-Hochberg adjusted within each chromosome, separately "
            "for pair and coverage (CNV) tests. --fdr is not controlled across chromosomes.",
            "Calls are statistical outliers, not validated structural variants.",
        ]
    if args.report:
        report_path = write_html_report(args.outdir, _report_summary(info), figures, calls)
        info["outputs"].append(os.path.relpath(report_path, args.outdir))
        logger.info(f"    Report: {report_path}")
    _write_run_info(args.outdir, info)

    logger.info(f"\n[✓] Pipeline finished successfully. Results saved in: {args.outdir}")
    return EXIT_OK


def main(argv=None):
    """Console entry point. Returns 0 on success; exits non-zero on failure."""
    args = build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)
    logger.debug("Arguments: %s", vars(args))

    try:
        return _run(args)
    except PipelineError as exc:
        logger.error(f"Error: {exc}")
        sys.exit(exc.exit_code)
    except (ValueError, OSError, RuntimeError) as exc:
        logger.debug("Traceback:", exc_info=True)
        logger.error(f"Error: {exc}")
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    sys.exit(main())
