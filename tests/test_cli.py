import csv
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pysam
import pytest

from hic_anomaly import cli
from hic_anomaly.detect import merge_adjacent_bins, run_ai
from tests.synth import make_bam

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROM_LEN = 2_000_000
BIN = 20_000  # 100 bins: above the 33-bin detection minimum
ZOOM = 5_000


def _is_png(path):
    with open(path, "rb") as fh:
        return fh.read(4) == b"\x89PNG"


def _original_matrix(bam_path, chrom, chrom_len, bin_size):
    """The original pipeline's qname-pairing matrix builder, kept as the reference."""
    nb = int(np.ceil(chrom_len / bin_size))
    m = np.zeros((nb, nb), dtype=np.int32)
    pending = {}
    with pysam.AlignmentFile(bam_path) as bam:
        for r in bam.fetch(chrom):
            if r.is_unmapped or r.mate_is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if r.is_duplicate or r.next_reference_id != r.reference_id:
                continue
            if r.query_name not in pending:
                pending[r.query_name] = r
                continue
            first = pending.pop(r.query_name)
            b1, b2 = first.reference_start // bin_size, r.reference_start // bin_size
            m[b1, b2] += 1
            if b1 != b2:
                m[b2, b1] += 1
    return m


@pytest.fixture(scope="module")
def bam(tmp_path_factory):
    path = tmp_path_factory.mktemp("bam") / "synth.bam"
    return make_bam(path, n_pairs=20_000, chrom_len=CHROM_LEN, seed=5)


@pytest.fixture(scope="module")
def bam2(tmp_path_factory):
    path = tmp_path_factory.mktemp("bam2") / "two.bam"
    return make_bam(path, n_pairs=30_000, chrom_len=CHROM_LEN, seed=6, chroms=("chr1", "chr2"))


def _args(bam, outdir, *extra):
    return [
        "--input", bam, "--chrom", "chr1", "--screen-bin", str(BIN),
        "--zoom-bin", str(ZOOM), "--outdir", str(outdir), *extra,
    ]  # fmt: skip


def test_help_exits_zero_and_lists_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--method", "--input2", "--ref-fasta", "--chrom", "--screen-bin", "--zoom-bin",
        "--contamination", "--threads", "--outdir", "--min-mapq", "--workers", "--cache-dir",
        "--no-balance", "--ignore-diags", "--no-oe", "--fdr", "--seed", "--report",
        "--triangular", "--verbose", "--quiet", "--strict-pairs", "--bam-threads",
    ):  # fmt: skip
        assert flag in text


def test_missing_required_args_exit_two():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--chrom", "chr1"])
    assert exc.value.code == 2


def test_v2_end_to_end(bam, tmp_path):
    out = tmp_path / "v2"
    cache = tmp_path / "cache"
    rc = cli.main(_args(bam, out, "--cache-dir", str(cache), "--report", "--workers", "1"))
    assert rc == 0

    calls = pd.read_csv(out / "anomaly_calls.csv")
    assert list(calls.columns) == cli.V2_FIELDS
    assert calls["q_value"].le(0.05).all()
    assert (calls["end"] > calls["start"]).all()
    assert _is_png(out / "hic_chr1_screening.png")
    assert _is_png(out / "hic_chr1_scores.png")
    assert "data:image/png;base64," in (out / "report.html").read_text()

    info = json.loads((out / "run_info.json").read_text())
    assert info["method"] == "v2" and info["seed"] == 42
    assert info["params"]["min_mapq"] == 30 and info["params"]["fdr"] == 0.05
    assert {"hic_anomaly", "python", "numpy", "pysam"} <= set(info["versions"])
    per_chrom = info["stats"]["per_chromosome"]["chr1"]
    assert per_chrom["bins"] == CHROM_LEN // BIN and per_chrom["cache_hit"] is False
    assert per_chrom["used"] > 0 and per_chrom["low_mapq"] > 0 and per_chrom["duplicate"] > 0
    assert info["stats"]["total"]["used"] == per_chrom["used"]
    assert info["n_calls"] == len(calls)
    assert "anomaly_calls.csv" in info["outputs"]
    assert "within each chromosome" in info["notes"][0]
    assert "not controlled across chromosomes" in info["notes"][0]

    # Second run reuses the cached matrix and, with the same seed, the same calls.
    assert len(list(cache.glob("*.chr1.bin20000.mapq30.npz"))) == 1
    out2 = tmp_path / "v2b"
    assert cli.main(_args(bam, out2, "--cache-dir", str(cache), "--triangular")) == 0
    info2 = json.loads((out2 / "run_info.json").read_text())
    assert info2["stats"]["per_chromosome"]["chr1"]["cache_hit"] is True
    assert (out2 / "anomaly_calls.csv").read_text() == (out / "anomaly_calls.csv").read_text()
    assert not (out2 / "report.html").exists()


def test_v2_multi_chrom_and_flags(bam2, tmp_path):
    out = tmp_path / "multi"
    rc = cli.main(
        [
            "--input",
            bam2,
            "--chrom",
            "chr1,chr2",
            "--chrom",
            "1",
            "--screen-bin",
            str(BIN),
            "--outdir",
            str(out),
            "--no-balance",
            "--min-mapq",
            "0",
            "--max-zoom",
            "0",
            "--no-log-heatmap",
            "--n-null",
            "2",
            "--windows",
            "4,8",
            "--quiet",
        ]  # fmt: skip
    )
    assert rc == 0
    info = json.loads((out / "run_info.json").read_text())
    assert info["chromosomes"] == ["chr1", "chr2"]
    assert info["stats"]["total"]["low_mapq"] == 0
    assert info["params"]["windows"] == [4, 8]
    for chrom in ("chr1", "chr2"):
        assert _is_png(out / f"hic_{chrom}_screening.png")
        assert _is_png(out / f"hic_{chrom}_scores.png")
    assert (out / "anomaly_calls.csv").exists()

    out_all = tmp_path / "all"
    assert cli.main(["--input", bam2, "--chrom", "all", "--screen-bin", str(BIN),
                     "--outdir", str(out_all), "--max-zoom", "0", "-q"]) == 0  # fmt: skip
    assert json.loads((out_all / "run_info.json").read_text())["chromosomes"] == ["chr1", "chr2"]


def test_v2_too_few_bins_gives_empty_calls(bam, tmp_path, capsys):
    out = tmp_path / "coarse"
    rc = cli.main(["--input", bam, "--chrom", "chr1", "--outdir", str(out)])  # 1 Mb bins
    assert rc == 0
    assert "needs at least" in capsys.readouterr().err
    assert len(pd.read_csv(out / "anomaly_calls.csv")) == 0


def test_v2_bin_limit_error(bam, tmp_path, capsys):
    out = tmp_path / "fine"
    with pytest.raises(SystemExit) as exc:
        cli.main(["--input", bam, "--chrom", "chr1", "--screen-bin", "500", "--outdir", str(out)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "4,000 bins" in err and "larger --screen-bin" in err and "--screen-bin 1000" in err
    assert not (out / "anomaly_calls.csv").exists()


def test_unknown_chrom_exits_two(bam, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--input", bam, "--chrom", "chrZ", "--outdir", str(tmp_path / "a")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "'chrZ' not found in BAM header" in err
    assert "Traceback" not in err


def test_bad_input_file_exits_one(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--input", "reads.txt", "--chrom", "chr1", "--outdir", str(tmp_path / "b")])
    assert exc.value.code == 1
    assert "Unsupported file format" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["--screen-bin", "0"],
        ["--screen-bin", "-1000"],
        ["--zoom-bin", "0"],
        ["--threads", "0"],
        ["--workers", "0"],
        ["--bam-threads", "0"],
        ["--min-mapq", "-1"],
        ["--max-zoom", "-1"],
        ["--seed", "-1"],
        ["--n-null", "-2"],
        ["--ignore-diags", "-1"],
        ["--fdr", "0"],
        ["--fdr", "1.5"],
        ["--fdr", "nan"],
        ["--contamination", "0"],
        ["--contamination", "0.6"],
        ["--screen-bin", "abc"],
    ],
)
def test_invalid_numeric_args_exit_two(bam, tmp_path, capsys, extra):
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(bam, tmp_path / "bad", *extra))
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "bad").exists()


def test_n_null_and_seed_help_is_accurate(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Accepted for backward compatibility; unused by the current detector" in text
    assert "permutation null" not in text.split("--n-null")[1].split("--windows")[0]
    assert "Random seed. Legacy: Isolation Forest. v2:" in text


def test_heatmap_label_count_is_calls_not_anchors(bam, tmp_path, monkeypatch):
    from hic_anomaly import viz

    real_heatmap = viz.plot_heatmap
    seen = {}

    def spy_heatmap(*a, **kw):
        seen.update(kw)
        return real_heatmap(*a, **kw)

    def fake_detect(logoe, bin_size, *a, return_scores=False, **kw):
        rows = [
            {"bin_i": 10, "bin_j": 30, "start_bp_i": 10 * bin_size, "start_bp_j": 30 * bin_size,
             "type": "inversion", "score": 5.0, "p_value": 1e-4, "q_value": 1e-3},
            {"bin_i": 50, "bin_j": 60, "start_bp_i": 50 * bin_size, "start_bp_j": 60 * bin_size,
             "type": "inversion", "score": 5.0, "p_value": 1e-4, "q_value": 1e-3},
        ]  # fmt: skip
        calls = pd.DataFrame(rows, columns=cli.DETECT_FIELDS)
        n = logoe.shape[0]
        scores = pd.DataFrame(
            {"bin": np.arange(n), "score": np.linspace(0, 5, n), "threshold": 4.0,
             "called": np.arange(n) == 10}
        )  # fmt: skip
        return (calls, scores) if return_scores else calls

    monkeypatch.setattr(cli, "detect_anomalies", fake_detect)
    monkeypatch.setattr(viz, "plot_heatmap", spy_heatmap)
    out = tmp_path / "label"
    assert cli.main(_args(bam, out, "--workers", "1", "--max-zoom", "0", "-q")) == 0
    calls = pd.read_csv(out / "anomaly_calls.csv")
    assert len(calls) == 2
    if cli._HEATMAP_TAKES_N_CALLS:
        assert seen["n_calls"] == len(calls)


def test_missing_chrom_message_on_stderr_not_traceback(bam, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--input", bam, "--chrom", "nope", "--outdir", str(tmp_path / "m")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "'nope' not found in BAM header" in err
    assert "Traceback" not in err


def test_small_matrix_warning(bam, tmp_path, capsys):
    # 2 Mb at 20 kb = 100 bins: below the 150-bin threshold.
    assert cli.main(_args(bam, tmp_path / "small", "--workers", "1", "--max-zoom", "0")) == 0
    err = capsys.readouterr().err
    assert "coarse/small matrix: detection sensitivity is limited" in err
    assert "use a smaller --screen-bin" in err

    # 2 Mb at 10 kb = 200 bins: no warning.
    rc = cli.main(
        ["--input", bam, "--chrom", "chr1", "--screen-bin", "10000",
         "--outdir", str(tmp_path / "ok2"), "--workers", "1", "--max-zoom", "0"]
    )  # fmt: skip
    assert rc == 0
    assert "coarse/small matrix" not in capsys.readouterr().err


def test_module_entry_point_bad_args_exit_two():
    res = subprocess.run(
        [sys.executable, "-m", "hic_anomaly", "--chrom", "chr1", "--input", "x.bam",
         "--screen-bin", "0"],
        cwd=REPO, capture_output=True, text=True,
    )  # fmt: skip
    assert res.returncode == 2
    assert "error:" in res.stderr


def test_legacy_end_to_end_matches_original(bam, tmp_path, capsys):
    out = tmp_path / "legacy"
    rc = cli.main(_args(bam, out, "--method", "legacy", "--contamination", "0.05"))
    assert rc == 0

    # Console output: the original's lines, nothing added.
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"[✓] Target: chr1 ({CHROM_LEN:,} bp)"
    assert lines[1] == f"--> Building whole-chromosome matrix at {BIN // 1000}kb..."
    assert lines[2].startswith("    Total reads: ")
    assert lines[3] == "--> Running Isolation Forest (contamination=0.05)..."
    assert lines[4].startswith("    Flagged candidate regions: ")
    assert lines[-1] == f"[✓] Pipeline finished successfully. Results saved in: {out}"
    assert len(lines) == 7

    # Calls: recompute with the original algorithm and compare field by field.
    matrix = _original_matrix(bam, "chr1", CHROM_LEN, BIN)
    flags, scores = run_ai(matrix, contamination=0.05)
    expected = merge_adjacent_bins(np.where(flags)[0], scores, BIN, "chr1", CHROM_LEN)
    assert expected, "synthetic data should yield legacy candidates"
    with open(out / "anomaly_candidates_chr1.tsv", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        assert reader.fieldnames == cli.LEGACY_FIELDS
        rows = list(reader)
    assert rows == [{k: str(v) for k, v in r.items()} for r in expected]

    assert _is_png(out / "hic_chr1_screening.png")
    zooms = sorted(os.listdir(out / "zoomed_regions"))
    want = [
        f"zoom_{i:02d}_{int(r['start_Mb'])}Mb_{ZOOM // 1000}kb.png"
        for i, r in enumerate(expected[:5], 1)
    ]
    assert zooms == sorted(want)
    assert all(_is_png(out / "zoomed_regions" / z) for z in zooms)
    assert not (out / "anomaly_calls.csv").exists()

    info = json.loads((out / "run_info.json").read_text())
    assert info["method"] == "legacy" and info["n_calls"] == len(expected)
    assert info["stats"]["per_chromosome"]["chr1"]["used"] == int(np.triu(matrix).sum())


def test_legacy_requests_legacy_style_explicitly(bam, tmp_path, monkeypatch):
    from hic_anomaly import viz

    real_matrix, real_zoom = viz.plot_matrix, viz.zoom_region_from_bam
    styles = {"matrix": [], "zoom": []}

    def spy_matrix(*a, **kw):
        styles["matrix"].append(kw.get("style"))
        return real_matrix(*a, **kw)

    def spy_zoom(*a, **kw):
        styles["zoom"].append(kw.get("style"))
        return real_zoom(*a, **kw)

    monkeypatch.setattr(viz, "plot_matrix", spy_matrix)
    monkeypatch.setattr(viz, "zoom_region_from_bam", spy_zoom)
    rc = cli.main(_args(bam, tmp_path / "o", "--method", "legacy", "--contamination", "0.05", "-q"))
    assert rc == 0
    # First matrix call is the screening heatmap. Later ones come from inside the
    # legacy zoom branch, which calls plot_matrix with its original (no style) call.
    assert styles["matrix"][0] == "legacy" and set(styles["matrix"][1:]) <= {None}
    assert styles["zoom"] and set(styles["zoom"]) == {"legacy"}
    assert viz.plot_matrix is spy_matrix  # cli no longer swaps viz functions itself
    assert not hasattr(cli, "_legacy_plot_matrix") and not hasattr(cli, "_legacy_plot_style")


def test_v2_zoom_is_modern_with_mapq(bam, tmp_path, monkeypatch):
    from hic_anomaly import viz

    real_zoom = viz.zoom_region_from_bam
    seen = []

    def spy_zoom(*a, **kw):
        seen.append((kw.get("style"), kw.get("min_mapq")))
        return real_zoom(*a, **kw)

    def fake_detect(logoe, bin_size, *a, return_scores=False, **kw):
        # One planted pair call, so the zoom branch always runs regardless of noise.
        calls = pd.DataFrame(
            [
                {
                    "bin_i": 10, "bin_j": 30, "start_bp_i": 10 * bin_size,
                    "start_bp_j": 30 * bin_size, "type": "inversion", "score": 5.0,
                    "p_value": 1e-4, "q_value": 1e-3,
                }
            ],
            columns=cli.DETECT_FIELDS,
        )  # fmt: skip
        n = logoe.shape[0]
        scores = pd.DataFrame(
            {"bin": np.arange(n), "score": np.linspace(0, 5, n), "threshold": 4.0,
             "called": np.arange(n) == 10}
        )  # fmt: skip
        return (calls, scores) if return_scores else calls

    monkeypatch.setattr(cli, "detect_anomalies", fake_detect)
    monkeypatch.setattr(viz, "zoom_region_from_bam", spy_zoom)
    out = tmp_path / "v2zoom"
    rc = cli.main(_args(bam, out, "--min-mapq", "20", "--workers", "1", "-q"))
    assert rc == 0
    assert seen and set(seen) == {("modern", 20)}
    assert len(seen) == 1  # the single planted call gets one zoom


def test_strict_pairs_and_bam_threads_forwarded(bam, tmp_path, monkeypatch):
    real_build = cli.build_matrices_parallel
    real_oe = cli.obs_over_exp
    calls = {"build": None, "oe": []}

    def spy_build(*a, **kw):
        calls["build"] = kw
        return real_build(*a, **kw)

    def spy_oe(*a, **kw):
        calls["oe"].append(kw.get("dense_output"))
        return real_oe(*a, **kw)

    monkeypatch.setattr(cli, "build_matrices_parallel", spy_build)
    monkeypatch.setattr(cli, "obs_over_exp", spy_oe)
    cache = tmp_path / "strict-cache"
    out = tmp_path / "strict"
    rc = cli.main(
        _args(bam, out, "--strict-pairs", "--bam-threads", "2", "--workers", "1",
              "--cache-dir", str(cache), "-q")
    )  # fmt: skip
    assert rc == 0
    assert calls["build"]["strict_pairs"] is True and calls["build"]["bam_threads"] == 2
    assert calls["oe"] == [True]
    # Strict matrices are cached under their own name; default names are unchanged.
    assert len(list(cache.glob("*.chr1.bin20000.mapq30.strict.npz"))) == 1
    assert not list(cache.glob("*.chr1.bin20000.mapq30.npz"))

    info = json.loads((out / "run_info.json").read_text())
    assert info["params"]["strict_pairs"] is True and info["params"]["bam_threads"] == 2


def test_cache_path_default_name_unchanged(tmp_path, bam):
    default = cli._cache_path(str(tmp_path), bam, "chr1", BIN, 30)
    assert default.endswith(".chr1.bin20000.mapq30.npz")
    strict = cli._cache_path(str(tmp_path), bam, "chr1", BIN, 30, strict_pairs=True)
    assert strict.endswith(".chr1.bin20000.mapq30.strict.npz")


def test_zoom_calls_for_v2(bam, tmp_path):
    calls = pd.DataFrame(
        {
            "chrom": "chr1",
            "start": [400_000, 0, 900_000],
            "end": [520_000, CHROM_LEN, 920_000],
            "type": ["cnv", "inversion", "deletion/dup"],
        }
    )
    figs = cli._zoom_calls(
        bam, "chr1", CHROM_LEN, calls, ZOOM, str(tmp_path), max_zoom=2, min_mapq=30
    )
    assert [os.path.basename(p) for _, p in figs] == [
        "zoom_chr1_01_0Mb_5kb.png",
        "zoom_chr1_02_0Mb_5kb.png",
    ]
    assert all(_is_png(p) for _, p in figs)
    # A span needing more than the dense limit at the zoom bin size is skipped.
    assert cli._zoom_calls(bam, "chr1", CHROM_LEN, calls.iloc[[1]], 500, str(tmp_path), 5) == []
    assert cli._zoom_calls(bam, "chr1", CHROM_LEN, calls, ZOOM, str(tmp_path), 0) == []


def test_calls_to_spans_marks_anchors_and_cnv_runs():
    df = pd.DataFrame(
        {
            "start_bp_i": [100_000, 400_000],
            "start_bp_j": [900_000, 480_000],
            "type": ["inversion", "cnv"],
        }
    )
    assert cli._calls_to_spans(df, BIN, 910_000) == [
        {"start_bp": 100_000, "end_bp": 120_000},
        {"start_bp": 900_000, "end_bp": 910_000},
        {"start_bp": 400_000, "end_bp": 500_000},
    ]


def test_suggest_bin_keeps_within_limit():
    for length in (2_000_000, 50_000_000, 248_956_422, 5_000_000_000):
        step = cli._suggest_bin(length)
        assert np.ceil(length / step) <= cli.MAX_DETECT_BINS


@pytest.mark.parametrize(
    "cmd",
    [["-m", "hic_anomaly"], ["universal_hic_pipeline.py"], ["cross_refrence_hits.py"]],
)
def test_entry_points_help(cmd):
    res = subprocess.run([sys.executable, *cmd, "--help"], cwd=REPO, capture_output=True, text=True)
    assert res.returncode == 0
    assert "usage:" in res.stdout
