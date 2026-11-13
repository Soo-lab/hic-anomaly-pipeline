import numpy as np
import pandas as pd

from hic_anomaly import viz
from hic_anomaly.report import write_html_report

CALLS = [{"start_bp": 15000, "end_bp": 20000}]


def _png(p):
    return p.exists() and p.read_bytes()[:4] == b"\x89PNG"


def test_heatmap_square_and_tri(tmp_path, toy_matrix):
    before = toy_matrix.copy()
    for tri in (False, True):
        out = tmp_path / f"h{tri}.png"
        viz.plot_heatmap(toy_matrix, "chr1", 1000, str(out), calls=CALLS, triangular=tri)
        assert _png(out)
    assert (toy_matrix == before).all()


def test_downsample_display_only():
    m = np.ones((3000, 3000), dtype=np.int32)
    d, f = viz._downsample(m)
    assert f == 2 and d.shape == (1500, 1500) and m.shape == (3000, 3000)


def test_legacy_plot_matrix(tmp_path, toy_matrix):
    out = tmp_path / "m.png"
    viz.plot_matrix(toy_matrix, "chr1", 1000, str(out))
    assert _png(out)


def test_tracks_pr_report(tmp_path):
    s = np.random.default_rng(0).random(50)
    t = tmp_path / "t.png"
    viz.plot_score_track(s, 0.9, 1000, str(t), calls=CALLS)
    c = tmp_path / "c.png"
    viz.plot_comparison(CALLS, [{"start_bp": 16000, "end_bp": 21000}], out=str(c))
    p = tmp_path / "p.png"
    viz.plot_pr_curves({"a": {"precision": [1, 0.8], "recall": [0, 1], "ap": 0.9}}, str(p))
    assert _png(t) and _png(c) and _png(p)
    df = pd.DataFrame({"chrom": ["chr1"], "start_Mb": [0.015], "score": [0.5]})
    r = write_html_report(str(tmp_path), {"bin_size": 1000}, [str(t), ("PR", str(p))], df)
    txt = open(r).read()
    assert "data:image/png;base64," in txt and "http" not in txt.replace("https://", "")
    assert "chr1" in txt


def test_style_validation(tmp_path, toy_matrix):
    import pytest

    with pytest.raises(ValueError):
        viz.plot_matrix(toy_matrix, "chr1", 1000, str(tmp_path / "x.png"), style="fancy")
    with pytest.raises(TypeError):
        viz.plot_matrix(toy_matrix, "chr1", 1000, str(tmp_path / "x.png"), style="legacy", vmin=0)


def test_score_track_none_threshold_and_per_bin(tmp_path):
    s = np.linspace(0, 1, 30)
    per_bin = np.full(30, 0.5)
    for thr in (None, np.nan, 0.5, per_bin, np.full(30, np.nan)):
        out = tmp_path / f"t_{hash(str(thr))}.png"
        viz.plot_score_track(s, thr, 1000, str(out))
        assert _png(out)
    viz.plot_score_track(None, None, 1000, str(tmp_path / "none.png"))
    assert _png(tmp_path / "none.png")


def test_modern_zoom_min_mapq(tmp_path):
    from tests.synth import make_bam

    bam = make_bam(str(tmp_path / "z.bam"), n_pairs=800, chrom_len=200_000)
    m0, used0 = viz.zoom_matrix_from_bam(bam, "chr1", 0, 200_000, 5000, min_mapq=0)
    m30, used30 = viz.zoom_matrix_from_bam(bam, "chr1", 0, 200_000, 5000, min_mapq=30)
    assert used30 < used0 and m30.sum() < m0.sum()
    assert (m0 == m0.T).all()
    png = viz.zoom_region_from_bam(
        bam,
        "chr1",
        200_000,
        50_000,
        80_000,
        5000,
        10_000,
        str(tmp_path),
        "r",
        style="modern",
        min_mapq=30,
    )
    assert _png(tmp_path / "zoom_r_5kb.png") and png.endswith(".png")


def test_legacy_zoom_rejects_mapq(tmp_path):
    import pytest

    from tests.synth import make_bam

    bam = make_bam(str(tmp_path / "z.bam"), n_pairs=50, chrom_len=200_000)
    with pytest.raises(ValueError):
        viz.zoom_region_from_bam(
            bam, "chr1", 200_000, 0, 50_000, 5000, 0, str(tmp_path), "r", min_mapq=20
        )


def test_log_ticks_never_lone():
    assert len(viz._log_ticks(1.0, 4.2)) >= 2
    assert viz._log_ticks(1.0, 417.0) == [1.0, 10.0, 100.0]


def test_n_calls_label_and_cmap_options(tmp_path, toy_matrix):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    viz._render_heatmap(fig, ax, toy_matrix, "chr1", 1000, calls=CALLS * 3, n_calls=7)
    assert any("n = 7" in t.get_text() for a in fig.axes for t in a.texts)
    plt.close(fig)
    for cm in ("cividis", "viridis", "hic_fall"):
        out = tmp_path / f"{cm}.png"
        viz.plot_heatmap(toy_matrix, "chr1", 1000, str(out), cmap=cm)
        assert _png(out)


def test_zoom_title_has_region_and_bin(tmp_path):
    from tests.synth import make_bam

    bam = make_bam(str(tmp_path / "z.bam"), n_pairs=300, chrom_len=200_000)
    viz.zoom_region_from_bam(
        bam, "chr1", 200_000, 50_000, 80_000, 5000, 10_000, str(tmp_path), "t", style="modern"
    )
    assert _png(tmp_path / "zoom_t_5kb.png")
