"""Every CLI option in --help is exercised by an actual run (not just listed)."""

import json

import pytest

from hic_anomaly import cli
from tests.synth import make_bam

BIN = "20000"


@pytest.fixture(scope="module")
def bam(tmp_path_factory):
    return make_bam(
        tmp_path_factory.mktemp("fl") / "s.bam", n_pairs=8000, chrom_len=2_000_000, seed=3
    )


def _base(bam, out):
    return ["--input", bam, "--chrom", "chr1", "--screen-bin", BIN, "--outdir", str(out),
            "--max-zoom", "0", "-q"]  # fmt: skip


# option string -> extra argv that uses it
V2_FLAGS = {
    "--threads": ["--threads", "2"],
    "--method": ["--method", "v2"],
    "--zoom-bin": ["--zoom-bin", "5000"],
    "--seed": ["--seed", "7"],
    "--min-mapq": ["--min-mapq", "10"],
    "--workers": ["--workers", "1"],
    "--strict-pairs": ["--strict-pairs"],
    "--bam-threads": ["--bam-threads", "2"],
    "--cache-dir": ["--cache-dir", "CACHE"],
    "--balance": ["--balance"],
    "--no-balance": ["--no-balance"],
    "--ignore-diags": ["--ignore-diags", "3"],
    "--oe": ["--oe"],
    "--no-oe": ["--no-oe"],
    "--fdr": ["--fdr", "0.1"],
    "--n-null": ["--n-null", "2"],
    "--windows": ["--windows", "3,6"],
    "--triangular": ["--triangular"],
    "--no-log-heatmap": ["--no-log-heatmap"],
    "--report": ["--report"],
    "-v": ["-v"],
}


@pytest.mark.parametrize("flag", sorted(V2_FLAGS))
def test_v2_flag_runs(flag, bam, tmp_path):
    extra = [str(tmp_path / "cache") if a == "CACHE" else a for a in V2_FLAGS[flag]]
    argv = _base(bam, tmp_path / "o")
    if flag == "-v":
        argv.remove("-q")
    assert cli.main([*argv, *extra]) == 0
    info = json.loads((tmp_path / "o" / "run_info.json").read_text())
    assert info["method"] == "v2"
    if flag == "--report":
        assert (tmp_path / "o" / "report.html").exists()
    if flag == "--seed":
        assert info["seed"] == 7
    if flag == "--cache-dir":
        assert list((tmp_path / "cache").glob("*.npz"))


def test_legacy_flags(bam, tmp_path):
    argv = ["--input", bam, "--chrom", "chr1", "--method", "legacy", "--screen-bin", BIN,
            "--zoom-bin", "5000", "--max-zoom", "1", "--contamination", "0.05", "--seed", "1",
            "--threads", "2", "--outdir", str(tmp_path), "-q"]  # fmt: skip
    assert cli.main(argv) == 0
    assert json.loads((tmp_path / "run_info.json").read_text())["method"] == "legacy"


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and "hic-anomaly" in capsys.readouterr().out


def test_fastq_without_input2_or_ref_errors_cleanly(tmp_path, capsys):
    fq = tmp_path / "r1.fastq"
    fq.write_text("@r\nACGT\n+\nIIII\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["--input", str(fq), "--chrom", "chr1", "--outdir", str(tmp_path / "o"),
                  "--input2", str(fq), "--ref-fasta", str(tmp_path / "missing.fa")])  # fmt: skip
    assert e.value.code not in (0, None)
    assert "Traceback" not in capsys.readouterr().err


def test_unsupported_format_errors_cleanly(tmp_path, capsys):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    with pytest.raises(SystemExit) as e:
        cli.main(["--input", str(p), "--chrom", "chr1", "--outdir", str(tmp_path / "o")])
    assert e.value.code == 1
    assert "Unsupported file format" in capsys.readouterr().err


def test_every_parser_option_is_covered():
    covered = set(V2_FLAGS) | {
        "--input", "--input2", "--ref-fasta", "--chrom", "--outdir", "--max-zoom",
        "--screen-bin", "--contamination", "--version", "--help", "-h", "-q", "--quiet",
        "--verbose",
    }  # fmt: skip
    opts = {s for a in cli.build_parser()._actions for s in a.option_strings}
    assert opts <= covered, sorted(opts - covered)
