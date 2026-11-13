"""End-to-end CLI wall time and peak RSS, v2 vs legacy, on planted synthetic BAMs.

Usage: python benchmarks/bench_validate_cli.py [--sizes 50000 200000 600000] [--workers-test]
Peak RSS comes from RUSAGE_CHILDREN of a fresh child per measurement (macOS bytes, Linux KiB).
"""

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.synth_planted import make_planted_bam  # noqa: E402

CHILD = (
    "import resource,subprocess,sys,time;t=time.time();"
    "rc=subprocess.call(sys.argv[1:],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
    "r=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;"
    "print(rc,time.time()-t,r)"
)


def measure(cmd):
    out = subprocess.check_output([sys.executable, "-c", CHILD, *cmd], text=True).split()
    rc, wall, rss = int(out[0]), float(out[1]), float(out[2])
    rss_mb = rss / 1048576 if sys.platform == "darwin" else rss / 1024
    return rc, wall, rss_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[50_000, 200_000, 600_000])
    ap.add_argument("--workers-test", action="store_true")
    a = ap.parse_args()
    env_cmd = [sys.executable, "-m", "hic_anomaly"]
    with tempfile.TemporaryDirectory() as td:
        print("| pairs | method | wall s | peak RSS MB | rc |\n|---|---|---|---|---|")
        for n in a.sizes:
            bam = make_planted_bam(os.path.join(td, f"p{n}.bam"), n)
            for m in ("v2", "legacy"):
                rc, w, r = measure(
                    [*env_cmd, "--input", bam, "--chrom", "chr1", "--method", m,
                     "--screen-bin", "250000", "--zoom-bin", "50000",
                     "--outdir", os.path.join(td, f"o{m}{n}"), "-q"]
                )  # fmt: skip
                print(f"| {n} | {m} | {w:.2f} | {r:.0f} | {rc} |")
        if a.workers_test:
            bam = make_planted_bam(
                os.path.join(td, "m.bam"), 400_000, chroms=("chr1", "chr2", "chr3", "chr4")
            )
            print("\n| workers | wall s | peak RSS MB |\n|---|---|---|")
            for wk in (1, 4):
                rc, w, r = measure(
                    [*env_cmd, "--input", bam, "--chrom", "all", "--workers", str(wk),
                     "--screen-bin", "250000", "--max-zoom", "0",
                     "--outdir", os.path.join(td, f"w{wk}"), "-q"]
                )  # fmt: skip
                print(f"| {wk} | {w:.2f} | {r:.0f} |  rc={rc}")


if __name__ == "__main__":
    main()
