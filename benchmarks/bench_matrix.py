"""Baseline (dict-pairing, dense) vs new sparse matrix build.

Usage: python benchmarks/bench_matrix.py [out_markdown]
"""

import argparse
import sys
import tempfile
import time
import tracemalloc
from datetime import date
from pathlib import Path

import numpy as np
import pysam

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hic_anomaly.matrix import build_contact_matrix_sparse  # noqa: E402
from tests.synth import make_bam  # noqa: E402


def baseline(bam_path, chrom, chrom_len, bin_size):
    """Logic of universal_hic_pipeline.create_contact_matrix at HEAD."""
    nb = int(np.ceil(chrom_len / bin_size))
    m = np.zeros((nb, nb), dtype=np.int32)
    pend, tot, used = {}, 0, 0
    bam = pysam.AlignmentFile(bam_path, "rb")
    for r in bam.fetch(chrom):
        tot += 1
        if (
            r.is_unmapped
            or r.mate_is_unmapped
            or r.is_secondary
            or r.is_supplementary
            or r.is_duplicate
        ):
            continue
        if r.next_reference_id != r.reference_id:
            continue
        if r.query_name not in pend:
            pend[r.query_name] = r
            continue
        r1 = pend.pop(r.query_name)
        b1, b2 = r1.reference_start // bin_size, r.reference_start // bin_size
        m[b1, b2] += 1
        if b1 != b2:
            m[b2, b1] += 1
        used += 1
    bam.close()
    return m, tot, used


def measure(fn):
    tracemalloc.start()
    t = time.perf_counter()
    fn()
    dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return dt, peak


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
    ap = argparse.ArgumentParser(description="Baseline vs sparse matrix build benchmark.")
    ap.add_argument("out_markdown", nargs="?", default=None, help="append markdown here")
    out = ap.parse_args().out_markdown
    cases = [
        (20_000, 5_000_000, 10_000),
        (100_000, 50_000_000, 10_000),
        (300_000, 100_000_000, 5_000),
    ]
    rows = [
        "| pairs | chrom len | bin | baseline s | baseline peak MB | new s | new peak MB |",
        "|---|---|---|---|---|---|---|",
    ]
    with tempfile.TemporaryDirectory() as d:
        for n, L, b in cases:
            bam = make_bam(Path(d) / f"s{n}.bam", n, L, seed=0)
            tb, mb = measure(lambda: baseline(bam, "chr1", L, b))
            tn, mn = measure(lambda: build_contact_matrix_sparse(bam, "chr1", L, b, min_mapq=0))
            rows.append(f"| {n} | {L} | {b} | {tb:.2f} | {mb:.1f} | {tn:.2f} | {mn:.1f} |")
            print(rows[-1], flush=True)
    text = f"\n### bench_matrix {date.today()}\n\n{platform_info()}\n\n" + "\n".join(rows) + "\n"
    if out:
        with open(out, "a") as f:
            f.write(text)


if __name__ == "__main__":
    main()
