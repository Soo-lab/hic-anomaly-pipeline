"""Deterministic synthetic Hi-C BAM generator (pysam)."""

import numpy as np
import pysam


def make_bam(path, n_pairs=2000, chrom_len=200_000, seed=0, chroms=("chr1",)):
    rng = np.random.default_rng(seed)
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": c, "LN": chrom_len} for c in chroms],
    }
    recs = []
    L = 50

    def mk(h, name, tid, pos, ntid, npos, flag, mapq, rev):
        a = pysam.AlignedSegment(h)
        a.query_name, a.query_sequence = name, "A" * L
        a.flag = flag | (16 if rev else 0)
        a.reference_id, a.reference_start = tid, pos
        a.mapping_quality, a.cigarstring = mapq, f"{L}M"
        a.next_reference_id, a.next_reference_start = ntid, npos
        a.query_qualities = pysam.qualitystring_to_array("I" * L)
        return a

    h = pysam.AlignmentHeader.from_dict(header)
    for i in range(n_pairs):
        t = int(rng.integers(len(chroms)))
        p1 = int(rng.integers(0, chrom_len - L))
        d = int(rng.exponential(chrom_len / 20))
        p2 = min(p1 + d, chrom_len - L - 1)
        mq = int(rng.choice([0, 10, 40, 60], p=[0.05, 0.05, 0.2, 0.7]))
        dup = 1024 if rng.random() < 0.03 else 0
        nm = f"p{i}"
        recs.append(mk(h, nm, t, p1, t, p2, 1 | 64 | dup | (32 if True else 0), mq, False))
        recs.append(mk(h, nm, t, p2, t, p1, 1 | 128 | dup, mq, True))
    recs.sort(key=lambda a: (a.reference_id, a.reference_start))
    with pysam.AlignmentFile(path, "wb", header=h) as f:
        for a in recs:
            f.write(a)
    pysam.index(str(path))
    return str(path)
