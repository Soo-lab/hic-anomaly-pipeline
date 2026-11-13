"""Larger deterministic synthetic BAM with planted structural events (validation helper)."""

import numpy as np
import pysam


def make_planted_bam(
    path, n_pairs=100_000, chrom_len=20_000_000, seed=1, chroms=("chr1",), with_mq=True
):
    """Distance-decay background plus an inversion-like off-diagonal blob and a CNV gain.

    Planted events (chroms[0]): blob joining 5-6 Mb with 12-13 Mb; coverage x3 at 15-16 Mb.
    """
    rng = np.random.default_rng(seed)
    L = 50
    h = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": c, "LN": chrom_len} for c in chroms],
        }
    )
    n_blob, n_cnv = n_pairs // 20, n_pairs // 10
    t = rng.integers(len(chroms), size=n_pairs)
    p1 = rng.integers(0, chrom_len - L - 1, n_pairs)
    d = rng.exponential(chrom_len / 30, n_pairs).astype(np.int64)
    p2 = np.minimum(p1 + d, chrom_len - L - 1)
    t[:n_blob] = 0
    p1[:n_blob] = rng.integers(5_000_000, 6_000_000, n_blob)
    p2[:n_blob] = rng.integers(12_000_000, 13_000_000, n_blob)
    s = slice(n_blob, n_blob + n_cnv)
    t[s] = 0
    p1[s] = rng.integers(15_000_000, 16_000_000, n_cnv)
    p2[s] = np.minimum(
        p1[s] + rng.exponential(chrom_len / 30, n_cnv).astype(np.int64), chrom_len - L - 1
    )
    mq = rng.choice([0, 10, 40, 60], p=[0.03, 0.04, 0.13, 0.8], size=n_pairs)
    recs = []
    for i in range(n_pairs):
        lo, hi = sorted((int(p1[i]), int(p2[i])))
        for k, (pos, mate) in enumerate(((lo, hi), (hi, lo))):
            a = pysam.AlignedSegment(h)
            a.query_name, a.query_sequence = f"p{i}", "A" * L
            a.flag = 1 | (64 if k == 0 else 128) | (16 if k else 0)
            a.reference_id, a.reference_start = int(t[i]), pos
            a.next_reference_id, a.next_reference_start = int(t[i]), mate
            a.mapping_quality, a.cigarstring = int(mq[i]), f"{L}M"
            a.query_qualities = pysam.qualitystring_to_array("I" * L)
            if with_mq:
                a.set_tag("MQ", int(mq[i]))
            recs.append(a)
    recs.sort(key=lambda a: (a.reference_id, a.reference_start))
    with pysam.AlignmentFile(path, "wb", header=h) as f:
        for a in recs:
            f.write(a)
    pysam.index(str(path))
    return str(path)
