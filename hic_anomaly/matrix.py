import json
import logging
import os
from array import array
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysam
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Chromosome & Reference Resolution


def resolve_name(name, candidates):
    candidates = set(candidates)
    if name in candidates:
        return name
    alt = name[3:] if name.startswith("chr") else "chr" + name
    return alt if alt in candidates else None


def resolve_chrom_and_length(bam_path, chrom):
    bam = pysam.AlignmentFile(bam_path, "rb")
    bam_chrom = resolve_name(chrom, bam.references)
    if bam_chrom is None:
        avail = list(bam.references)[:10]
        bam.close()
        raise ValueError(f"'{chrom}' not found in BAM header. Available: {avail}...")
    chrom_len = bam.get_reference_length(bam_chrom)
    bam.close()
    return bam_chrom, chrom_len


# Contact Matrix Construction

_CHUNK = 1_000_000  # pairs buffered before they are reduced to (bin key, count)

# SAM flag bits
_F_PAIRED = 0x1
_F_UNMAP = 0x4
_F_MUNMAP = 0x8
_F_READ1 = 0x40
_F_SECONDARY = 0x100
_F_DUP = 0x400
_F_SUPP = 0x800
_F_REJECT = _F_UNMAP | _F_MUNMAP | _F_SECONDARY | _F_SUPP | _F_DUP

_STAT_KEYS = (
    "total",
    "unmapped",
    "secondary_supp",
    "duplicate",
    "low_mapq",
    "trans",
    "unpaired",
    "out_of_range",
    "mate_second",
    "orphan",
    "used",
)
_DROP_KEYS = (
    "unmapped",
    "secondary_supp",
    "duplicate",
    "low_mapq",
    "trans",
    "unpaired",
    "orphan",
    "out_of_range",
)


def _reduce_keys(keys):
    u, c = np.unique(keys, return_counts=True)
    return u, c.astype(np.int64)


def _merge_parts(parts):
    """Merge [(sorted unique keys, counts), ...] into one (keys, counts)."""
    if not parts:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    if len(parts) == 1:
        return parts[0]
    keys = np.concatenate([p[0] for p in parts])
    cnt = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=cnt).astype(np.int64)


def _flush_pairs(pos, nb, bin_size, parts):
    """Bin a flat [p1, p2, p1, p2, ...] buffer; return the number out of range."""
    if not len(pos):
        return 0
    xy = np.frombuffer(pos, dtype=np.int64).reshape(-1, 2)
    b1 = xy[:, 0] // bin_size
    b2 = xy[:, 1] // bin_size
    ok = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (b1 < nb) & (b2 < nb)
    lo = np.minimum(b1, b2)[ok]
    hi = np.maximum(b1, b2)[ok]
    if len(lo):
        parts.append(_reduce_keys(lo * nb + hi))
    return int(len(ok) - ok.sum())


def _flush_trans(buf, nb, bin_size, n_refs, parts):
    """Reduce a flat [pos, mate_tid, ...] buffer to (bin * n_refs + mate_tid, count)."""
    if not len(buf):
        return
    xy = np.frombuffer(buf, dtype=np.int64).reshape(-1, 2)
    b = xy[:, 0] // bin_size
    ok = (xy[:, 0] >= 0) & (b < nb) & (xy[:, 1] >= 0)
    if ok.any():
        parts.append(_reduce_keys(b[ok] * n_refs + xy[ok, 1]))


def build_contact_matrix_sparse(
    bam_path,
    chrom,
    chrom_len,
    bin_size,
    min_mapq=30,
    keep_trans=False,
    bam_threads=1,
    strict_pairs=False,
    log_level=logging.INFO,
):
    """Stream a coordinate-sorted, indexed BAM into a symmetric CSR int32 matrix.

    Returns ``(matrix, stats)``, or ``(matrix, stats, trans)`` when
    ``keep_trans=True``. The matrix is never densified; memory is
    O(chunk + nnz) (plus the pending-mate table in strict mode).

    Read filters, applied to every record in this order (first match wins, and
    the reason is tallied in ``stats``): unmapped or mate unmapped,
    secondary/supplementary, duplicate, unpaired, MAPQ < ``min_mapq``, trans.

    Pairing modes
    -------------
    ``strict_pairs=False`` (default, fastest, no per-read-name state):
        each pair is counted once, from its leftmost mate, using the mate
        coordinates stored on that record (RNEXT/PNEXT). Only the leftmost
        record is inspected, so the mate's own flags are not seen. Exact
        divergence from name-based pairing:

        * mate duplicate flag: SAM has no "mate is duplicate" bit, so a pair
          whose leftmost mate is clean but whose rightmost mate alone is
          flagged duplicate IS counted (name pairing drops it). Pairs whose
          leftmost mate is the flagged one are dropped in both modes.
          ``samtools markdup`` and Picard flag both mates of a duplicate pair
          together, so on their output the two modes agree.
        * mate MAPQ: taken from the ``MQ`` tag (``samtools fixmate -m``, bwa
          mem + fixmate, Picard). If the tag is absent only the leftmost
          mate's MAPQ is tested; ``stats["no_mq_tag"]`` counts the pairs that
          were accepted without a mate MAPQ check, and a warning is logged.
        * a record with PNEXT that disagrees with the mate's real position
          (BAM not fixmate-clean) is binned at PNEXT.

    ``strict_pairs=True`` (exact, matches the original pipeline):
        mates are joined by read name and a pair is counted only when BOTH
        records pass every filter (duplicate, MAPQ, ...) on their own flags;
        positions come from the two records themselves. Memory is bounded by
        the number of pairs spanning the current position: a record whose mate
        lies upstream and was not kept is discarded at once rather than
        stored. Reads whose mate never passes are tallied as ``orphan``.

    Trans contacts
    --------------
    Trans pairs never enter the cis matrix; ``stats["trans"]`` always counts
    the trans reads on ``chrom`` that pass the read filters. With
    ``keep_trans=True`` a third value is returned::

        {"chroms": [reference names in BAM header order],
         "counts": CSR int32 of shape (n_bins, n_references)}

    where ``counts[b, t]`` is the number of pairs joining bin ``b`` of
    ``chrom`` to reference ``t``, and ``stats["trans_by_chrom"]`` holds the
    per-partner totals. A trans mate sits on another contig, so it is always
    judged from this record alone (own flags plus the ``MQ`` tag), in both
    pairing modes.

    ``bam_threads`` is the number of htslib decompression threads; keep it at
    1 inside process pools.
    """
    nb = int(np.ceil(chrom_len / bin_size))
    stats = dict.fromkeys(_STAT_KEYS, 0)
    parts, tparts = [], []
    pos = array("q")
    tbuf = array("q")
    app = pos.append
    tapp = tbuf.append
    flush_at = 2 * _CHUNK
    total = n_unmapped = n_secsup = n_dup = n_unpaired = n_lowq = n_trans = 0
    n_oor = n_orphan = n_no_mq = max_pending = 0
    check_mq = min_mapq > 0
    pend = {}
    pop = pend.pop

    bam = pysam.AlignmentFile(bam_path, "rb", threads=max(1, int(bam_threads or 1)))
    try:
        refs = list(bam.references)
        n_refs = max(len(refs), 1)
        for r in bam.fetch(chrom):
            total += 1
            f = r.flag
            if f & _F_REJECT:
                if f & (_F_UNMAP | _F_MUNMAP):
                    n_unmapped += 1
                elif f & (_F_SECONDARY | _F_SUPP):
                    n_secsup += 1
                else:
                    n_dup += 1
                continue
            if not strict_pairs and not f & _F_PAIRED:
                n_unpaired += 1
                continue
            if r.mapping_quality < min_mapq:
                n_lowq += 1
                continue
            mtid = r.next_reference_id
            if mtid != r.reference_id:
                if not f & _F_PAIRED:
                    n_unpaired += 1
                elif check_mq and r.has_tag("MQ") and r.get_tag("MQ") < min_mapq:
                    n_lowq += 1
                else:
                    n_trans += 1
                    if keep_trans:
                        tapp(r.reference_start)
                        tapp(mtid)
                        if len(tbuf) >= flush_at:
                            _flush_trans(tbuf, nb, bin_size, n_refs, tparts)
                            del tbuf[:]
                continue
            p1 = r.reference_start
            if strict_pairs:
                q = r.query_name
                p0 = pop(q, None)
                if p0 is None:
                    if f & _F_PAIRED and r.next_reference_start < p1:
                        # Mate is upstream and was not kept: it can never pair.
                        n_orphan += 1
                    else:
                        pend[q] = p1
                        if len(pend) > max_pending:
                            max_pending = len(pend)
                    continue
                app(p0)
                app(p1)
            else:
                p2 = r.next_reference_start
                if p2 < p1 or (p1 == p2 and not f & _F_READ1):
                    continue  # rightmost mate: its pair is counted from the other record
                if check_mq:
                    if r.has_tag("MQ"):
                        if r.get_tag("MQ") < min_mapq:
                            n_lowq += 1
                            continue
                    else:
                        n_no_mq += 1
                app(p1)
                app(p2)
            if len(pos) >= flush_at:
                n_oor += _flush_pairs(pos, nb, bin_size, parts)
                del pos[:]
    finally:
        bam.close()
    n_oor += _flush_pairs(pos, nb, bin_size, parts)
    del pos[:]
    n_orphan += len(pend)
    pend.clear()

    u, c = _merge_parts(parts)
    i, j = u // nb, u % nb
    used = int(c.sum())
    off = i != j
    rows = np.concatenate([i, j[off]])
    cols = np.concatenate([j, i[off]])
    vals = np.concatenate([c, c[off]]).astype(np.int32)
    mat = sp.coo_matrix((vals, (rows, cols)), shape=(nb, nb)).tocsr()
    mat.sum_duplicates()

    stats.update(
        total=total,
        unmapped=n_unmapped,
        secondary_supp=n_secsup,
        duplicate=n_dup,
        low_mapq=n_lowq,
        trans=n_trans,
        unpaired=n_unpaired,
        out_of_range=n_oor,
        orphan=n_orphan,
        used=used,
    )
    if strict_pairs:
        stats["max_pending"] = max_pending
    else:
        # Kept reads that were the rightmost record of a cis pair.
        stats["mate_second"] = total - (
            n_unmapped + n_secsup + n_dup + n_unpaired + n_lowq + n_trans + n_oor + used
        )
        stats["no_mq_tag"] = n_no_mq
        if n_no_mq:
            logger.warning(
                "%s: %d of %d pairs carry no MQ tag, so the mate's MAPQ was not checked; "
                "run `samtools fixmate -m` or use strict_pairs=True for exact filtering",
                chrom,
                n_no_mq,
                used + n_oor,
            )

    trans = None
    if keep_trans:
        _flush_trans(tbuf, nb, bin_size, n_refs, tparts)
        tu, tc = _merge_parts(tparts)
        tmat = sp.coo_matrix(
            (tc.astype(np.int32), (tu // n_refs, tu % n_refs)), shape=(nb, n_refs)
        ).tocsr()
        per_ref = np.asarray(tmat.sum(axis=0)).ravel()
        stats["trans_by_chrom"] = {refs[t]: int(per_ref[t]) for t in np.flatnonzero(per_ref)}
        trans = {"chroms": refs, "counts": tmat}

    logger.log(
        log_level,
        "%s: %d reads, %d pairs used; dropped %s",
        chrom,
        total,
        used,
        {k: stats[k] for k in _DROP_KEYS if stats[k]},
    )
    if keep_trans:
        return mat, stats, trans
    return mat, stats


def create_contact_matrix(bam_path, chrom, chrom_len, bin_size):
    """Dense int32 matrix with the original pipeline's exact semantics.

    Name-based pairing (both mates must be mapped, primary, non-duplicate and
    cis), no MAPQ filter. Returns (matrix, total_reads, used_pairs). The
    per-reason read summary is logged at DEBUG.
    """
    mat, st = build_contact_matrix_sparse(
        bam_path,
        chrom,
        chrom_len,
        bin_size,
        min_mapq=0,
        strict_pairs=True,
        log_level=logging.DEBUG,
    )
    return mat.toarray().astype(np.int32), st["total"], st["used"]


# Parallel build and cache


def _worker(args):
    bam_path, chrom, bin_size, min_mapq, keep_trans, bam_threads, strict_pairs = args
    name, clen = resolve_chrom_and_length(bam_path, chrom)
    return chrom, build_contact_matrix_sparse(
        bam_path,
        name,
        clen,
        bin_size,
        min_mapq=min_mapq,
        keep_trans=keep_trans,
        bam_threads=bam_threads,
        strict_pairs=strict_pairs,
    )


def build_matrices_parallel(
    bam_path,
    chroms,
    bin_size,
    workers=4,
    min_mapq=30,
    keep_trans=False,
    bam_threads=None,
    strict_pairs=False,
):
    """Return {chrom: (csr, stats)}, one process per contig.

    With ``keep_trans=True`` each value is ``(csr, stats, trans)`` (see
    ``build_contact_matrix_sparse``). ``bam_threads=None`` picks 1 htslib
    thread per worker when a process pool is used (the pool already uses the
    cores) and up to 4 when the build runs serially.
    """
    n_jobs = len(chroms)
    serial = workers <= 1 or n_jobs <= 1
    if bam_threads is None:
        bam_threads = min(4, os.cpu_count() or 1) if serial else 1
    jobs = [
        (bam_path, c, bin_size, min_mapq, keep_trans, bam_threads, strict_pairs) for c in chroms
    ]
    if serial:
        return dict(map(_worker, jobs))
    try:
        with ProcessPoolExecutor(max_workers=min(workers, n_jobs)) as ex:
            return dict(ex.map(_worker, jobs))
    except (OSError, NotImplementedError) as e:
        logger.warning("multiprocessing unavailable (%s); running serially", e)
        return dict(map(_worker, jobs))


def save_matrix_npz(path, mat, stats=None, bias=None, trans=None):
    """Cache a CSR matrix (+ stats, optional bias and trans structure) as .npz."""
    arrs = {
        "data": mat.data,
        "indices": mat.indices,
        "indptr": mat.indptr,
        "shape": np.array(mat.shape),
        "stats": np.array(json.dumps(stats or {})),
    }
    if bias is not None:
        arrs["bias"] = bias
    if trans is not None:
        t = sp.csr_matrix(trans["counts"])
        arrs.update(
            trans_data=t.data,
            trans_indices=t.indices,
            trans_indptr=t.indptr,
            trans_shape=np.array(t.shape),
            trans_chroms=np.array(json.dumps(list(trans["chroms"]))),
        )
    np.savez_compressed(path, **arrs)


def load_matrix_npz(path, with_trans=False):
    """Return (csr, stats, bias_or_None), or None if path is missing.

    ``with_trans=True`` appends a fourth item: the trans structure saved by
    ``save_matrix_npz(..., trans=...)``, or None if the file has none.
    """
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    mat = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    bias = z["bias"] if "bias" in z.files else None
    stats = json.loads(str(z["stats"]))
    if not with_trans:
        return mat, stats, bias
    trans = None
    if "trans_data" in z.files:
        counts = sp.csr_matrix(
            (z["trans_data"], z["trans_indices"], z["trans_indptr"]),
            shape=tuple(z["trans_shape"]),
        )
        trans = {"chroms": json.loads(str(z["trans_chroms"])), "counts": counts}
    return mat, stats, bias, trans


# Normalization


def ice_balance(
    sparse,
    max_iter=200,
    tol=1e-5,
    mask_low_mad=True,
    ignore_diags=2,
    mad_max=5.0,
    min_nnz=10,
    min_nnz_frac=0.25,
    mad_min_fold=3.0,
    rescale="counts",
):
    """Iterative correction (Imakaev 2012). Returns (balanced CSR, bias).

    bias = per-bin weight (NaN for masked bins); balanced = v*b_i*b_j. The
    first ``ignore_diags`` diagonals are excluded from balancing and from the
    returned matrix.

    Bin masking is adaptive, so small or shallow matrices are not gutted:

    * no coverage: always masked.
    * too few nonzeros: masked when a bin has fewer than
      ``min(min_nnz, max(1, floor(min_nnz_frac * median nnz of covered bins)))``
      stored entries. ``min_nnz`` is therefore an upper cap that only applies
      once typical rows are dense enough to support it; set
      ``min_nnz_frac=None`` for the fixed threshold.
    * low coverage (``mask_low_mad``): masked when log10 coverage is more than
      ``mad_max`` MADs below the median AND coverage is more than
      ``mad_min_fold`` times below the median. The MAD is floored at the value
      Poisson noise alone would give at the median depth. In noise-limited
      data the MAD is tiny, so without these two guards an ordinary 1.5-fold
      dip was masked; ICE corrects such bins and they are now kept.
      ``mad_min_fold=1`` disables the fold guard.

    ``rescale="counts"`` (default) scales the result so its total equals the
    raw total of the balanced entries, i.e. values stay on the read-count
    scale that ``obs_over_exp``'s pseudo-count assumes. ``rescale="unit"``
    gives the cooler convention (mean row sum 1).
    """
    if rescale not in ("counts", "unit"):
        raise ValueError("rescale must be 'counts' or 'unit'")
    m = sp.coo_matrix(sparse)
    n = m.shape[0]
    r, c, v = m.row, m.col, m.data.astype(np.float64)
    keep = (np.abs(r - c) >= ignore_diags) & (v != 0)
    r, c, v = r[keep], c[keep], v[keep]
    cov = np.bincount(r, v, n)
    nnz = np.bincount(r, minlength=n)
    covered = cov > 0
    nnz_thr = min_nnz
    if min_nnz_frac is not None and covered.any():
        nnz_thr = min(min_nnz, max(1, int(min_nnz_frac * np.median(nnz[covered]))))
    bad = ~covered | (nnz < nnz_thr)
    if mask_low_mad and (~bad).any():
        lc = np.log10(cov[~bad])
        med = np.median(lc)
        mad = np.median(np.abs(lc - med))
        mad = max(mad, 0.6745 / (np.log(10) * np.sqrt(10**med)))
        cutoff = min(med - mad_max * mad, med - np.log10(max(mad_min_fold, 1.0)))
        bad |= np.log10(np.where(covered, cov, 1)) < cutoff
    ok = ~bad
    ent = ok[r] & ok[c]
    r, c, v = r[ent], c[ent], v[ent]
    w = np.ones(n)
    for _ in range(max_iter):
        vv = v / (w[r] * w[c])
        s = np.bincount(r, vv, n)
        if not ok.any():
            break
        s = s / s[ok].mean()
        s[bad] = 1.0
        s[s == 0] = 1.0
        w *= s
        if np.abs(s[ok] - 1).max() < tol:
            break
    vv = v / (w[r] * w[c])
    if rescale == "counts":
        tot = vv.sum()
        scale = tot / v.sum() if tot > 0 else 1.0
    else:
        rs = np.bincount(r, vv, n)
        scale = rs[ok].mean() if ok.any() and rs[ok].mean() > 0 else 1.0
    w = w * np.sqrt(scale)
    bias = np.where(ok, 1.0 / w, np.nan)
    bal = sp.coo_matrix((v / (w[r] * w[c]), (r, c)), shape=(n, n)).tocsr()
    return bal, bias


def expected_by_diagonal(mat, mask=None, trim=0.02, smooth=5):
    """Trimmed-mean contact per diagonal (zeros included), smoothed.

    mask: bool array, True = bad bin (excluded). Returns array len N; a
    diagonal with no valid position gets 0. The moving-average window is
    clamped to the matrix size (largest odd width <= N), so any N >= 0 works.
    """
    n = mat.shape[0]
    if n == 0:
        return np.zeros(0)
    ok = np.ones(n, bool) if mask is None else ~np.asarray(mask, bool)
    coo = sp.coo_matrix(mat)
    keep = ok[coo.row] & ok[coo.col] & (coo.col >= coo.row)
    d = (coo.col - coo.row)[keep]
    v = coo.data[keep].astype(np.float64)
    order = np.lexsort((v, d))
    d, v = d[order], v[order]
    bounds = np.searchsorted(d, np.arange(n + 1))
    # valid positions on diagonal k: count of i with ok[i] and ok[i+k]
    f = np.fft.rfft(ok.astype(float), 2 * n)
    npos = np.rint(np.fft.irfft(f.conj() * f, 2 * n)[:n])
    exp = np.zeros(n)
    for k in range(n):
        tot = int(npos[k])
        if tot == 0:
            exp[k] = np.nan
            continue
        x = v[bounds[k] : bounds[k + 1]]
        z = tot - len(x)
        t = int(trim * tot)
        lo = max(t - z, 0)
        sel = x[lo : len(x) - t] if t else x
        exp[k] = sel.sum() / max(tot - 2 * t, 1)
    good = ~np.isnan(exp)
    out = np.where(good, exp, 0.0)
    width = int(smooth) if smooth else 0
    if width > n:
        width = n if n % 2 else n - 1
    if width > 1:
        ker = np.ones(width) / width
        num = np.convolve(out, ker, mode="same")
        den = np.convolve(good.astype(float), ker, mode="same")
        out = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        out[~good] = 0.0
    return out


def obs_over_exp(mat, expected, pseudo=1.0, dense_output=False):
    """Distance-normalised signal, one definition for every input type::

        L[i, j] = log2((obs[i, j] + pseudo) / (expected[|i - j|] + pseudo))

    ``pseudo`` (default 1) is a pseudo-count on the read-count scale, so pass
    raw counts or an ``ice_balance(..., rescale="counts")`` matrix (the
    default); on a unit-sum matrix a pseudo-count of 1 would flatten the map.

    * dense in -> dense out, every pixel computed with the formula (NaN in
      ``mat`` stays NaN).
    * sparse in -> sparse out, formula evaluated on the STORED entries only.
      An unstored pixel has obs = 0, whose exact value is
      ``-log2(expected + pseudo) + log2(pseudo)``; in the sparse result it
      reads as 0 instead. The two agree when expected << pseudo (far from the
      diagonal, where sparsity matters) and differ by up to
      ``log2(1 + expected/pseudo)`` near it.
    * sparse in with ``dense_output=True`` -> dense out, exact everywhere and
      identical to the dense path. Use this when the result is densified
      anyway.
    """
    if pseudo <= 0:
        raise ValueError("pseudo must be > 0")
    expected = np.asarray(expected, dtype=np.float64)
    loge = np.log2(expected + pseudo)
    if sp.issparse(mat):
        c = sp.coo_matrix(mat)
        val = np.log2(c.data.astype(np.float64) + pseudo) - loge[np.abs(c.row - c.col)]
        if not dense_output:
            return sp.coo_matrix((val, (c.row, c.col)), shape=c.shape).tocsr()
        n0, n1 = c.shape
        idx = np.abs(np.subtract.outer(np.arange(n0), np.arange(n1)))
        out = np.log2(pseudo) - loge[idx]
        out[c.row, c.col] = val
        return out
    a = np.asarray(mat, dtype=np.float64)
    idx = np.abs(np.subtract.outer(np.arange(a.shape[0]), np.arange(a.shape[1])))
    return np.log2(a + pseudo) - loge[idx]
