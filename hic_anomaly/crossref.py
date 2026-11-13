import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from hic_anomaly.stats_utils import (
    betabinom_two_sided,
    bh_fdr,
    block_overdispersion,
    estimate_rho_mom,
    log2_ratio_ci,
)


def parse_region(region_str):
    """Parses strings like 'chr1:45000000-47000000' or '1:45,000,000-47,000,000'."""
    region_clean = region_str.replace(",", "").strip()
    chrom, coords = region_clean.split(":")
    start_str, end_str = coords.split("-")
    return chrom, int(start_str), int(end_str)


def normalize_chrom(chrom_name):
    """Strips 'chr' prefix to eliminate chromosome naming mismatches."""
    chrom = str(chrom_name).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def _load_anomalies(path):
    df = pd.read_csv(path, sep=None, engine="python")
    if "region" in df.columns:
        parsed = df["region"].apply(parse_region)
        df["chrom"] = [p[0] for p in parsed]
        df["start"] = [p[1] for p in parsed]
        df["end"] = [p[2] for p in parsed]
    # Accept the column names written by merge_adjacent_bins / the CLI.
    for new, old in (("start", "start_bp"), ("end", "end_bp")):
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    missing = {"chrom", "start", "end"} - set(df.columns)
    if missing:
        raise ValueError(f"anomalies file lacks columns: {sorted(missing)}")
    df["norm_chrom"] = df["chrom"].apply(normalize_chrom)
    return df


def score_bins(df_imb, p0=0.5, rho=None, trim=0.05):
    """Beta-binomial test of every bin (genome-wide family). Adds rho, p_bin, q_bin."""
    k = df_imb["hap1_count"].to_numpy(dtype=float)
    n = k + df_imb["hap2_count"].to_numpy(dtype=float)
    if rho is None:
        rho = estimate_rho_mom(k, n, mu=p0, trim=trim)
    p = betabinom_two_sided(k, n, mu=p0, rho=rho)
    p[n == 0] = np.nan
    return rho, p, bh_fdr(p)


def cross_reference(
    anomalies_file,
    imbalance_file,
    output_file,
    min_imbalance_thresh=0.15,
    buffer_bp=250_000,
    p0=0.5,
    fdr=0.05,
    rho=None,
    robust=True,
    center="auto",
):
    """Test allelic imbalance in anomaly regions (from `detect`) with a beta-binomial model.

    `min_imbalance_thresh` (default 0.15, same as the CLI `--min_imbalance`) is the minimum
    mean |(H1-H2)/(H1+H2)| a region needs, in addition to q <= `fdr`, to be validated.
    Without hap1_count/hap2_count columns no p-value can be computed: p/q are NaN, a
    UserWarning is raised and no region is validated.

    Per region: k = summed hap1 reads over overlapping bins, n = hap1 + hap2. Null:
    k ~ BetaBin(n, mean p0, overdispersion rho); rho is estimated by method of moments
    on the trimmed bulk of all bins (or fixed via `rho`) and rescaled for summing bins.
    Two-sided exact p; BH q across all tested regions (the candidate family). `q_all`
    additionally pools every bin in the imbalance file into the family.

    robust=True (default): the overdispersion used for a region is measured at the
    REGION scale. Sliding windows of the same number of bins are tiled along each
    chromosome and `block_overdispersion` fits an effective rho to their sums, so
    block-correlated bins (shared mapping bias, correlated allelic ratios) are
    accounted for (a cluster-aware effective n) instead of assuming independent bins.
    With center="auto" the null mean is the median window fraction (empirical null
    centre; absorbs genome-wide reference bias, i.e. a genomic-control style location
    correction; use center="fixed" to test against `p0` exactly). Fallback to the
    independent-bin rescaling when fewer than 20 windows exist or if `rho` is fixed.
    Limits: (i) the window rho is estimated from the same data (noisy for few
    windows, mildly inflated when many regions are truly imbalanced); (ii) a genuine
    chromosome-wide imbalance is absorbed by center="auto"; (iii) p-values remain
    anti-conservative for structure that is correlated at scales larger than the
    region or varies between regions. robust=False restores the independent-bin
    model, which is anti-conservative under correlated bins. No causal
    interpretation is implied.
    """
    df_anom = _load_anomalies(anomalies_file)
    if not robust and center != "fixed":
        warnings.warn(
            "independent-bins mode ignores center='auto': the null mean is p0="
            f"{p0} (equivalent to center='fixed'); no empirical re-centring is done.",
            UserWarning,
            stacklevel=2,
        )

    # Haplotype-resolved allelic counts. Columns: chrom, start, end, hap1_count, hap2_count
    df_imb = pd.read_csv(imbalance_file, sep=None, engine="python")
    df_imb["norm_chrom"] = df_imb["chrom"].apply(normalize_chrom)
    have_counts = {"hap1_count", "hap2_count"} <= set(df_imb.columns)
    if have_counts:
        df_imb["hap1_count"] = df_imb["hap1_count"].fillna(0)
        df_imb["hap2_count"] = df_imb["hap2_count"].fillna(0)
    if "imbalance" not in df_imb.columns:
        if not have_counts:
            raise ValueError("imbalance file needs hap1_count/hap2_count or imbalance")
        total = df_imb["hap1_count"] + df_imb["hap2_count"]
        df_imb["imbalance"] = np.where(
            total > 0, (df_imb["hap1_count"] - df_imb["hap2_count"]) / total.where(total > 0, 1), 0.0
        )

    if not have_counts:
        warnings.warn(
            "imbalance file has no hap1_count/hap2_count columns: p_value/q_value cannot be "
            "computed (NaN) and cross_validated will be False for every region. Provide "
            "read counts for a statistical test.",
            UserWarning,
            stacklevel=2,
        )

    chr_means = df_imb.groupby("norm_chrom")["imbalance"].mean().to_dict()
    chr_stds = (
        df_imb.groupby("norm_chrom")["imbalance"].std().replace(0, 1e-6).fillna(1e-6).to_dict()
    )
    if have_counts:
        rho_hat, p_bin, _ = score_bins(df_imb, p0, rho)
    else:
        rho_hat, p_bin = 0.0, np.array([])

    by_chrom = {c: g for c, g in df_imb.groupby("norm_chrom")}
    win_cache = {}

    def region_null(m):
        """(mu, rho) for a region of m bins, measured on windows of m bins; cached."""
        if m not in win_cache:
            srt = [g.sort_values("start") for g in by_chrom.values()]
            ks = [g["hap1_count"].to_numpy(float) for g in srt]
            ns = [(g["hap1_count"] + g["hap2_count"]).to_numpy(float) for g in srt]
            fixed = p0 if center == "fixed" else None
            win_cache[m] = block_overdispersion(ks, ns, m, mu=fixed)
        return win_cache[m]

    results = []
    for _, a in df_anom.iterrows():
        base = {"chrom": a["chrom"], "start": a["start"], "end": a["end"]}
        g = by_chrom.get(a["norm_chrom"])
        if g is not None:
            q_start = max(0, a["start"] - buffer_bp)
            ov = g[(g["end"] > q_start) & (g["start"] < a["end"] + buffer_bp)]
        else:
            ov = df_imb.iloc[0:0]
        if ov.empty:
            results.append({**base, "overlap_bin_count": 0, "total_phased_reads": 0,
                            "mean_imbalance": 0.0, "mean_abs_imbalance": 0.0,
                            "max_abs_imbalance": 0.0, "skew_direction": "NONE",
                            "z_score": 0.0, "p_value": np.nan, "p_value_ttest": np.nan,
                            "log2_effect": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                            "rho": rho_hat})
            continue

        imb = ov["imbalance"].to_numpy()
        mean_imb = float(np.mean(imb))
        if len(imb) > 1 and np.std(imb) > 0:
            p_t = float(stats.ttest_1samp(imb, popmean=0.0).pvalue)  # legacy, ignores n
        else:
            p_t = np.nan
        z = (mean_imb - chr_means.get(a["norm_chrom"], 0.0)) / chr_stds.get(a["norm_chrom"], 1.0)

        row = {**base, "overlap_bin_count": len(ov), "total_phased_reads": 0,
               "mean_imbalance": round(mean_imb, 4),
               "mean_abs_imbalance": round(float(np.mean(np.abs(imb))), 4),
               "max_abs_imbalance": round(float(np.max(np.abs(imb))), 4),
               "skew_direction": "HAP1_GAIN" if mean_imb > 0.05
               else ("HAP2_GAIN" if mean_imb < -0.05 else "BALANCED"),
               "z_score": round(float(z), 3), "p_value": np.nan, "p_value_ttest": p_t,
               "log2_effect": np.nan, "ci_low": np.nan, "ci_high": np.nan, "rho": rho_hat}
        if have_counts:
            nb = (ov["hap1_count"] + ov["hap2_count"]).to_numpy(dtype=float)
            k1, nn = float(ov["hap1_count"].sum()), float(nb.sum())
            row["total_phased_reads"] = int(nn)
            if nn > 1:
                # Variance of summed bins -> equivalent rho on the pooled n.
                rho_eff = rho_hat * float((nb * (nb - 1)).sum()) / (nn * (nn - 1))
                mu_use = p0
                if robust and rho is None:
                    mu_w, rho_w, _ = region_null(len(ov))
                    if np.isfinite(rho_w):
                        rho_eff = rho_w
                        mu_use = mu_w
                row["rho_region"] = rho_eff
                row["p0_used"] = mu_use
                row["p_value"] = float(betabinom_two_sided(int(k1), int(nn), mu_use, rho_eff)[0])
                eff, lo, hi = log2_ratio_ci(k1, nn)
                row.update(log2_effect=float(eff), ci_low=float(lo), ci_high=float(hi))
        results.append(row)

    out_df = pd.DataFrame(results)
    out_df["q_value"] = bh_fdr(out_df["p_value"].to_numpy())
    if len(p_bin):
        pooled = bh_fdr(np.concatenate([out_df["p_value"].to_numpy(), p_bin]))
        out_df["q_all"] = pooled[: len(out_df)]
    else:
        out_df["q_all"] = np.nan
    out_df["cross_validated"] = (
        (out_df["mean_abs_imbalance"] >= min_imbalance_thresh) & (out_df["q_value"] <= fdr)
    )

    out_df = out_df.sort_values(
        by=["cross_validated", "q_value", "mean_abs_imbalance"],
        ascending=[False, True, False],
        na_position="last",
    )
    out_df.to_csv(output_file, sep="\t", index=False)
    print(f"[ok] Saved {len(out_df)} candidates to: {output_file} (rho={rho_hat:.4f})")
    print(f"     Cross-validated hits: {int(out_df['cross_validated'].sum())} / {len(out_df)}")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Cross-reference Hi-C contact anomalies with haplotype imbalance."
    )
    parser.add_argument(
        "--anomalies",
        required=True,
        help="Path to anomaly calls from the detect step (CSV/TSV with region or chrom,start,end)",
    )
    parser.add_argument(
        "--imbalance",
        required=True,
        help="Binned allelic imbalance TSV/CSV: chrom,start,end,hap1_count,hap2_count "
        "(counts are required for p-values)",
    )
    parser.add_argument(
        "--out",
        default="cross_validated_anomalies.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--min_imbalance",
        type=float,
        default=0.15,
        help="Minimum |(H1-H2)/(H1+H2)| threshold to validate a hit (default: 0.15)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=250_000,
        help="Base-pair buffer flanking the anomaly coordinates (default: 250kb)",
    )

    parser.add_argument("--p0", type=float, default=0.5, help="Expected hap1 fraction (0.5)")
    parser.add_argument("--fdr", type=float, default=0.05, help="BH q threshold (0.05)")
    parser.add_argument(
        "--rho", type=float, default=None, help="Fix beta-binomial rho (default: estimate)"
    )
    parser.add_argument(
        "--independent-bins",
        action="store_true",
        help="Use the bin-level rho rescaled for independent bins (anti-conservative if "
        "bins are correlated); default measures overdispersion at the region scale",
    )
    parser.add_argument(
        "--center",
        choices=("auto", "fixed"),
        default="auto",
        help="auto: null mean = median window fraction; fixed: test against --p0",
    )

    args = parser.parse_args()
    cross_reference(
        args.anomalies,
        args.imbalance,
        args.out,
        args.min_imbalance,
        args.buffer,
        p0=args.p0,
        fdr=args.fdr,
        rho=args.rho,
        robust=not args.independent_bins,
        center=args.center,
    )


if __name__ == "__main__":
    main()
