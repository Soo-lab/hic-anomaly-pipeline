import argparse
import numpy as np
import pandas as pd
from scipy import stats


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


def cross_reference(
    anomalies_file,
    imbalance_file,
    output_file,
    min_imbalance_thresh=0.20,
    buffer_bp=500_000,
):
    # 1. Load Isolation Forest anomaly candidates
    df_anom = pd.read_csv(anomalies_file, sep=None, engine="python")

    # Standardize column names
    if "region" in df_anom.columns:
        parsed = df_anom["region"].apply(parse_region)
        df_anom["chrom"] = [p[0] for p in parsed]
        df_anom["start"] = [p[1] for p in parsed]
        df_anom["end"] = [p[2] for p in parsed]

    df_anom["norm_chrom"] = df_anom["chrom"].apply(normalize_chrom)

    # 2. Load haplotype-resolved allelic imbalance data
    # Expected TSV columns: chrom, start, end, hap1_count, hap2_count, imbalance
    df_imb = pd.read_csv(imbalance_file, sep=None, engine="python")
    df_imb["norm_chrom"] = df_imb["chrom"].apply(normalize_chrom)

    # Compute imbalance if missing
    if "imbalance" not in df_imb.columns:
        total = df_imb["hap1_count"] + df_imb["hap2_count"]
        # Safe divide avoiding ZeroDivisionError
        df_imb["imbalance"] = np.where(
            total > 0, (df_imb["hap1_count"] - df_imb["hap2_count"]) / total, 0.0
        )

    # Background chromosome-level statistics for z-scoring
    chr_means = df_imb.groupby("norm_chrom")["imbalance"].mean().to_dict()
    chr_stds = (
        df_imb.groupby("norm_chrom")["imbalance"]
        .std()
        .replace(0, 1e-6)
        .fillna(1e-6)
        .to_dict()
    )

    results = []

    # 3. Intersect and cross-reference
    for _, anom_row in df_anom.iterrows():
        chrom = anom_row["norm_chrom"]
        # Add buffer to account for Hi-C bin boundary resolution limits
        q_start = max(0, anom_row["start"] - buffer_bp)
        q_end = anom_row["end"] + buffer_bp

        # Overlap mask
        mask = (
            (df_imb["norm_chrom"] == chrom)
            & (df_imb["end"] > q_start)
            & (df_imb["start"] < q_end)
        )
        overlapping_bins = df_imb[mask]

        if overlapping_bins.empty:
            results.append(
                {
                    "chrom": anom_row["chrom"],
                    "start": anom_row["start"],
                    "end": anom_row["end"],
                    "overlap_bin_count": 0,
                    "mean_abs_imbalance": 0.0,
                    "max_abs_imbalance": 0.0,
                    "skew_direction": "NONE",
                    "p_value": 1.0,
                    "z_score": 0.0,
                    "cross_validated": False,
                }
            )
            continue

        imb_vals = overlapping_bins["imbalance"].values
        hap1_sum = (
            overlapping_bins["hap1_count"].sum()
            if "hap1_count" in overlapping_bins
            else 0
        )
        hap2_sum = (
            overlapping_bins["hap2_count"].sum()
            if "hap2_count" in overlapping_bins
            else 0
        )

        mean_imb = float(np.mean(imb_vals))
        mean_abs_imb = float(np.mean(np.abs(imb_vals)))
        max_abs_imb = float(np.max(np.abs(imb_vals)))

        # One-sample t-test: does this region's imbalance deviate significantly from 0?
        if len(imb_vals) > 1 and np.std(imb_vals) > 0:
            t_stat, p_val = stats.ttest_1samp(imb_vals, popmean=0.0)
        else:
            p_val = 1.0

        # Genome/chromosome relative z-score
        c_mean = chr_means.get(chrom, 0.0)
        c_std = chr_stds.get(chrom, 1.0)
        z_score = (mean_imb - c_mean) / c_std

        # Cross-validation decision logic
        is_supported = (mean_abs_imb >= min_imbalance_thresh) and (
            p_val < 0.05 or abs(z_score) >= 2.0
        )

        skew_dir = (
            "HAP1_GAIN"
            if mean_imb > 0.05
            else ("HAP2_GAIN" if mean_imb < -0.05 else "BALANCED")
        )

        results.append(
            {
                "chrom": anom_row["chrom"],
                "start": anom_row["start"],
                "end": anom_row["end"],
                "overlap_bin_count": len(overlapping_bins),
                "total_phased_reads": int(hap1_sum + hap2_sum),
                "mean_imbalance": round(mean_imb, 4),
                "mean_abs_imbalance": round(mean_abs_imb, 4),
                "max_abs_imbalance": round(max_abs_imb, 4),
                "skew_direction": skew_dir,
                "z_score": round(float(z_score), 3),
                "p_value": float(f"{p_val:.2e}"),
                "cross_validated": is_supported,
            }
        )

    out_df = pd.DataFrame(results)

    # Sort high-priority hits to the top
    out_df.sort_values(
        by=["cross_validated", "mean_abs_imbalance", "z_score"],
        ascending=[False, False, False],
        inplace=True,
    )

    out_df.to_csv(output_file, sep="\t", index=False)
    print(
        f"[✓] Analysis complete. Saved {len(out_df)} candidates to: {output_file}"
    )
    print(
        f"    Cross-validated hits: {out_df['cross_validated'].sum()} / {len(out_df)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-reference Hi-C contact anomalies with haplotype imbalance."
    )
    parser.add_argument(
        "--anomalies",
        required=True,
        help="Path to Isolation Forest flagged regions (CSV/TSV)",
    )
    parser.add_argument(
        "--imbalance",
        required=True,
        help="Path to Script 3 allelic imbalance binned TSV",
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

    args = parser.parse_args()
    cross_reference(
        args.anomalies,
        args.imbalance,
        args.out,
        args.min_imbalance,
        args.buffer,
    )
