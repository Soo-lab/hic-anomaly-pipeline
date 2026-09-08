import argparse
import csv
import os
import shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
import pysam
from sklearn.ensemble import IsolationForest

# Format Normalization & Preprocessing (FASTQ / FASTA / SAM -> BAM)


def run_cmd(cmd, step_name):
    """Executes a shell command and streams errors if it fails."""
    print(f"[{step_name}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"Error during {step_name}:\n{result.stderr}\n")
        sys.exit(result.returncode)


def check_tool_exists(tool_name):
    if shutil.which(tool_name) is None:
        raise EnvironmentError(
            f"Required external binary '{tool_name}' not found in PATH. "
            f"Please install it (e.g. via conda: 'conda install -c bioconda {tool_name}')."
        )


def align_fastq_or_fasta(r1_path, r2_path, ref_fasta, outdir, threads=4):
    """Aligns paired FASTQ/FASTA reads using BWA and produces a sorted, indexed BAM."""
    check_tool_exists("bwa")
    check_tool_exists("samtools")

    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.splitext(os.path.basename(r1_path))[0]
    out_bam = os.path.join(outdir, f"{prefix}_aligned_sorted.bam")

    print(f"--> Aligning paired inputs against reference: {ref_fasta}")

    # Index reference if missing (.bwt index check)
    if not os.path.exists(f"{ref_fasta}.bwt"):
        print("--> Reference index missing. Running bwa index...")
        run_cmd(["bwa", "index", ref_fasta], "BWA Index")

    bwa_cmd = [
        "bwa",
        "mem",
        "-5SP",
        "-t",
        str(threads),
        ref_fasta,
        r1_path,
        r2_path,
    ]
    sort_cmd = [
        "samtools",
        "sort",
        "-@",
        str(threads),
        "-o",
        out_bam,
        "-",
    ]

    p1 = subprocess.Popen(bwa_cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(sort_cmd, stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.communicate()

    if p2.returncode != 0:
        raise RuntimeError(
            "Alignment and sorting pipeline failed via BWA/samtools."
        )

    # Index the final BAM
    run_cmd(["samtools", "index", out_bam], "Index BAM")
    return out_bam


def convert_sam_to_bam(sam_path, outdir, threads=4):
    """Converts a raw SAM file to a coordinate-sorted and indexed BAM."""
    check_tool_exists("samtools")
    os.makedirs(outdir, exist_ok=True)

    prefix = os.path.splitext(os.path.basename(sam_path))[0]
    out_bam = os.path.join(outdir, f"{prefix}_converted_sorted.bam")

    print(f"--> Converting and sorting SAM to BAM: {sam_path}")
    sort_cmd = ["samtools", "sort", "-@", str(threads), "-o", out_bam, sam_path]
    run_cmd(sort_cmd, "SAM to Sorted BAM")

    run_cmd(["samtools", "index", out_bam], "Index BAM")
    return out_bam


def standardize_input_to_bam(args):
    """Detects input formats and standardizes to an indexed, coordinate-sorted BAM."""
    in_path = args.input
    in_lower = in_path.lower()

    # Case 1: Standard BAM
    if in_lower.endswith(".bam"):
        if not (
            os.path.exists(in_path + ".bai")
            or os.path.exists(in_path.replace(".bam", ".bai"))
        ):
            print("--> BAM index missing. Indexing with pysam...")
            pysam.index(in_path)
        return in_path

    # Case 2: Plain SAM
    elif in_lower.endswith(".sam"):
        return convert_sam_to_bam(in_path, args.outdir, threads=args.threads)

    # Case 3: Paired FASTQ / FASTA
    elif any(
        in_lower.endswith(ext)
        for ext in [
            ".fq",
            ".fastq",
            ".fq.gz",
            ".fastq.gz",
            ".fa",
            ".fasta",
            ".fa.gz",
            ".fasta.gz",
        ]
    ):
        if not args.input2:
            raise ValueError(
                "FASTQ/FASTA format detected. You must provide the second mate via --input2 (paired-end reads)."
            )
        if not args.ref_fasta:
            raise ValueError(
                "Raw sequence alignment requires a reference FASTA genome via --ref-fasta."
            )
        return align_fastq_or_fasta(
            args.input,
            args.input2,
            args.ref_fasta,
            args.outdir,
            threads=args.threads,
        )

    else:
        raise ValueError(
            f"Unsupported file format for input: '{in_path}'. "
            f"Accepted formats: .bam, .sam, .fastq/.fq (.gz), .fasta/.fa (.gz)"
        )

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
        raise ValueError(
            f"'{chrom}' not found in BAM header. Available: {avail}..."
        )
    chrom_len = bam.get_reference_length(bam_chrom)
    bam.close()
    return bam_chrom, chrom_len


# Contact Matrix Construction


def create_contact_matrix(bam_path, chrom, chrom_len, bin_size):
    num_bins = int(np.ceil(chrom_len / bin_size))
    matrix = np.zeros((num_bins, num_bins), dtype=np.int32)

    bamfile = pysam.AlignmentFile(bam_path, "rb")
    read_pairs = {}
    total_reads = 0
    used_pairs = 0

    try:
        for read in bamfile.fetch(chrom):
            total_reads += 1

            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_duplicate:
                continue
            # Keep cis contacts only
            if read.next_reference_id != read.reference_id:
                continue

            qname = read.query_name
            if qname not in read_pairs:
                read_pairs[qname] = read
                continue

            read1 = read_pairs.pop(qname)
            pos1 = read1.reference_start
            pos2 = read.reference_start

            if pos1 < 0 or pos2 < 0:
                continue

            bin1 = pos1 // bin_size
            bin2 = pos2 // bin_size

            if bin1 >= num_bins or bin2 >= num_bins:
                continue

            if bin1 == bin2:
                matrix[bin1, bin2] += 1
            else:
                matrix[bin1, bin2] += 1
                matrix[bin2, bin1] += 1
            used_pairs += 1

    finally:
        bamfile.close()

    return matrix, total_reads, used_pairs


# Feature Extraction & Anomaly Detection


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def extract_bin_features(matrix):
    M = matrix.astype(np.float64)
    n = M.shape[0]
    total = M.sum(axis=1)

    local5 = np.zeros(n)
    local10 = np.zeros(n)
    diagonal = np.diag(M).copy()

    for i in range(n):
        a5, b5 = max(0, i - 5), min(n, i + 6)
        a10, b10 = max(0, i - 10), min(n, i + 11)
        local5[i] = M[i, a5:b5].sum()
        local10[i] = M[i, a10:b10].sum()

    distal = np.maximum(total - local10, 0)
    local_distal_ratio = local10 / (distal + 1.0)

    features = np.column_stack([
        np.log1p(total),
        np.log1p(local5),
        np.log1p(local10),
        np.log1p(distal),
        np.log1p(diagonal),
        np.log1p(local_distal_ratio),
        robust_z(total),
        robust_z(local10),
    ])
    return features


def run_ai(matrix, contamination=0.01, random_state=42):
    # Mask unmappable bins (zero contacts) from training to prevent false positives
    total_contacts = matrix.sum(axis=1)
    mappable_indices = np.where(total_contacts > 0)[0]

    features = extract_bin_features(matrix)
    labels = np.ones(matrix.shape[0], dtype=int)
    scores = np.zeros(matrix.shape[0], dtype=float)

    if len(mappable_indices) == 0:
        return labels == -1, scores

    train_features = features[mappable_indices]
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_features)

    raw_scores = -model.decision_function(train_features)
    lo, hi = np.percentile(raw_scores, 1), np.percentile(raw_scores, 99)

    if hi <= lo:
        norm_scores = np.zeros_like(raw_scores)
    else:
        norm_scores = np.clip((raw_scores - lo) / (hi - lo), 0, 1)

    train_preds = model.predict(train_features)
    labels[mappable_indices] = train_preds
    scores[mappable_indices] = norm_scores

    anomaly = labels == -1
    return anomaly, scores


def merge_adjacent_bins(
    anomaly_indices,
    scores,
    bin_size,
    chrom,
    chrom_len,
    merge_gap_bins=1,
):
    if len(anomaly_indices) == 0:
        return []

    anomaly_indices = np.sort(np.asarray(anomaly_indices))
    regions = []

    def make_region(start, end, max_score):
        end_bp = min((end + 1) * bin_size, chrom_len)
        start_bp = start * bin_size
        return {
            "chrom": chrom,
            "start_bin": start,
            "end_bin": end,
            "start_bp": start_bp,
            "end_bp": end_bp,
            "start_Mb": start_bp / 1e6,
            "end_Mb": end_bp / 1e6,
            "size_Mb": (end_bp - start_bp) / 1e6,
            "max_ai_score": max_score,
        }

    start = int(anomaly_indices[0])
    previous = int(anomaly_indices[0])
    max_score = float(scores[previous])

    for idx in anomaly_indices[1:]:
        idx = int(idx)
        if idx - previous <= merge_gap_bins + 1:
            previous = idx
            max_score = max(max_score, float(scores[idx]))
        else:
            regions.append(make_region(start, previous, max_score))
            start = previous = idx
            max_score = float(scores[idx])

    regions.append(make_region(start, previous, max_score))
    return regions


# Plotting & Zooming


def plot_matrix(matrix, chrom, bin_size, output_path, title="Hi-C Heatmap"):
    plt.figure(figsize=(10, 10))
    contact_map = np.log1p(matrix)
    plt.imshow(
        contact_map,
        cmap="Reds",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    plt.title(f"{title}: {chrom} ({bin_size // 1000}kb bins)")
    plt.xlabel("Genomic Bin")
    plt.ylabel("Genomic Bin")
    plt.colorbar(label="log(1 + contacts)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def zoom_region_from_bam(
    bam_path,
    chrom,
    chrom_len,
    start_bp,
    end_bp,
    zoom_bin_size,
    padding_bp,
    outdir,
    region_id,
):
    start_bp = max(0, start_bp - padding_bp)
    end_bp = min(chrom_len, end_bp + padding_bp)

    zoom_len = end_bp - start_bp
    num_bins = int(np.ceil(zoom_len / zoom_bin_size))
    matrix = np.zeros((num_bins, num_bins), dtype=np.int32)

    bamfile = pysam.AlignmentFile(bam_path, "rb")
    read_pairs = {}

    try:
        for read in bamfile.fetch(chrom, start_bp, end_bp):
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_duplicate:
                continue
            if read.next_reference_id != read.reference_id:
                continue

            # Skip pairs whose mate starts outside the zoom region
            if not (start_bp <= read.next_reference_start < end_bp):
                continue

            qname = read.query_name
            if qname not in read_pairs:
                read_pairs[qname] = read
                continue

            read1 = read_pairs.pop(qname)
            pos1, pos2 = read1.reference_start, read.reference_start

            bin1 = (pos1 - start_bp) // zoom_bin_size
            bin2 = (pos2 - start_bp) // zoom_bin_size

            if bin1 >= num_bins or bin2 >= num_bins:
                continue

            if bin1 == bin2:
                matrix[bin1, bin2] += 1
            else:
                matrix[bin1, bin2] += 1
                matrix[bin2, bin1] += 1
    finally:
        bamfile.close()

    out_img = os.path.join(
        outdir, f"zoom_{region_id}_{zoom_bin_size // 1000}kb.png"
    )
    plot_matrix(
        matrix,
        f"{chrom}:{start_bp:,}-{end_bp:,}",
        zoom_bin_size,
        out_img,
        title="Hi-C Zoomed",
    )
    return out_img

# Main Orchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Universal Hi-C Analysis Pipeline: Accepts BAM, SAM, FASTQ, or FASTA."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file: BAM, SAM, or R1 FASTQ/FASTA (.gz supported)",
    )
    parser.add_argument(
        "--input2",
        default=None,
        help="Input R2 mate file (Required for FASTQ/FASTA)",
    )
    parser.add_argument(
        "--ref-fasta",
        default=None,
        help="Reference genome FASTA file (Required if aligning raw FASTQ/FASTA)",
    )
    parser.add_argument(
        "--chrom",
        required=True,
        help="Target chromosome (e.g. chr1 or 1)",
    )
    parser.add_argument(
        "--screen-bin",
        type=int,
        default=1_000_000,
        help="Screening bin size (default: 1 Mb)",
    )
    parser.add_argument(
        "--zoom-bin",
        type=int,
        default=250_000,
        help="Zoom bin size (default: 250 kb)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="AI contamination fraction (default: 0.01)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads for BWA / samtools sorting (default: 4)",
    )
    parser.add_argument(
        "--outdir",
        default="universal_hic_output",
        help="Output directory",
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Standardize any format to sorted, indexed BAM
    bam_file = standardize_input_to_bam(args)

    # 2. Extract reference specs
    chrom_name, chrom_len = resolve_chrom_and_length(bam_file, args.chrom)
    print(f"[✓] Target: {chrom_name} ({chrom_len:,} bp)")

    # 3. Build low-res screening matrix
    print(f"--> Building whole-chromosome matrix at {args.screen_bin // 1000}kb...")
    matrix, total_reads, used_pairs = create_contact_matrix(
        bam_file, chrom_name, chrom_len, args.screen_bin
    )
    print(f"    Total reads: {total_reads:,} | Valid pairs: {used_pairs:,}")

    coarse_png = os.path.join(args.outdir, f"hic_{chrom_name}_screening.png")
    plot_matrix(
        matrix,
        chrom_name,
        args.screen_bin,
        coarse_png,
        title="Whole-Chromosome Screening",
    )

    # 4. Anomaly detection
    print(f"--> Running Isolation Forest (contamination={args.contamination})...")
    anomalies, scores = run_ai(matrix, contamination=args.contamination)
    anomaly_bins = np.where(anomalies)[0]
    regions = merge_adjacent_bins(
        anomaly_bins, scores, args.screen_bin, chrom_name, chrom_len
    )
    print(f"    Flagged candidate regions: {len(regions)}")

    calls_tsv = os.path.join(args.outdir, f"anomaly_candidates_{chrom_name}.tsv")
    with open(calls_tsv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chrom",
                "start_bin",
                "end_bin",
                "start_bp",
                "end_bp",
                "start_Mb",
                "end_Mb",
                "size_Mb",
                "max_ai_score",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(regions)

    # 5. Fine zoom
    zoom_dir = os.path.join(args.outdir, "zoomed_regions")
    os.makedirs(zoom_dir, exist_ok=True)
    for i, r in enumerate(regions[:5], 1):
        zoom_region_from_bam(
            bam_file,
            chrom_name,
            chrom_len,
            r["start_bp"],
            r["end_bp"],
            args.zoom_bin,
            padding_bp=500_000,
            outdir=zoom_dir,
            region_id=f"{i:02d}_{int(r['start_Mb'])}Mb",
        )

    print(f"\n[✓] Pipeline finished successfully. Results saved in: {args.outdir}")


if __name__ == "__main__":
    main()
