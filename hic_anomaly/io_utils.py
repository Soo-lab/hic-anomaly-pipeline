import logging
import os
import shutil
import subprocess
import sys
import tempfile

import pysam

logger = logging.getLogger(__name__)

# Format Normalization & Preprocessing (FASTQ / FASTA / SAM -> BAM)


def run_cmd(cmd, step_name):
    """Executes a shell command and streams errors if it fails."""
    logger.info(f"[{step_name}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Error during {step_name}:\n{result.stderr}")
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

    logger.info(f"--> Aligning paired inputs against reference: {ref_fasta}")

    # Index reference if missing (.bwt index check)
    if not os.path.exists(f"{ref_fasta}.bwt"):
        logger.info("--> Reference index missing. Running bwa index...")
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

    with tempfile.TemporaryFile() as err1, tempfile.TemporaryFile() as err2:
        p1 = subprocess.Popen(bwa_cmd, stdout=subprocess.PIPE, stderr=err1)
        try:
            p2 = subprocess.Popen(sort_cmd, stdin=p1.stdout, stdout=subprocess.DEVNULL, stderr=err2)
        except Exception:
            p1.kill()
            p1.wait()
            p1.stdout.close()
            raise
        p1.stdout.close()  # let bwa get SIGPIPE if samtools dies
        rc2 = p2.wait()
        rc1 = p1.wait()
        if rc1 != 0 or rc2 != 0:
            err1.seek(0)
            err2.seek(0)
            raise RuntimeError(
                f"Alignment pipeline failed (bwa exit {rc1}, samtools sort "
                f"exit {rc2}).\nbwa stderr tail:\n"
                f"{err1.read()[-2000:].decode(errors='replace')}\n"
                f"samtools stderr tail:\n"
                f"{err2.read()[-2000:].decode(errors='replace')}"
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

    logger.info(f"--> Converting and sorting SAM to BAM: {sam_path}")
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
            os.path.exists(in_path + ".bai") or os.path.exists(in_path.replace(".bam", ".bai"))
        ):
            logger.info("--> BAM index missing. Indexing with pysam...")
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
