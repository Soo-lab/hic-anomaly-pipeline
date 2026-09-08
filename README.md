# Hi-C Anomaly detector

An automated pipeline for detecting structural genomic anomalies from Hi-C contact maps and cross-referencing candidates with haplotype-resolved allelic imbalance.

#format Agnostic
#unsupervised anomaly detection

# Quick Start
```bash
pip install -r requirements.txt

python universal_hic_pipeline.py \
  --input sample.bam \
  --chrom chr21 \
  --screen-bin 100000 \
  --zoom-bin 25000 \
  --outdir ./results
```
