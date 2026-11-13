# Runtime image: Python package plus bwa and samtools (needed for FASTQ/FASTA and SAM inputs).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends bwa samtools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY hic_anomaly ./hic_anomaly
COPY universal_hic_pipeline.py cross_refrence_hits.py ./

RUN pip install --no-cache-dir .

WORKDIR /data
ENTRYPOINT ["hic-anomaly"]
CMD ["--help"]
