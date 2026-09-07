# OCR Models

## McCATMuS

Model: McCATMuS_nfd_nofix_V1.mlmodel

DOI: 10.5281/zenodo.13788177

Download:

    kraken get 10.5281/zenodo.13788177

Environment setup

1. Clone repository
2. Create venv
3. Install Python dependencies
4. Install appropriate PaddlePaddle build:

   CPU-only machine:
       CPU-compatible PaddlePaddle

   GPU-enabled machine:
       NVIDIA/CUDA-compatible PaddlePaddle

5. Verify Paddle installation
6. Select CPU/GPU configuration
7. Run OCR

==========================
psql \
  -h localhost \
  -U digital_history \
  -d defector_reports \
  -f sql/schema.sql


python -m scripts.ingest_source \
    data/raw/source_01.pdf \
    --sequence 1

python -m scripts.ingest_source \
    data/raw/source_01.pdf \
    --sequence 2

python -m scripts.run_paddle \
    --source-file 1 \
    --all

python -m scripts.run_paddle \
    --source-file 2 \
    --all

python -m scripts.detect_reports \
    --source-file 1 \
    --all

python -m scripts.detect_reports \
    --source-file 2 \
    --all
    
python -m scripts.segment_reports \
    --source-file 1

python -m scripts.segment_reports \
    --source-file 2

# Missed Boundary
python -m scripts.split_report \
    <report_id> \
    <source_page_number> \
    --source-file <source_file_id>

# False Boundary
python -m scripts.merge_reports \
    <keep_report_id> \
    <wrong_report_id>

python -m scripts.ingest_ocr \
    --source-file 1 \
    --all

python -m scripts.ingest_ocr \
    --source-file 2 \
    --all
    
python -m scripts.review_corrections \
    --page 4 #optional

python -m scripts.export_corrected_report \
    <report_id>

# Summary
python -m scripts.ingest_source data/raw/source_01.pdf --sequence 1

python -m scripts.run_paddle --source-file 1 --all

python -m scripts.detect_reports --source-file 1 --all

python -m scripts.segment_reports --source-file 1

python -m scripts.ingest_ocr --source-file 1 --all

python -m scripts.review_corrections
