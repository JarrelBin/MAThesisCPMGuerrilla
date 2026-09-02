# scripts/run_pipeline.py
from pathlib import Path

from src.ingest import ingest_document
from src.preprocess import pdf_to_images


def main():

    input_path = Path(
        "data/raw/primary_source_to_ocr_sample.pdf"
    )

    image_dir = Path(
        "data/images/primary_source_to_ocr_sample"
    )

    # -------------------------
    # Stage 1: Ingest
    # -------------------------

    print("\n=== DOCUMENT INGESTION ===")

    document = ingest_document(input_path)

    print(f"Filename: {document['filename']}")
    print(f"SHA-256: {document['sha256']}")

    # -------------------------
    # Stage 2: PDF -> images
    # -------------------------

    print("\n=== PDF TO IMAGE CONVERSION ===")

    image_paths = pdf_to_images(
        pdf_path=input_path,
        output_dir=image_dir,
        dpi=300
    )

    print(
        f"\nPages generated: {len(image_paths)}"
    )


if __name__ == "__main__":
    main()
