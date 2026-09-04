import argparse
from pathlib import Path

from src.ocr_paddle import PaddleOCREngine


DOCUMENT = "primary_source_to_ocr_sample"


def main():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on a document page."
    )

    parser.add_argument(
        "page",
        type=int,
        help="Page number to OCR",
    )

    args = parser.parse_args()

    page_name = f"page_{args.page:04d}"

    image_path = Path(
        f"data/images/{DOCUMENT}/{page_name}.png"
    )

    output_dir = Path(
        f"data/ocr/paddle/{DOCUMENT}"
    )

    json_output = output_dir / f"{page_name}.json"
    text_output = output_dir / f"{page_name}.txt"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("Loading PaddleOCR...")

    engine = PaddleOCREngine(
        lang="en"
    )

    print(f"Processing: {image_path}")

    regions = engine.recognize_page(
        image_path=image_path,
        json_output_path=json_output,
        text_output_path=text_output,
    )

    print()
    print(f"Detected regions: {len(regions)}")
    print(f"JSON: {json_output}")
    print(f"Text: {text_output}")


if __name__ == "__main__":
    main()
