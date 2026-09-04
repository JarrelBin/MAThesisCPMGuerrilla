from pathlib import Path

from src.ocr_paddle import PaddleOCREngine


def main():

    image_path = Path(
        "data/images/"
        "primary_source_to_ocr_sample/"
        "page_0006.png"
    )

    output_dir = Path(
        "data/ocr/paddle/"
        "primary_source_to_ocr_sample"
    )

    json_output = (
        output_dir / "page_0006.json"
    )

    text_output = (
        output_dir / "page_0006.txt"
    )

    print("Loading PaddleOCR...")

    engine = PaddleOCREngine(
        lang="en"
    )

    print(
        f"Processing: {image_path}"
    )

    regions = engine.recognize_page(
        image_path=image_path,
        json_output_path=json_output,
        text_output_path=text_output,
    )

    print(
        f"Detected regions: {len(regions)}"
    )

    print(
        f"JSON saved: {json_output}"
    )

    print(
        f"Text saved: {text_output}"
    )


if __name__ == "__main__":
    main()
