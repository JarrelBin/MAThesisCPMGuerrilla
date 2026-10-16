import argparse

from pathlib import Path

from src.paths import (
    from_project_relative,
    paddle_directory,
)

from src.database import get_connection
from src.ocr_paddle import PaddleOCREngine
from src.repository import (
    get_pages_for_source_file,
    get_source_file,
)


def make_source_name(
    filename: str,
) -> str:
    """
    Convert a PDF filename into the directory name
    used for OCR output.

    Example:
        reports_01.pdf -> reports_01
    """

    return Path(filename).stem


def process_page(
    engine: PaddleOCREngine,
    *,
    source_page_number: int,
    image_path: Path,
    output_dir: Path,
    overwrite: bool,
):
    page_name = (
        f"page_{source_page_number:04d}"
    )

    json_output = (
        output_dir
        / f"{page_name}.json"
    )

    text_output = (
        output_dir
        / f"{page_name}.txt"
    )

    if (
        not overwrite
        and json_output.exists()
        and text_output.exists()
    ):
        print(
            f"[SKIP] {page_name}: "
            "OCR already exists"
        )

        return

    if not image_path.exists():
        raise FileNotFoundError(
            f"Page image not found: "
            f"{image_path}"
        )

    print(
        f"[OCR ] {page_name}"
    )

    regions = engine.recognize_page(
        image_path=image_path,
        json_output_path=json_output,
        text_output_path=text_output,
    )

    print(
        f"      regions: {len(regions)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR on pages registered "
            "in PostgreSQL."
        )
    )

    parser.add_argument(
        "--source-file",
        type=int,
        required=True,
        help="source_file_id",
    )

    group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--page",
        type=int,
        help="Process one source PDF page",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Process all pages",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing OCR output",
    )

    args = parser.parse_args()

    # ---------------------------------
    # Read source/pages from database
    # ---------------------------------

    with get_connection() as connection:

        source = get_source_file(
            connection,
            source_file_id=args.source_file,
        )

        if source is None:
            raise RuntimeError(
                f"Source file "
                f"{args.source_file} "
                f"does not exist."
            )

        pages = get_pages_for_source_file(
            connection,
            source_file_id=args.source_file,
        )

    (
        source_file_id,
        corpus_id,
        filename,
        source_path,
        file_sequence,
        page_count,
    ) = source

    if not pages:
        raise RuntimeError(
            "No registered pages found."
        )

    # ---------------------------------
    # Optional page selection
    # ---------------------------------

    if args.page is not None:

        pages = [
            page
            for page in pages
            if page[1] == args.page
        ]

        if not pages:
            raise RuntimeError(
                f"Source page "
                f"{args.page} not found."
            )

    # ---------------------------------
    # Output directory
    # ---------------------------------

    output_dir = paddle_directory(
        filename
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------
    # Summary
    # ---------------------------------

    print()
    print("=== PaddleOCR ===")
    print()

    print(
        f"Source file ID: {source_file_id}"
    )

    print(
        f"Source:         {filename}"
    )

    print(
        f"Sequence:       {file_sequence}"
    )

    print(
        f"Pages:          {len(pages)}"
    )

    print(
        f"Output:         {output_dir}"
    )

    print()

    # ---------------------------------
    # Load Paddle only once
    # ---------------------------------

    print(
        "Loading PaddleOCR..."
    )

    engine = PaddleOCREngine(
        lang="en",
        device="gpu",
    )

    print()

    # ---------------------------------
    # Process pages
    # ---------------------------------

    for (
        page_id,
        source_page_number,
        corpus_page_number,
        image_path,
    ) in pages:

        process_page(
            engine,
            source_page_number=(
                source_page_number
            ),
            image_path=from_project_relative(
                image_path
            ),
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

    print()
    print(
        "OCR processing complete."
    )


if __name__ == "__main__":
    main()
