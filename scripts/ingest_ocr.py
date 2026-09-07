import argparse
import json

from pathlib import Path
from src.paths import paddle_directory
from importlib.metadata import version

PADDLEOCR_VERSION = version(
    "paddleocr"
)

PADDLEPADDLE_VERSION = version(
    "paddlepaddle"
)

from src.database import get_connection
from src.repository import (
    get_pages_for_source_file,
    get_source_file,
    get_ocr_run,
    insert_ocr_run,
    insert_ocr_region,
    insert_token,
)
from src.tokenize import tokenize_regions


PADDLE_PARAMETERS = {
    "enable_mkldnn": False,
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "return_word_box": True,
}


def ingest_page(
    connection,
    *,
    page_id: int,
    source_page_number: int,
    json_path: Path,
) -> tuple[int, int, int] | None:
    """
    Ingest one saved PaddleOCR page.

    Returns:
        (ocr_run_id, region_count, token_count)

    Returns None if the Paddle run is already present.
    """

    # ---------------------------------
    # Idempotency
    # ---------------------------------

    existing_run = get_ocr_run(
        connection,
        page_id=page_id,
        engine="paddleocr",
    )

    if existing_run is not None:
        return None

    # ---------------------------------
    # Load OCR artifact
    # ---------------------------------

    if not json_path.exists():
        raise FileNotFoundError(
            f"OCR JSON not found: {json_path}"
        )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        regions = json.load(file)

    # Make sure this is the newer JSON format.
    for region in regions:
        if "paddle_parts" not in region:
            raise ValueError(
                f"{json_path} does not contain "
                "'paddle_parts'. Regenerate this OCR page "
                "with return_word_box=True."
            )

    tokens = tokenize_regions(regions)

    # Group tokens by OCR region.
    tokens_by_region = {}

    for token in tokens:
        tokens_by_region.setdefault(
            token["region_index"],
            [],
        ).append(token)

    # ---------------------------------
    # OCR run
    # ---------------------------------

    ocr_run_id = insert_ocr_run(
        connection,
        page_id=page_id,
        engine="paddleocr",
        engine_version=PADDLEOCR_VERSION,
        framework_version=PADDLEPADDLE_VERSION,
        model_name="default English pipeline",
        language="en",
        parameters=PADDLE_PARAMETERS,
        raw_output_path=str(json_path),
    )

    region_count = 0
    token_count = 0

    # ---------------------------------
    # Regions + tokens
    # ---------------------------------

    for region in regions:

        region_id = insert_ocr_region(
            connection,
            ocr_run_id=ocr_run_id,
            region=region,
        )

        region_count += 1

        for token in tokens_by_region.get(
            region["region_index"],
            [],
        ):
            insert_token(
                connection,
                region_id=region_id,
                token=token,
            )

            token_count += 1

    return (
        ocr_run_id,
        region_count,
        token_count,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ingest saved PaddleOCR JSON into PostgreSQL."
        )
    )

    parser.add_argument(
        "--source-file",
        type=int,
        required=True,
        help="source_file_id",
    )   

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--page",
        type=int,
        help="Ingest one source PDF page",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest every page in the source file",
    )

    args = parser.parse_args()

    # ---------------------------------
    # Source file + physical pages
    # ---------------------------------

    with get_connection() as connection:

        source = get_source_file(
            connection,
            source_file_id=args.source_file,
        )

        if source is None:
            raise RuntimeError(
                f"Source file ID "
                f"{args.source_file} "
                f"does not exist."
            )

        pages = get_pages_for_source_file(
            connection,
            source_file_id=args.source_file,
        )

    if not pages:
        raise RuntimeError(
            "No pages found for source file."
        )

    (
        source_file_id,
        corpus_id,
        filename,
        source_path,
        file_sequence,
        page_count,
    ) = source

    # ---------------------------------
    # Derive OCR directory
    # ---------------------------------

    ocr_dir = paddle_directory(
        filename
    )

    if not ocr_dir.exists():
        raise FileNotFoundError(
            f"OCR directory not found: "
            f"{ocr_dir}"
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
                f"{args.page} does not exist."
            )

    # ---------------------------------
    # Get physical pages
    # ---------------------------------

    with get_connection() as connection:

        source = get_source_file(
            connection,
            source_file_id=args.source_file,
        )

        if source is None:
            raise RuntimeError(
                f"Source file ID "
                f"{args.source_file} "
                f"does not exist."
            )

        pages = get_pages_for_source_file(
            connection,
            source_file_id=args.source_file,
        )

    if not pages:
        raise RuntimeError(
            "No pages found for source file."
        )

    if args.page is not None:

        pages = [
            page
            for page in pages
            if page[1] == args.page
        ]

        if not pages:
            raise RuntimeError(
                f"Source page {args.page} "
                "does not exist."
            )

    print()
    print("=== OCR DATABASE INGESTION ===")
    print()

    total_regions = 0
    total_tokens = 0
    inserted_pages = 0
    skipped_pages = 0

    # ---------------------------------
    # One transaction per page
    # ---------------------------------

    for (
        page_id,
        source_page_number,
        corpus_page_number,
        image_path,
    ) in pages:

        page_name = (
            f"page_{source_page_number:04d}"
        )
        
        ocr_dir = (
            Path("data/ocr/paddle")
            / Path(filename).stem
        )
        json_path = (
            ocr_dir
            / f"{page_name}.json"
        )

        try:

            with get_connection() as connection:

                result = ingest_page(
                    connection,
                    page_id=page_id,
                    source_page_number=(
                        source_page_number
                    ),
                    json_path=json_path,
                )

                if result is None:

                    skipped_pages += 1

                    print(
                        f"[SKIP] {page_name}: "
                        "Paddle OCR already ingested"
                    )

                    continue

                (
                    ocr_run_id,
                    region_count,
                    token_count,
                ) = result

                connection.commit()

            inserted_pages += 1
            total_regions += region_count
            total_tokens += token_count

            print(
                f"[ OK ] {page_name}: "
                f"run={ocr_run_id}, "
                f"regions={region_count}, "
                f"tokens={token_count}"
            )

        except Exception as error:

            print()
            print(
                f"[FAIL] {page_name}: {error}"
            )

            raise

    print()
    print("--- Summary ---")
    print(
        f"Pages inserted: {inserted_pages}"
    )
    print(
        f"Pages skipped:  {skipped_pages}"
    )
    print(
        f"Regions:        {total_regions}"
    )
    print(
        f"Tokens:         {total_tokens}"
    )


if __name__ == "__main__":
    main()
