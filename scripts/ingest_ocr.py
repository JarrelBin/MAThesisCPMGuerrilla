import argparse
import json

from importlib.metadata import (version,PackageNotFoundError,)
from pathlib import Path

from src.database import get_connection
from src.paths import (
    paddle_directory,
    to_project_relative,
)
from src.repository import (
    get_ocr_run,
    get_pages_for_ocr_ingestion,
    get_source_file,
    insert_ocr_region,
    insert_ocr_run,
    insert_token,
)
from src.tokenize import tokenize_regions


# ============================================================
# Installed OCR versions
# ============================================================

PADDLEOCR_VERSION = version(
    "paddleocr"
)
def get_paddlepaddle_version() -> str:
    """
    Retrieve the PaddlePaddle version on either
    CPU or GPU installations.
    """

    for package_name in (
        "paddlepaddle-gpu",
        "paddlepaddle",
    ):
        try:
            return version(package_name)

        except PackageNotFoundError:
            continue

    raise RuntimeError(
        "Neither paddlepaddle nor paddlepaddle-gpu "
        "is installed."
    )


PADDLEPADDLE_VERSION = get_paddlepaddle_version()


# ============================================================
# OCR provenance
#
# Keep this synchronized with the actual PaddleOCR settings
# used by src/ocr_paddle.py.
# ============================================================

PADDLE_PARAMETERS = {
    "device": "cpu",
    "enable_mkldnn": False,
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "return_word_box": True,
}


# ============================================================
# Check Chinese characters
#
#
# 
# ============================================================

def contains_chinese(text: str) -> bool:
    """
    Detect Han ideographs in an OCR token.

    Excludes Chinese-script tokens while preserving
    English and romanized names.
    """
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        or "\U00020000" <= char <= "\U000323AF"
        for char in text
    )

# ============================================================
# Single-page ingestion
# ============================================================

def ingest_page(
    connection,
    *,
    page_id: int,
    json_path: Path,
) -> tuple[int, int, int] | None:
    """
    Ingest one saved PaddleOCR JSON artifact.

    Returns:
        (
            ocr_run_id,
            region_count,
            token_count,
        )

    Returns None when this PaddleOCR run has already
    been ingested for the page.

    Confirmed sketch handling occurs before this function
    is called.
    """

    # --------------------------------------------------------
    # Idempotency
    # --------------------------------------------------------

    existing_run = get_ocr_run(
        connection,
        page_id=page_id,
        engine="paddleocr",
    )

    if existing_run is not None:
        return None

    # --------------------------------------------------------
    # OCR artifact
    # --------------------------------------------------------

    if not json_path.is_file():
        raise FileNotFoundError(
            f"OCR JSON not found: "
            f"{json_path}"
        )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        regions = json.load(
            file
        )

    # --------------------------------------------------------
    # Validate expected Paddle format
    #
    # We require paddle_parts because our tokenizer relies
    # on Paddle's word/component boxes.
    # --------------------------------------------------------

    for region in regions:

        if "paddle_parts" not in region:
            raise ValueError(
                f"{json_path} does not contain "
                "'paddle_parts'. "
                "Regenerate this OCR page with "
                "return_word_box=True."
            )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    tokens = tokenize_regions(
        regions
    )
    
    # Exclude tokens containing Han characters.
    # Original PaddleOCR JSON remains unchanged.
    tokens = [
        token
        for token in tokens
        if not contains_chinese(
            str(token["raw_token"])
        )
    ]

    tokens_by_region = {}

    for token in tokens:

        region_index = token[
            "region_index"
        ]

        tokens_by_region.setdefault(
            region_index,
            [],
        ).append(
            token
        )

    # --------------------------------------------------------
    # OCR run
    # --------------------------------------------------------

    ocr_run_id = insert_ocr_run(
        connection,
        page_id=page_id,
        engine="paddleocr",
        engine_version=PADDLEOCR_VERSION,
        framework_version=PADDLEPADDLE_VERSION,
        model_name="default English pipeline",
        language="en",
        parameters=PADDLE_PARAMETERS,

        # Store a portable project-relative path rather
        # than an Ubuntu/Windows-specific absolute path.
        raw_output_path=to_project_relative(
            json_path
        ),
    )

    region_count = 0
    token_count = 0

    # --------------------------------------------------------
    # Regions + tokens
    # --------------------------------------------------------

    for region in regions:

        region_id = insert_ocr_region(
            connection,
            ocr_run_id=ocr_run_id,
            region=region,
        )

        region_count += 1

        region_tokens = (
            tokens_by_region.get(
                region["region_index"],
                [],
            )
        )

        for token in region_tokens:

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


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Ingest saved PaddleOCR JSON "
            "into PostgreSQL."
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
        help=(
            "Ingest one source PDF page"
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Ingest every eligible page "
            "in the source file"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Source + pages
    # ========================================================

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

        # IMPORTANT:
        #
        # Use the OCR-specific repository function here
        # because we need page_type and
        # page_type_reviewed.
        pages = get_pages_for_ocr_ingestion(
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

    # ========================================================
    # Paddle output directory
    # ========================================================

    ocr_dir = paddle_directory(
        filename
    )

    if not ocr_dir.is_dir():
        raise FileNotFoundError(
            f"OCR directory not found: "
            f"{ocr_dir}"
        )

    # ========================================================
    # Optional one-page selection
    # ========================================================

    if args.page is not None:

        pages = [
            page
            for page in pages
            if page[1] == args.page
        ]

        if not pages:
            raise RuntimeError(
                f"Source page "
                f"{args.page} "
                f"does not exist."
            )

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "=== OCR DATABASE INGESTION ==="
    )
    print()

    print(
        f"Source file ID: {source_file_id}"
    )

    print(
        f"Source:         {filename}"
    )

    print(
        f"OCR directory:  {ocr_dir}"
    )

    print()

    # ========================================================
    # Counters
    # ========================================================

    inserted_pages = 0

    skipped_existing = 0
    skipped_sketches = 0

    total_regions = 0
    total_tokens = 0

    # ========================================================
    # One transaction per page
    # ========================================================

    for (
        page_id,
        source_page_number,
        corpus_page_number,
        image_path,
        page_type,
        page_type_reviewed,
    ) in pages:

        page_name = (
            f"page_{source_page_number:04d}"
        )

        # ----------------------------------------------------
        # Confirmed sketch
        #
        # Paddle JSON/TXT remains on disk for provenance.
        # We simply exclude the page from the textual
        # PostgreSQL OCR/token layer.
        # ----------------------------------------------------

        if (
            page_type == "sketch"
            and page_type_reviewed
        ):

            skipped_sketches += 1

            print(
                f"[SKIP] {page_name}: "
                "confirmed sketch"
            )

            continue

        # ----------------------------------------------------
        # OCR artifact
        # ----------------------------------------------------

        json_path = (
            ocr_dir
            / f"{page_name}.json"
        )

        # ----------------------------------------------------
        # Ingest
        # ----------------------------------------------------

        try:

            with get_connection() as connection:

                result = ingest_page(
                    connection,
                    page_id=page_id,
                    json_path=json_path,
                )

                # --------------------------------------------
                # Existing OCR run
                # --------------------------------------------

                if result is None:

                    skipped_existing += 1

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

            # -----------------------------------------------
            # Successful page
            # -----------------------------------------------

            inserted_pages += 1

            total_regions += (
                region_count
            )

            total_tokens += (
                token_count
            )

            print(
                f"[ OK ] {page_name}: "
                f"run={ocr_run_id}, "
                f"regions={region_count}, "
                f"tokens={token_count}"
            )

        except Exception as error:

            print()
            print(
                f"[FAIL] {page_name}: "
                f"{error}"
            )

            raise

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("--- Summary ---")
    print()

    print(
        f"Pages inserted:          "
        f"{inserted_pages}"
    )

    print(
        f"Existing pages skipped:  "
        f"{skipped_existing}"
    )

    print(
        f"Sketch pages skipped:    "
        f"{skipped_sketches}"
    )

    print(
        f"Regions inserted:        "
        f"{total_regions}"
    )

    print(
        f"Tokens inserted:         "
        f"{total_tokens}"
    )


if __name__ == "__main__":
    main()
