import argparse
import json

from src.database import get_connection
from src.page_classification import (
    classify_from_ocr,
)
from src.paths import paddle_directory
from src.repository import (
    get_source_file,
)


def get_pages_for_classification(
    connection,
    *,
    source_file_id: int,
):
    """
    Retrieve pages together with their existing
    classification/review state.

    Human-reviewed pages are returned so the script can
    explicitly report that they were protected.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                page_id,
                source_page_number,
                corpus_page_number,
                page_type,
                page_type_reviewed
            FROM page
            WHERE source_file_id = %s
            ORDER BY source_page_number;
            """,
            (source_file_id,),
        )

        return cursor.fetchall()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Identify probable sketch/diagram pages "
            "from saved PaddleOCR output."
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
        help="Classify one source PDF page",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Classify all pages",
    )

    args = parser.parse_args()

    # =========================================================
    # Source + pages
    # =========================================================

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

        pages = get_pages_for_classification(
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

    ocr_dir = paddle_directory(
        filename
    )

    if not ocr_dir.exists():
        raise FileNotFoundError(
            f"OCR directory not found: "
            f"{ocr_dir}"
        )

    # =========================================================
    # Optional one-page selection
    # =========================================================

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

    # =========================================================
    # Classification
    # =========================================================

    print()
    print("=== PAGE CLASSIFICATION ===")
    print()

    print(
        f"Source: {filename}"
    )

    print()

    probable_sketches = 0
    normal_pages = 0
    protected_pages = 0

    for (
        page_id,
        source_page_number,
        corpus_page_number,
        page_type,
        page_type_reviewed,
    ) in pages:

        page_name = (
            f"page_{source_page_number:04d}"
        )

        # -----------------------------------------------------
        # HUMAN REVIEW ALWAYS WINS
        #
        # Never allow an algorithmic rerun to overwrite a
        # classification already reviewed by a human.
        # -----------------------------------------------------

        if page_type_reviewed:

            protected_pages += 1

            print(
                f"[KEEP] {page_name} "
                f"| human-reviewed "
                f"| type={page_type}"
            )

            continue

        # -----------------------------------------------------
        # Paddle artifact
        # -----------------------------------------------------

        json_path = (
            ocr_dir
            / f"{page_name}.json"
        )

        if not json_path.exists():
            raise FileNotFoundError(
                f"OCR JSON missing: "
                f"{json_path}"
            )

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            regions = json.load(
                file
            )

        result = classify_from_ocr(
            regions
        )

        # -----------------------------------------------------
        # Probable sketch
        # -----------------------------------------------------

        if result["probable_sketch"]:

            probable_sketches += 1

            with get_connection() as connection:

                with connection.cursor() as cursor:

                    cursor.execute(
                        """
                        UPDATE page
                        SET
                            page_type = 'sketch',
                            page_type_reviewed = FALSE
                        WHERE
                            page_id = %s
                            AND page_type_reviewed = FALSE;
                        """,
                        (page_id,),
                    )

                connection.commit()

            marker = "SKETCH"

        # -----------------------------------------------------
        # Not a sketch candidate
        # -----------------------------------------------------

        else:

            normal_pages += 1

            # Important:
            #
            # Do not change page_type to 'report' here.
            # Report segmentation owns that decision.
            #
            # If this page was previously an unreviewed
            # algorithmic sketch candidate, however, remove
            # that stale candidate classification.
            with get_connection() as connection:

                with connection.cursor() as cursor:

                    cursor.execute(
                        """
                        UPDATE page
                        SET page_type = 'unknown'
                        WHERE
                            page_id = %s
                            AND page_type = 'sketch'
                            AND page_type_reviewed = FALSE;
                        """,
                        (page_id,),
                    )

                connection.commit()

            marker = "-"

        # -----------------------------------------------------
        # Diagnostic output
        # -----------------------------------------------------

        print(
            f"[{marker:6}] "
            f"{page_name} "
            f"| score={result['score']} "
            f"| words={result['word_count']} "
            f"| regions={result['region_count']} "
            f"| long={result['long_region_count']} "
            f"| short="
            f"{result['short_region_ratio']:.2f}"
        )

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("--- Summary ---")
    print()

    print(
        f"Probable sketches: "
        f"{probable_sketches}"
    )

    print(
        f"Not classified as sketch: "
        f"{normal_pages}"
    )

    print(
        f"Human-reviewed/protected: "
        f"{protected_pages}"
    )


if __name__ == "__main__":
    main()
