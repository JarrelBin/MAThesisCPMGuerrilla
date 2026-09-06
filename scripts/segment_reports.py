import argparse
import json

from src.paths import paddle_directory
from src.database import get_connection
from src.report_detection import detect_report_boundary
from src.repository import (
    insert_report,
    assign_page_to_report,
    recalculate_report_sequences,
)


def page_has_report(
    connection,
    page_id: int,
) -> bool:
    """
    Return True if the page is already assigned
    to a historical report.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT report_id
            FROM page
            WHERE page_id = %s;
            """,
            (page_id,),
        )

        result = cursor.fetchone()

    return (
        result is not None
        and result[0] is not None
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Segment a source file into "
            "historical reports."
        )
    )

    parser.add_argument(
        "--source-file",
        type=int,
        required=True,
        help="source_file_id to segment",
    )

    args = parser.parse_args()

    # =========================================================
    # Database transaction
    # =========================================================

    with get_connection() as connection:

        # -----------------------------------------------------
        # Find source file
        # -----------------------------------------------------

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    corpus_id,
                    filename
                FROM source_file
                WHERE source_file_id = %s;
                """,
                (args.source_file,),
            )

            source = cursor.fetchone()

        if source is None:
            raise RuntimeError(
                f"Source file ID "
                f"{args.source_file} "
                f"does not exist."
            )

        corpus_id, filename = source

        # -----------------------------------------------------
        # Portable OCR directory
        #
        # src.paths resolves this relative to PROJECT_ROOT.
        # -----------------------------------------------------

        ocr_dir = paddle_directory(
            filename
        )

        if not ocr_dir.exists():
            raise FileNotFoundError(
                f"OCR directory not found: "
                f"{ocr_dir}"
            )

        # -----------------------------------------------------
        # Physical pages
        # -----------------------------------------------------

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    page_id,
                    source_page_number,
                    corpus_page_number
                FROM page
                WHERE source_file_id = %s
                ORDER BY source_page_number;
                """,
                (args.source_file,),
            )

            pages = cursor.fetchall()

        if not pages:
            raise RuntimeError(
                "No pages registered for "
                "this source file."
            )

        # -----------------------------------------------------
        # Safety check
        #
        # Do not automatically overwrite existing
        # report assignments.
        # -----------------------------------------------------

        already_segmented = any(
            page_has_report(
                connection,
                page_id,
            )
            for (
                page_id,
                _,
                _,
            ) in pages
        )

        if already_segmented:
            raise RuntimeError(
                "This source file already contains "
                "report assignments. Refusing to "
                "segment automatically."
            )

        # =====================================================
        # Segmentation
        # =====================================================

        print()
        print("=== REPORT SEGMENTATION ===")
        print()

        print(
            f"Source file ID: {args.source_file}"
        )

        print(
            f"Source:         {filename}"
        )

        print(
            f"OCR directory:  {ocr_dir}"
        )

        print()

        current_report_id = None
        report_page_number = 0

        detected_reports = 0
        unassigned_pages = 0

        for (
            page_id,
            source_page_number,
            corpus_page_number,
        ) in pages:

            # ---------------------------------------------
            # Saved Paddle OCR JSON
            # ---------------------------------------------

            json_path = (
                ocr_dir
                / f"page_{source_page_number:04d}.json"
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

            # ---------------------------------------------
            # Detect report boundary
            # ---------------------------------------------

            detection = (
                detect_report_boundary(
                    regions
                )
            )

            # =============================================
            # New report
            # =============================================

            if detection.is_candidate:

                current_report_id = (
                    insert_report(
                        connection,
                        corpus_id=corpus_id,
                        report_ref_raw=(
                            detection.report_ref_raw
                        ),
                    )
                )

                detected_reports += 1

                report_page_number = 1

                assign_page_to_report(
                    connection,
                    page_id=page_id,
                    report_id=current_report_id,
                    report_page_number=1,
                    is_front_page=True,
                )

                print(
                    f"[NEW ] "
                    f"page "
                    f"{source_page_number:04d} "
                    f"-> report_id="
                    f"{current_report_id}"
                )

                if detection.report_ref_raw:
                    print(
                        f"       REF: "
                        f"{detection.report_ref_raw}"
                    )

                continue

            # =============================================
            # Continuation page
            # =============================================

            if current_report_id is not None:

                report_page_number += 1

                assign_page_to_report(
                    connection,
                    page_id=page_id,
                    report_id=current_report_id,
                    report_page_number=(
                        report_page_number
                    ),
                    is_front_page=False,
                )

                print(
                    f"[CONT] "
                    f"page "
                    f"{source_page_number:04d} "
                    f"-> report_id="
                    f"{current_report_id}, "
                    f"report_page="
                    f"{report_page_number}"
                )

                continue

            # =============================================
            # Before first detected report
            # =============================================

            unassigned_pages += 1

            print(
                f"[NONE] "
                f"page "
                f"{source_page_number:04d} "
                f"-> unassigned/front matter"
            )

        # =====================================================
        # Recalculate logical report ordering
        #
        # report_id NEVER changes.
        # report_sequence is derived from physical page order.
        # =====================================================

        recalculate_report_sequences(
            connection,
            corpus_id=corpus_id,
        )

        # Commit all report assignments together.
        connection.commit()

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("--- Summary ---")
    print()

    print(
        f"Reports detected: "
        f"{detected_reports}"
    )

    print(
        f"Unassigned pages: "
        f"{unassigned_pages}"
    )

    print()
    print(
        "Report segmentation complete."
    )


if __name__ == "__main__":
    main()
