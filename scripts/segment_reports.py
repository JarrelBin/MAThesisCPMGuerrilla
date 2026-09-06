import argparse
import json
from pathlib import Path

from src.database import get_connection
from src.report_detection import (
    detect_report_boundary,
)
from src.repository import (
    insert_report,
    assign_page_to_report,
    recalculate_report_sequences,
)


CORPUS_NAME = "Defector Reports"


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

    with get_connection() as connection:

        # --------------------------------
        # Find corpus/source file
        # --------------------------------

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    sf.corpus_id,
                    sf.filename
                FROM source_file sf
                WHERE sf.source_file_id = %s;
                """,
                (args.source_file,),
            )

            source = cursor.fetchone()

        if source is None:
            raise RuntimeError(
                "Source file not found."
            )

        corpus_id = source[0]
        filename = source[1]

        source_name = Path(filename).stem

        ocr_dir = (
            Path("data/ocr/paddle")
            / source_name
        )

        # --------------------------------
        # Get physical pages
        # --------------------------------

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

        # --------------------------------
        # Safety check
        # --------------------------------

        already_segmented = any(
            _page_has_report(
                connection,
                page_id
            )
            for page_id, _, _ in pages
        )

        if already_segmented:
            raise RuntimeError(
                "This source file already contains "
                "report assignments. Refusing to "
                "segment automatically."
            )

        print()
        print(
            f"Segmenting: {filename}"
        )
        print()

        current_report_id = None
        report_page_number = 0
        detected_reports = 0

        # --------------------------------
        # Walk pages
        # --------------------------------

        for (
            page_id,
            source_page_number,
            corpus_page_number,
        ) in pages:

            json_path = (
                args.ocr_dir
                / f"page_{source_page_number:04d}.json"
            )

            if not json_path.exists():
                raise FileNotFoundError(
                    f"OCR JSON missing: {json_path}"
                )

            with json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                regions = json.load(file)

            detection = (
                detect_report_boundary(
                    regions
                )
            )

            # ----------------------------
            # New report
            # ----------------------------

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
                    f"Page {source_page_number:04d} "
                    f"-> NEW REPORT "
                    f"{current_report_id}"
                )

                if detection.report_ref_raw:
                    print(
                        f"             REF: "
                        f"{detection.report_ref_raw}"
                    )

                continue

            # ----------------------------
            # Continuation page
            # ----------------------------

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
                    f"Page {source_page_number:04d} "
                    f"-> report "
                    f"{current_report_id}, "
                    f"page "
                    f"{report_page_number}"
                )

            else:
                # Anything before the first detected
                # report remains unassigned.
                print(
                    f"Page {source_page_number:04d} "
                    f"-> unassigned/front matter"
                )

        # --------------------------------
        # Recalculate logical report order
        # --------------------------------

        recalculate_report_sequences(
            connection,
            corpus_id=corpus_id,
        )

        connection.commit()

    print()
    print(
        f"Reports detected: "
        f"{detected_reports}"
    )


def _page_has_report(
    connection,
    page_id: int,
) -> bool:

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


if __name__ == "__main__":
    main()
