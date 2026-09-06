import argparse

from src.database import get_connection
from src.repository import split_report


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Split an incorrectly combined "
            "historical report."
        )
    )

    parser.add_argument(
        "report_id",
        type=int,
        help="Existing report_id to split",
    )

    parser.add_argument(
        "source_page",
        type=int,
        help=(
            "Source PDF page that should become "
            "page 1 of the new report"
        ),
    )

    parser.add_argument(
        "--source-file",
        type=int,
        required=True,
        help="source_file_id",
    )

    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help=(
            "Optional raw reference for "
            "the new report"
        ),
    )

    args = parser.parse_args()

    with get_connection() as connection:

        # Resolve source page -> page_id and corpus.
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    p.page_id,
                    sf.corpus_id
                FROM page p
                JOIN source_file sf
                    ON p.source_file_id =
                       sf.source_file_id
                WHERE p.source_file_id = %s
                  AND p.source_page_number = %s;
                """,
                (
                    args.source_file,
                    args.source_page,
                ),
            )

            result = cursor.fetchone()

        if result is None:
            raise RuntimeError(
                "Source page not found."
            )

        page_id, corpus_id = result

        new_report_id = split_report(
            connection,
            report_id=args.report_id,
            split_page_id=page_id,
            corpus_id=corpus_id,
            report_ref_raw=args.ref,
        )

        connection.commit()

    print()
    print("Report split complete.")
    print(
        f"Old report ID: {args.report_id}"
    )
    print(
        f"New report ID: {new_report_id}"
    )


if __name__ == "__main__":
    main()
