import argparse

from src.database import get_connection
from src.repository import merge_reports


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Merge a falsely separated report "
            "back into another report."
        )
    )

    parser.add_argument(
        "keep_report_id",
        type=int,
        help=(
            "Report ID that should remain"
        ),
    )

    parser.add_argument(
        "merge_report_id",
        type=int,
        help=(
            "Report ID whose pages should "
            "be moved into the retained report"
        ),
    )

    args = parser.parse_args()

    with get_connection() as connection:

        # Find corpus of retained report.
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT corpus_id
                FROM report
                WHERE report_id = %s;
                """,
                (args.keep_report_id,),
            )

            result = cursor.fetchone()

        if result is None:
            raise RuntimeError(
                "Retained report does not exist."
            )

        corpus_id = result[0]

        merge_reports(
            connection,
            keep_report_id=args.keep_report_id,
            merge_report_id=args.merge_report_id,
            corpus_id=corpus_id,
        )

        connection.commit()

    print()
    print("Reports merged.")
    print(
        f"Retained report ID: "
        f"{args.keep_report_id}"
    )
    print(
        f"Removed report ID: "
        f"{args.merge_report_id}"
    )


if __name__ == "__main__":
    main()
