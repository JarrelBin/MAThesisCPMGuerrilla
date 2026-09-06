from src.corpus_text import (
    get_all_reports,
)


def main():

    reports = get_all_reports()

    print(
        f"Reports: {len(reports)}"
    )

    print()

    for report in reports:

        print(
            f"Report ID: "
            f"{report['report_id']}"
        )

        print(
            f"Sequence:  "
            f"{report['report_sequence']}"
        )

        print(
            f"Reference: "
            f"{report['report_ref']}"
        )

        print(
            f"Tokens:    "
            f"{len(report['tokens'])}"
        )

        print()

        print(
            report["text"][:500]
        )

        print()
        print("-" * 70)


if __name__ == "__main__":
    main()
