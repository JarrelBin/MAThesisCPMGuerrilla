import argparse

from src.database import get_connection
from src.paths import PROJECT_ROOT


def get_report(
    connection,
    report_id: int,
):
    """
    Retrieve report metadata.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                r.report_id,
                r.report_sequence,
                COALESCE(
                    r.report_ref,
                    r.report_ref_raw
                ) AS report_ref
            FROM report r
            WHERE r.report_id = %s;
            """,
            (report_id,),
        )

        return cursor.fetchone()


def get_report_tokens(
    connection,
    report_id: int,
):
    """
    Retrieve effective analytical tokens for an entire
    report in correct page/token order.

    Accepted corrections are already substituted by
    analytical_token.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                report_page_number,
                position_in_page,
                token,
                token_type,
                raw_token,
                was_corrected
            FROM analytical_token
            WHERE report_id = %s
            ORDER BY
                report_page_number,
                position_in_page;
            """,
            (report_id,),
        )

        return cursor.fetchall()


def detokenize(tokens: list[tuple[str, str]]) -> str:
    """
    Reconstruct readable text from analytical tokens.

    This is intended for exported/readable text rather
    than token-based NLP analysis.
    """

    text = ""

    no_space_before = {
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        ")",
        "]",
        "}",
        "%",
    }

    no_space_after = {
        "(",
        "[",
        "{",
    }

    for token, token_type in tokens:

        if not text:
            text = token
            continue

        if token in no_space_before:
            text += token

        elif text[-1] in no_space_after:
            text += token

        else:
            text += " " + token

    return text


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export one complete corrected "
            "historical report."
        )
    )

    parser.add_argument(
        "report_id",
        type=int,
        help="Database report_id to export",
    )

    args = parser.parse_args()

    # =========================================================
    # Retrieve report
    # =========================================================

    with get_connection() as connection:

        report = get_report(
            connection,
            args.report_id,
        )

        if report is None:
            raise RuntimeError(
                f"Report ID {args.report_id} "
                "does not exist."
            )

        rows = get_report_tokens(
            connection,
            args.report_id,
        )

    if not rows:
        raise RuntimeError(
            f"No analytical tokens found for "
            f"report ID {args.report_id}."
        )

    (
        report_id,
        report_sequence,
        report_ref,
    ) = report

    # =========================================================
    # Group tokens by report page
    # =========================================================

    pages = {}

    corrected_count = 0

    for (
        report_page_number,
        position_in_page,
        token,
        token_type,
        raw_token,
        was_corrected,
    ) in rows:

        pages.setdefault(
            report_page_number,
            [],
        ).append(
            (
                token,
                token_type,
            )
        )

        if was_corrected:
            corrected_count += 1

    # =========================================================
    # Reconstruct report
    #
    # Preserve page boundaries as blank lines.
    # =========================================================

    page_texts = []

    for page_number in sorted(pages):

        page_text = detokenize(
            pages[page_number]
        )

        page_texts.append(
            page_text
        )

    report_text = "\n\n".join(
        page_texts
    )

    # =========================================================
    # Output directory
    # =========================================================

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "ocr"
        / "corrected"
        / "reports"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"report_"
            f"{report_sequence:04d}"
            f"_id_{report_id}.txt"
        )
    )

    output_path.write_text(
        report_text + "\n",
        encoding="utf-8",
    )

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("=== CORRECTED REPORT EXPORT ===")
    print()

    print(
        f"Report ID:       {report_id}"
    )

    print(
        f"Report sequence: {report_sequence}"
    )

    print(
        f"Reference:       "
        f"{report_ref or '[none]'}"
    )

    print(
        f"Pages:           {len(pages)}"
    )

    print(
        f"Tokens:          {len(rows)}"
    )

    print(
        f"Corrected tokens:{corrected_count:>6}"
    )

    print()

    print(
        f"Exported: {output_path}"
    )


if __name__ == "__main__":
    main()
