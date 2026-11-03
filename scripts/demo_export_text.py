
"""Export corrected OCR tokens as plain text."""

import argparse
from pathlib import Path

from src.database import get_connection



def export_text(
    source_id: int,
    output: Path,
    corrected: bool = True,
):
    token_expression = (
        "COALESCE(c.corrected_token, t.raw_token)"
        if corrected
        else "t.raw_token"
    )

    query = f"""
        SELECT
            p.source_page_number,
            t.position_in_page,
            {token_expression} AS final_token,
            t.token_type

        FROM token t

        JOIN ocr_region r
            ON t.region_id = r.region_id

        JOIN ocr_run o
            ON r.ocr_run_id = o.ocr_run_id

        JOIN page p
            ON o.page_id = p.page_id

        LEFT JOIN token_correction c
            ON c.token_id = t.token_id
            AND c.accepted = TRUE

        WHERE p.source_file_id = %s

        ORDER BY
            p.source_page_number,
            t.position_in_page;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (source_id,))
            rows = cursor.fetchall()

    if not rows:
        print("No OCR tokens found.")
        return

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages = {}

    for page, position, token, token_type in rows:
        pages.setdefault(page, []).append(
            (token, token_type)
        )

    with output.open("w", encoding="utf-8") as file:
        for page_number, tokens in pages.items():
            file.write(f"=== PAGE {page_number} ===\n\n")

            text = ""

            for token, token_type in tokens:
                if (
                    token_type == "punctuation"
                    and token in (".", ",", ";", ":", "!", "?", ")", "]")
                ):
                    text = text.rstrip() + token
                elif token in ("(", "["):
                    text += " " + token
                else:
                    text += " " + token

            file.write(text.strip())
            file.write("\n\n")

    print(f"Exported: {output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Export original uncorrected OCR tokens",
    )
    args = parser.parse_args()

    export_text(
        args.source_file,
        args.output,
        corrected=not args.raw,
    )

if __name__ == "__main__":
    main()

