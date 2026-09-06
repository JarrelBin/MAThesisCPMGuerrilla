import argparse
import json
from pathlib import Path

from src.database import get_connection
from src.report_detection import (
    detect_report_boundary,
)
from src.repository import (
    get_pages_for_source_file,
    get_source_file,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect candidate historical "
            "report boundaries."
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
    )

    group.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    with get_connection() as connection:

        source = get_source_file(
            connection,
            source_file_id=args.source_file,
        )

        if source is None:
            raise RuntimeError(
                "Source file not found."
            )

        pages = get_pages_for_source_file(
            connection,
            source_file_id=args.source_file,
        )

    filename = source[2]

    source_name = Path(
        filename
    ).stem

    ocr_dir = (
        Path("data/ocr/paddle")
        / source_name
    )

    if args.page is not None:

        pages = [
            page
            for page in pages
            if page[1] == args.page
        ]

        if not pages:
            raise RuntimeError(
                "Requested page not found."
            )

    print()
    print(
        "=== REPORT BOUNDARY DETECTION ==="
    )
    print()

    print(
        f"Source: {filename}"
    )
    print()

    candidate_count = 0

    for (
        page_id,
        source_page_number,
        corpus_page_number,
        image_path,
    ) in pages:

        json_path = (
            ocr_dir
            / (
                f"page_"
                f"{source_page_number:04d}"
                f".json"
            )
        )

        if not json_path.exists():
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

        result = detect_report_boundary(
            regions
        )

        if result.is_candidate:
            candidate_count += 1

        marker = (
            "REPORT START"
            if result.is_candidate
            else "-"
        )

        signals = []

        if result.has_our_ref:
            signals.append(
                "OUR REF"
            )

        if (
            result.fuzzy_our_ref
            and not result.has_our_ref
        ):
            signals.append(
                f"FUZZY OUR REF "
                f"({result.our_ref_score:.1f})"
            )

        if result.has_statement_by:
            signals.append(
                "STATEMENT BY"
            )

        if result.has_statement_of:
            signals.append(
                "STATEMENT OF"
            )

        if result.has_interview_report:
            signals.append(
                "INTERVIEW REPORT"
            )

        signal_text = (
            ", ".join(signals)
            if signals
            else "none"
        )

        print(
            f"Page "
            f"{source_page_number:04d} "
            f"| {marker:12} "
            f"| {signal_text}"
        )

        if result.report_ref_raw:

            print(
                f"          REF: "
                f"{result.report_ref_raw}"
            )

    print()
    print(
        f"Candidate report starts: "
        f"{candidate_count}"
    )


if __name__ == "__main__":
    main()
