import argparse
import json
from pathlib import Path

from src.tokenize import (
    tokenize_regions,
    tokenize_text,
)


DOCUMENT = "primary_source_to_ocr_sample"


TEST_STRINGS = [
    "D.S.B.",
    "H.S.B.",
    "K/Lumpur",
    "CTs",
    "2nd",
    "(f)",
    "Area 'E'",
    "SAMNAK OK",
    "Inspector-General",
    "Kedah/Perlis",
    "BAN BALA",
]


def print_rule_tests():
    print()
    print("=== TOKENIZATION RULE TESTS ===")
    print()

    for text in TEST_STRINGS:

        tokens = tokenize_text(text)

        rendered = [
            (
                token["raw_token"],
                token["token_type"],
            )
            for token in tokens
        ]

        print(
            f"{text!r:25} -> {rendered}"
        )


def print_page_tokens(page: int):
    page_name = f"page_{page:04d}"

    ocr_path = Path(
        f"data/ocr/paddle/"
        f"{DOCUMENT}/"
        f"{page_name}.json"
    )

    if not ocr_path.exists():
        raise FileNotFoundError(
            f"OCR JSON not found: {ocr_path}"
        )

    with ocr_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        regions = json.load(file)

    tokens = tokenize_regions(
        regions
    )

    print()
    print(
        f"=== PAGE {page:04d} TOKENS ==="
    )
    print()

    print(
        f"Total tokens: {len(tokens)}"
    )
    print()

    for token in tokens:

        print(
            f"{token['position_in_page']:4}  "
            f"{token['raw_token']!r:20} "
            f"{token['token_type']:12} "
            f"region="
            f"{token['region_index']:02} "
            f"part="
            f"{token['part_index']:02}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect historical OCR tokenization."
        )
    )

    parser.add_argument(
        "page",
        type=int,
        nargs="?",
        help=(
            "Optional OCR page to inspect, "
            "e.g. 2, 4, or 6"
        ),
    )

    parser.add_argument(
        "--rules",
        action="store_true",
        help=(
            "Run standalone tokenization "
            "rule tests."
        ),
    )

    args = parser.parse_args()

    if args.rules:
        print_rule_tests()

    if args.page is not None:
        print_page_tokens(
            args.page
        )

    if not args.rules and args.page is None:
        parser.error(
            "Provide a page number "
            "and/or --rules."
        )


if __name__ == "__main__":
    main()
