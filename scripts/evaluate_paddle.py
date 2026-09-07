import argparse
from pathlib import Path

from src.evaluate import evaluate_text
from src.paths import GROUND_TRUTH_DIR, PADDLE_OCR_DIR


DEFAULT_PILOT_PAGES = [
    2,
    4,
    6,
]


def get_gt_path(
    source_name: str,
    page_number: int,
) -> Path:
    """
    Return canonical ground-truth path for one page.
    """

    return (
        GROUND_TRUTH_DIR
        / "canonical"
        / source_name
        / f"page_{page_number:04d}.txt"
    )


def get_paddle_path(
    source_name: str,
    page_number: int,
) -> Path:
    """
    Return raw PaddleOCR text path for one page.
    """

    return (
        PADDLE_OCR_DIR
        / source_name
        / f"page_{page_number:04d}.txt"
    )


def evaluate_page(
    source_name: str,
    page_number: int,
) -> dict:
    """
    Evaluate one PaddleOCR page against canonical GT.
    """

    gt_path = get_gt_path(
        source_name,
        page_number,
    )

    paddle_path = get_paddle_path(
        source_name,
        page_number,
    )

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Canonical GT not found: {gt_path}"
        )

    if not paddle_path.exists():
        raise FileNotFoundError(
            f"Paddle OCR not found: {paddle_path}"
        )

    reference = gt_path.read_text(
        encoding="utf-8"
    )

    hypothesis = paddle_path.read_text(
        encoding="utf-8"
    )

    result = evaluate_text(
        reference,
        hypothesis,
    )

    return {
        "page": page_number,
        "cer": result["cer"],
        "wer": result["wer"],
        "reference": result["reference"],
        "hypothesis": result["hypothesis"],
    }


def evaluate_combined(
    results: list[dict],
) -> dict:
    """
    Evaluate all selected pages as one combined text.

    We deliberately recompute CER/WER over the combined
    reference/hypothesis instead of averaging page scores.
    """

    reference = " ".join(
        result["reference"]
        for result in results
    )

    hypothesis = " ".join(
        result["hypothesis"]
        for result in results
    )

    return evaluate_text(
        reference,
        hypothesis,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PaddleOCR against "
            "canonical ground truth."
        )
    )

    parser.add_argument(
        "--source",
        default="primary_source_to_ocr_sample",
        help=(
            "Source name without .pdf "
            "(default: primary_source_to_ocr_sample)"
        ),
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--page",
        type=int,
        help="Evaluate one page",
    )

    group.add_argument(
        "--pilot",
        action="store_true",
        help="Evaluate pilot pages 2, 4, and 6",
    )

    args = parser.parse_args()

    if args.page is not None:
        pages = [
            args.page
        ]
    else:
        pages = DEFAULT_PILOT_PAGES

    results = []

    print()
    print("=== PADDLE OCR EVALUATION ===")
    print()
    print(f"Source: {args.source}")
    print()

    for page_number in pages:

        result = evaluate_page(
            args.source,
            page_number,
        )

        results.append(
            result
        )

        print(
            f"Page {page_number:04d}: "
            f"CER {result['cer']:.4%} | "
            f"WER {result['wer']:.4%}"
        )

    # ---------------------------------
    # Combined pilot
    # ---------------------------------

    if len(results) > 1:

        combined = evaluate_combined(
            results
        )

        print()
        print("--- Combined Pilot ---")
        print()

        print(
            f"CER: {combined['cer']:.4%}"
        )

        print(
            f"WER: {combined['wer']:.4%}"
        )


if __name__ == "__main__":
    main()
