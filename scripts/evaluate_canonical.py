import argparse
from pathlib import Path

from src.evaluate import evaluate_text


DOCUMENT = "primary_source_to_ocr_sample"

PILOT_PAGES = [2, 4, 6]


def load_page(
    page: int,
    engine: str,
):
    """
    Load canonical GT and OCR output for one page.

    If a layout-normalized _eval.txt file exists,
    use it instead of the raw OCR text.
    """

    page_name = f"page_{page:04d}"

    # -------------------------
    # Ground truth
    # -------------------------

    gt_path = Path(
        f"data/ground_truth/canonical/"
        f"{DOCUMENT}/{page_name}.txt"
    )

    # -------------------------
    # OCR output
    # -------------------------

    ocr_dir = Path(
        f"data/ocr/{engine}/{DOCUMENT}"
    )

    eval_path = (
        ocr_dir / f"{page_name}_eval.txt"
    )

    raw_path = (
        ocr_dir / f"{page_name}.txt"
    )

    if eval_path.exists():
        ocr_path = eval_path
        output_type = "layout-normalized"
    else:
        ocr_path = raw_path
        output_type = "raw"

    # -------------------------
    # Validate
    # -------------------------

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Canonical GT not found: {gt_path}"
        )

    if not ocr_path.exists():
        raise FileNotFoundError(
            f"OCR output not found: {ocr_path}"
        )

    reference = gt_path.read_text(
        encoding="utf-8"
    )

    hypothesis = ocr_path.read_text(
        encoding="utf-8"
    )

    return (
        reference,
        hypothesis,
        output_type,
    )


def evaluate_single_page(
    page: int,
    engine: str,
):
    reference, hypothesis, output_type = (
        load_page(
            page=page,
            engine=engine,
        )
    )

    result = evaluate_text(
        reference,
        hypothesis,
    )

    print()
    print("=== Canonical GT Evaluation ===")
    print()

    print(f"Engine:      {engine}")
    print(f"Page:        {page:04d}")
    print(f"OCR input:   {output_type}")

    print()

    print(
        f"CER: {result['cer']:.4%}"
    )

    print(
        f"WER: {result['wer']:.4%}"
    )


def evaluate_pilot(
    engine: str,
):
    references = []
    hypotheses = []

    print()
    print("=== Pilot Evaluation ===")
    print()

    print(f"Engine: {engine}")
    print()

    # -------------------------
    # Individual pages
    # -------------------------

    for page in PILOT_PAGES:

        reference, hypothesis, output_type = (
            load_page(
                page=page,
                engine=engine,
            )
        )

        result = evaluate_text(
            reference,
            hypothesis,
        )

        print(
            f"Page {page:04d}: "
            f"CER {result['cer']:.4%} | "
            f"WER {result['wer']:.4%} | "
            f"{output_type}"
        )

        # evaluate_text returns normalized
        # reference and hypothesis.
        references.append(
            result["reference"]
        )

        hypotheses.append(
            result["hypothesis"]
        )

    # -------------------------
    # Combined pilot
    # -------------------------

    combined_reference = " ".join(
        references
    )

    combined_hypothesis = " ".join(
        hypotheses
    )

    combined_result = evaluate_text(
        combined_reference,
        combined_hypothesis,
    )

    print()
    print("--- Combined Pilot ---")
    print()

    print(
        f"CER: "
        f"{combined_result['cer']:.4%}"
    )

    print(
        f"WER: "
        f"{combined_result['wer']:.4%}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OCR output against "
            "canonical ground truth."
        )
    )

    # -------------------------
    # Optional page
    # -------------------------

    parser.add_argument(
        "page",
        type=int,
        nargs="?",
        help=(
            "Page number to evaluate, "
            "e.g. 2, 4, or 6"
        ),
    )

    # -------------------------
    # Engine
    # -------------------------

    parser.add_argument(
        "--engine",
        default="paddle",
        choices=[
            "paddle",
            "kraken",
        ],
        help=(
            "OCR engine to evaluate "
            "(default: paddle)"
        ),
    )

    # -------------------------
    # Pilot mode
    # -------------------------

    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Evaluate all pilot pages "
            "(2, 4, and 6)"
        ),
    )

    args = parser.parse_args()

    # -------------------------
    # Decide evaluation mode
    # -------------------------

    if args.pilot:

        evaluate_pilot(
            engine=args.engine
        )

        return

    if args.page is None:

        parser.error(
            "Provide a page number "
            "or use --pilot."
        )

    evaluate_single_page(
        page=args.page,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
