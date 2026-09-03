import json
from pathlib import Path

from src.evaluate import evaluate_regions


GT_PATH = Path(
    "data/ground_truth/"
    "primary_source_to_ocr_sample/"
    "page_0002/"
    "regions.json"
)


def main():

    with GT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        regions = json.load(file)

    result = evaluate_regions(regions)

    print()
    print("=== PaddleOCR Evaluation ===")
    print()
    print(
        f"Regions evaluated: "
        f"{result['evaluated_regions']}"
    )

    print(
        f"CER: {result['cer']:.4%}"
    )

    print(
        f"WER: {result['wer']:.4%}"
    )


if __name__ == "__main__":
    main()
