import json
from pathlib import Path


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

    total = len(regions)

    correct = sum(
        r["annotation_status"] == "correct"
        for r in regions
    )

    illegible = sum(
        r["annotation_status"] == "illegible"
        for r in regions
    )

    unreviewed = sum(
        r["annotation_status"] == "unreviewed"
        for r in regions
    )

    print(f"Total regions:     {total}")
    print(f"Reviewed:          {correct}")
    print(f"Illegible:         {illegible}")
    print(f"Unreviewed:        {unreviewed}")

    if unreviewed == 0:
        print("\nGT review complete.")
    else:
        print("\nWARNING: GT is incomplete.")


if __name__ == "__main__":
    main()
