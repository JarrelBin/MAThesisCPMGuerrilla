import json
from pathlib import Path


def create_gt_template(
    ocr_path: Path,
    output_path: Path,
) -> None:

    with ocr_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        regions = json.load(file)

    gt_regions = []

    for region in regions:

        gt_region = {
            "region_index": region["region_index"],

            "ocr_text": region["raw_text"],

            "gt_text": None,

            "annotation_status": "unreviewed",

            "confidence": region["confidence"],

            "bbox": region["bbox"],
            "polygon": region["polygon"],
        }

        gt_regions.append(gt_region)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            gt_regions,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():

    ocr_path = Path(
        "data/ocr/paddle/"
        "primary_source_to_ocr_sample/"
        "page_0002.json"
    )

    output_path = Path(
        "data/ground_truth/"
        "primary_source_to_ocr_sample/"
        "page_0002/"
        "regions.json"
    )

    create_gt_template(
        ocr_path=ocr_path,
        output_path=output_path,
    )

    print(
        f"GT template created: {output_path}"
    )


if __name__ == "__main__":
    main()
