import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


IMAGE_PATH = Path(
    "data/images/"
    "primary_source_to_ocr_sample/"
    "page_0002.png"
)

GT_PATH = Path(
    "data/ground_truth/"
    "primary_source_to_ocr_sample/"
    "page_0002/"
    "regions.json"
)


def load_regions():
    with GT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_regions(regions):
    with GT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            regions,
            file,
            ensure_ascii=False,
            indent=2,
        )


def show_region(image, region, padding=200):
    """
    Display the OCR region with surrounding context.

    Draw the bounding box using the original PaddleOCR
    coordinates first, then crop the annotated image.
    """

    bbox = region["bbox"]

    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    # Context boundaries in ORIGINAL image coordinates
    left = max(0, x_min - padding)
    top = max(0, y_min - padding)

    right = min(
        image.width,
        x_max + padding,
    )

    bottom = min(
        image.height,
        y_max + padding,
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    # Show the full original image
    ax.imshow(image)

    # Draw Paddle's bbox directly in original coordinates
    rectangle = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )

    ax.add_patch(rectangle)

    # Instead of cropping the PIL image,
    # zoom matplotlib into the context area.
    ax.set_xlim(left, right)

    # Image coordinates have y=0 at the top.
    ax.set_ylim(bottom, top)

    ax.set_title(
        f"Region {region['region_index']} "
        f"| OCR: {region['ocr_text']}"
    )

    ax.axis("off")

    plt.tight_layout()
    plt.show(block=False)

    return fig

def main():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Page image not found: {IMAGE_PATH}"
        )

    if not GT_PATH.exists():
        raise FileNotFoundError(
            f"GT file not found: {GT_PATH}"
        )

    regions = load_regions()

    with Image.open(IMAGE_PATH) as image:

        for region in regions:

            if (
                region["annotation_status"]
                != "unreviewed"
            ):
                continue

            # -------------------------
            # Show source context
            # -------------------------

            fig = show_region(
                image,
                region,
                padding=200,
            )

            print()
            print("=" * 70)

            print(
                f"Region "
                f"{region['region_index']} "
                f"/ {len(regions)}"
            )

            print()
            print("OCR:")
            print(region["ocr_text"])

            print()
            print(
                "Confidence:",
                f"{region['confidence']:.6f}",
            )

            print()
            print(
                "Transcribe the text inside "
                "the highlighted box."
            )

            print()
            print(
                "Commands: "
                ":skip  "
                ":illegible  "
                ":quit"
            )

            transcription = input(
                "\nGT > "
            )

            # -------------------------
            # Close image automatically
            # -------------------------

            plt.close(fig)

            # -------------------------
            # Commands
            # -------------------------

            if transcription == ":quit":

                save_regions(regions)

                print(
                    "Progress saved."
                )

                return

            if transcription == ":skip":
                continue

            if transcription == ":illegible":

                region["gt_text"] = None

                region[
                    "annotation_status"
                ] = "illegible"

                save_regions(regions)

                continue

            # -------------------------
            # Normal GT
            # -------------------------

            region["gt_text"] = transcription

            region[
                "annotation_status"
            ] = "correct"

            save_regions(regions)

            print("Saved.")

    print()
    print("All regions reviewed.")


if __name__ == "__main__":
    main()
