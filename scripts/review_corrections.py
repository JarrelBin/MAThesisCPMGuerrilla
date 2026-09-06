from src.paths import (
    from_project_relative,
)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.correction_candidates import (
    get_correction_candidates,
)
from src.database import get_connection
from src.repository import (
    insert_token_correction,
    mark_token_reviewed,
)


CONTEXT_PADDING_X = 350
CONTEXT_PADDING_Y = 100


def build_context(candidate):
    context = candidate["context"]

    before = " ".join(
        word
        for word in [
            context["before_2"],
            context["before_1"],
        ]
        if word
    )

    after = " ".join(
        word
        for word in [
            context["after_1"],
            context["after_2"],
        ]
        if word
    )

    return (
        f"{before} "
        f">>> {candidate['token']} <<< "
        f"{after}"
    ).strip()


def show_source_crop(candidate):
    """
    Display source-image context and highlight the
    Paddle component containing the token.
    """

    image_path = from_project_relative(
        candidate["image_path"]
    )

    if not image_path.exists():
        print(
            f"WARNING: image not found: "
            f"{image_path}"
        )
        return None

    bbox = candidate["bbox"]

    if any(
        value is None
        for value in bbox.values()
    ):
        print(
            "WARNING: token has no bounding box."
        )
        return None

    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    image = Image.open(
        image_path
    )

    left = max(
        0,
        x_min - CONTEXT_PADDING_X,
    )

    right = min(
        image.width,
        x_max + CONTEXT_PADDING_X,
    )

    top = max(
        0,
        y_min - CONTEXT_PADDING_Y,
    )

    bottom = min(
        image.height,
        y_max + CONTEXT_PADDING_Y,
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.imshow(image)

    rectangle = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )

    ax.add_patch(
        rectangle
    )

    # Zoom while retaining original coordinates.
    ax.set_xlim(
        left,
        right,
    )

    ax.set_ylim(
        bottom,
        top,
    )

    ax.set_title(
        f"Page {candidate['page']:04d} "
        f"| Token ID {candidate['token_id']} "
        f"| OCR: {candidate['token']}"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.show(
        block=False
    )

    return fig, image


def close_source_crop(
    display,
):
    if display is None:
        return

    fig, image = display

    plt.close(fig)
    image.close()


def print_candidate(
    candidate,
    number,
    total,
):
    print()
    print("=" * 72)
    print()

    print(
        f"Candidate {number} / {total}"
    )

    print(
        f"Token ID: {candidate['token_id']}"
    )

    print(
        f"Page:     {candidate['page']:04d}"
    )

    print()

    print("Context:")
    print(
        build_context(
            candidate
        )
    )

    print()

    print(
        f"Raw OCR:     "
        f"{candidate['token']}"
    )

    neighbor = candidate[
        "neighbor"
    ]

    if neighbor is not None:

        print(
            f"Suggested:   "
            f"{neighbor['candidate']}"
        )

        print(
            f"Similarity:  "
            f"{neighbor['score']:.1f}"
        )

        print(
            f"Suggestion frequency: "
            f"{neighbor['candidate_frequency']}"
        )

    else:

        print(
            "Suggested:   none"
        )

    print()

    print(
        f"OCR confidence: "
        f"{candidate['confidence']:.4f}"
    )

    print(
        f"Raw frequency:  "
        f"{candidate['frequency']}"
    )

    print(
        "Reasons:        "
        + ", ".join(
            candidate["reasons"]
        )
    )

    print()

def save_correction(
    candidate,
    corrected,
    notes,
):
    with get_connection() as connection:

        insert_token_correction(
            connection,
            token_id=candidate["token_id"],
            corrected_token=corrected,
            method="human",
            confidence=None,
            notes=notes,
        )

        mark_token_reviewed(
            connection,
            token_id=candidate["token_id"],
        )

        connection.commit()

def save_no_correction(
    candidate,
):
    with get_connection() as connection:

        mark_token_reviewed(
            connection,
            token_id=candidate["token_id"],
        )

        connection.commit()


def main():

    with get_connection() as connection:

        candidates = (
            get_correction_candidates(
                connection
            )
        )

    if not candidates:
        print(
            "No unreviewed correction "
            "candidates found."
        )
        return

    total = len(candidates)

    for number, candidate in enumerate(
        candidates,
        start=1,
    ):

        print_candidate(
            candidate,
            number,
            total,
        )

        display = show_source_crop(
            candidate
        )

        neighbor = candidate[
            "neighbor"
        ]

        if neighbor is not None:

            print(
                "[y] accept suggestion"
            )

        print(
            "[e] enter different correction"
        )

        print(
            "[n] OCR is correct"
        )

        print(
            "[s] skip for now"
        )

        print(
            "[q] save and quit"
        )

        print()

        choice = input(
            "> "
        ).strip().casefold()

        should_continue = True

        match choice:

            # ---------------------------------
            # Quit
            # ---------------------------------

            case "q":
                print(
                    "Exiting correction review."
                )

                should_continue = False

            # ---------------------------------
            # Skip
            # ---------------------------------

            case "s":
                pass

            # ---------------------------------
            # OCR is already correct
            # ---------------------------------

            case "n":

                with get_connection() as connection:

                    mark_token_reviewed(
                        connection,
                        token_id=candidate["token_id"],
                    )

                    connection.commit()

                print(
                    "Marked as reviewed "
                    "(no correction)."
                )

            # ---------------------------------
            # Accept suggestion
            # ---------------------------------

            case "y" if neighbor is not None:

                corrected = neighbor["candidate"]

                with get_connection() as connection:

                    insert_token_correction(
                        connection,
                        token_id=candidate["token_id"],
                        corrected_token=corrected,
                        method="human",
                        confidence=None,
                        notes=(
                            "Accepted lexical-neighbour "
                            "suggestion during review."
                        ),
                    )

                    mark_token_reviewed(
                        connection,
                        token_id=candidate["token_id"],
                    )

                    connection.commit()

                print(
                    f"Correction saved: "
                    f"{candidate['token']} "
                    f"-> {corrected}"
                )

            # ---------------------------------
            # Manual correction
            # ---------------------------------

            case "e":

                # Image stays open while this input
                # is being entered.
                corrected = input(
                    "Corrected token > "
                ).strip()

                if corrected:

                    with get_connection() as connection:

                        insert_token_correction(
                            connection,
                            token_id=candidate["token_id"],
                            corrected_token=corrected,
                            method="human",
                            confidence=None,
                            notes=(
                                "Manually entered during "
                                "correction review."
                            ),
                        )

                        mark_token_reviewed(
                            connection,
                            token_id=candidate["token_id"],
                        )

                        connection.commit()

                    print(
                        f"Correction saved: "
                        f"{candidate['token']} "
                        f"-> {corrected}"
                    )

                else:
                    print(
                        "Empty correction ignored."
                    )

            # ---------------------------------
            # y without suggestion
            # ---------------------------------

            case "y":
                print(
                    "No suggestion is available."
                )

            # ---------------------------------
            # Invalid command
            # ---------------------------------

            case _:
                print(
                    "Unknown command; candidate "
                    "left unreviewed."
                )


        # ---------------------------------
        # One common cleanup point
        # ---------------------------------

        close_source_crop(
            display
        )

        if not should_continue:
            return

    print()
    print(
        "All current correction "
        "candidates reviewed."
    )

if __name__ == "__main__":
    main()
