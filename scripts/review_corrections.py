from src.paths import (
    from_project_relative,
)

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.correction_candidates import (
    get_correction_candidates,
)
from src.database import get_connection
from src.repository import (
    get_token_context,
    get_tokens_with_correction_candidates,
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

def group_candidates(
    rows,
):
    """
    Convert database candidate rows into one review
    object per token.
    """

    tokens = {}

    for (
            token_id,
            raw_token,
            position_in_page,

            x_min,
            y_min,
            x_max,
            y_max,

            page_id,
            source_page_number,
            image_path,

            ocr_confidence,

            candidate_id,
            method,
            suggested_token,
            details,
    ) in rows:

        if token_id not in tokens:

            tokens[token_id] = {
                "token_id":
                    token_id,

                "token":
                    raw_token,

                "position_in_page":
                    position_in_page,

                "page_id":
                    page_id,

                "page":
                    source_page_number,

                "image_path":
                    image_path,

                "ocr_confidence":
                    ocr_confidence,
                
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },

                "candidates": [],
            }

        tokens[
            token_id
        ]["candidates"].append(
            {
                "candidate_id":
                    candidate_id,

                "method":
                    method,

                "suggested_token":
                    suggested_token,

                "details":
                    details,
            }
        )

    return list(
        tokens.values()
    )

def print_machine_candidates(candidate):

    print("Machine evidence:")
    print()

    for evidence in candidate["candidates"]:

        method = evidence["method"]
        suggestion = evidence["suggested_token"]

        print(f"  {method}:")

        if suggestion is not None:
            print(f"    suggestion: {suggestion}")
        else:
            print("    detection only")

        details = evidence["details"] or {}

        if isinstance(details, str):
            import json
            details = json.loads(details)

        model = details.get("model")
        reason = details.get("reason")
        context = details.get("context")

        if model:
            print(f"    model: {model}")

        if reason:
            print(f"    reason: {reason}")

        if context:
            print(f"    context: {context}")

        print()

def choose_default_suggestion(
    candidate,
):
    """
    Determine whether there is an unambiguous machine
    suggestion that can be offered as the [y] action.

    If methods disagree, require manual entry instead.
    """

    suggestions = {
        evidence["suggested_token"]
        for evidence in candidate["candidates"]
        if evidence["suggested_token"]
        is not None
    }

    if len(suggestions) == 1:
        return next(
            iter(suggestions)
        )

    return None

def format_context(
    context_rows,
    target_token_id,
):
    parts = []

    for (
        token_id,
        token,
        position,
    ) in context_rows:

        if token_id == target_token_id:

            parts.append(
                f">>> {token} <<<"
            )

        else:

            parts.append(
                token
            )

    return " ".join(
        parts
    )

def main():
    parser = argparse.ArgumentParser(
        description="Review OCR correction candidates."
    )

    parser.add_argument(
        "--page",
        type=int,
        help=(
            "Review candidates from one "
            "source page only"
        ),
    )

    args = parser.parse_args()

    # =========================================================
    # Load stored machine candidates
    # =========================================================

    with get_connection() as connection:

        rows = (
            get_tokens_with_correction_candidates(
                connection
            )
        )

    candidates = group_candidates(
        rows
    )

    # =========================================================
    # Optional page filter
    # =========================================================

    if args.page is not None:

        candidates = [
            candidate
            for candidate in candidates
            if candidate["page"] == args.page
        ]

    if not candidates:
        print(
            "No unreviewed correction "
            "candidates found."
        )
        return

    total = len(
        candidates
    )

    # =========================================================
    # Review loop
    # =========================================================

    for number, candidate in enumerate(
        candidates,
        start=1,
    ):

        # -----------------------------------------------------
        # Context
        # -----------------------------------------------------

        with get_connection() as connection:

            context_rows = get_token_context(
                connection,
                token_id=candidate["token_id"],
                window=5,
            )

        context = format_context(
            context_rows,
            candidate["token_id"],
        )

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        print()
        print("=" * 72)
        print()

        print(
            f"Candidate {number} / {total}"
        )

        print(
            f"Token ID: "
            f"{candidate['token_id']}"
        )

        print(
            f"Page:     "
            f"{candidate['page']:04d}"
        )

        print()

        print("Context:")
        print(
            context
        )

        print()

        print(
            f"Raw OCR:        "
            f"{candidate['token']}"
        )

        print(
            f"OCR confidence: "
            f"{candidate['ocr_confidence']:.4f}"
        )

        print()

        # -----------------------------------------------------
        # Machine evidence
        # -----------------------------------------------------

        print_machine_candidates(
            candidate
        )

        # -----------------------------------------------------
        # Determine whether machine methods provide one
        # unambiguous correction suggestion.
        # -----------------------------------------------------

        default_suggestion = (
            choose_default_suggestion(
                candidate
            )
        )

        # -----------------------------------------------------
        # Source image
        # -----------------------------------------------------

        display = show_source_crop(
            candidate
        )

        # =====================================================
        # Commands
        # =====================================================

        if default_suggestion is not None:

            print(
                f"[y] accept suggestion: "
                f"{default_suggestion}"
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

        # =====================================================
        # Decision
        # =====================================================

        match choice:

            # -------------------------------------------------
            # Quit
            # -------------------------------------------------

            case "q":

                print(
                    "Exiting correction review."
                )

                should_continue = False

            # -------------------------------------------------
            # Skip
            #
            # Nothing is written to PostgreSQL.
            # Candidate will appear again next time.
            # -------------------------------------------------

            case "s":

                pass

            # -------------------------------------------------
            # OCR is already correct
            # -------------------------------------------------

            case "n":

                with get_connection() as connection:

                    mark_token_reviewed(
                        connection,
                        token_id=(
                            candidate[
                                "token_id"
                            ]
                        ),
                    )

                    connection.commit()

                print(
                    "Marked as reviewed "
                    "(no correction)."
                )

            # -------------------------------------------------
            # Accept machine suggestion
            # -------------------------------------------------

            case "y" if (
                default_suggestion
                is not None
            ):

                corrected = (
                    default_suggestion
                )

                with get_connection() as connection:

                    insert_token_correction(
                        connection,
                        token_id=(
                            candidate[
                                "token_id"
                            ]
                        ),
                        corrected_token=corrected,
                        method="human",
                        confidence=None,
                        notes=(
                            "Accepted machine-generated "
                            "suggestion during review."
                        ),
                    )

                    mark_token_reviewed(
                        connection,
                        token_id=(
                            candidate[
                                "token_id"
                            ]
                        ),
                    )

                    connection.commit()

                print(
                    f"Correction saved: "
                    f"{candidate['token']} "
                    f"-> {corrected}"
                )

            # -------------------------------------------------
            # Manual correction
            # -------------------------------------------------

            case "e":

                # Image remains open while you inspect it
                # and type the correction.
                corrected = input(
                    "Corrected token > "
                ).strip()

                if corrected:

                    with get_connection() as connection:

                        insert_token_correction(
                            connection,
                            token_id=(
                                candidate[
                                    "token_id"
                                ]
                            ),
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
                            token_id=(
                                candidate[
                                    "token_id"
                                ]
                            ),
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

            # -------------------------------------------------
            # y when machine methods do not provide one
            # unambiguous suggestion
            # -------------------------------------------------

            case "y":

                print(
                    "No single machine suggestion "
                    "is available."
                )

            # -------------------------------------------------
            # Invalid command
            # -------------------------------------------------

            case _:

                print(
                    "Unknown command; candidate "
                    "left unreviewed."
                )

        # =====================================================
        # Common cleanup
        # =====================================================

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
