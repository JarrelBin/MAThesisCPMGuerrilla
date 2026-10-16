from src.paths import (
    from_project_relative,
)

import argparse
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

def load_vision_evidence(candidate):
    """Attach latest vision evidence for each machine candidate."""
    ids = [e["candidate_id"] for e in candidate["candidates"]]
    if not ids:
        return
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT ON (candidate_id)
                    candidate_id, model_name, model_quantization,
                    transcription, decision
                FROM qwen_vision_evidence
                WHERE candidate_id = ANY(%s)
                ORDER BY candidate_id, created_at DESC, evidence_id DESC
                """,
                (ids,),
            )
            records = cursor.fetchall()
    lookup = {
        row[0]: {
            "model": row[1], "quantization": row[2],
            "transcription": row[3], "decision": row[4],
        }
        for row in records
    }
    for evidence in candidate["candidates"]:
        evidence["vision"] = lookup.get(evidence["candidate_id"])


def print_machine_candidates(candidate):
    print("Machine evidence:\n")
    for evidence in candidate["candidates"]:
        print(f"  {evidence['method']}:")
        suggestion = evidence["suggested_token"]
        print(f"    suggestion: {suggestion}" if suggestion is not None else "    detection only")
        details = evidence["details"] or {}
        if isinstance(details, str):
            details = json.loads(details)
        for field in ("model", "reason", "context"):
            if details.get(field):
                print(f"    {field}: {details[field]}")
        vision = evidence.get("vision")
        if vision:
            print(f"    vision: {vision['model']} {vision['quantization']} | {vision['decision']}")
            print(f"    visual transcription: {vision['transcription']}")
        print()


def vision_supports_original(candidate):
    """Conservatively detect vision agreement with the exact OCR spelling.

    A CONTRADICT label alone is insufficient: it might reflect a third
    transcription. Require the exact raw word once and no proposed word.
    If the crop contains multiple instances, leave the choice to the human.
    """
    raw = candidate["token"]
    words_pattern = r"[^\W_]+(?:['’\-][^\W_]+)*"
    for evidence in candidate["candidates"]:
        if evidence["method"] != "qwen" or not evidence["suggested_token"]:
            continue
        vision = evidence.get("vision")
        if not vision or vision["decision"] != "CONTRADICT":
            continue
        words = re.findall(words_pattern, vision["transcription"], flags=re.UNICODE)
        if words.count(raw) == 1 and evidence["suggested_token"] not in words:
            return True
    return False


def choose_review_action(candidate):
    """Return (action, suggested_token) for the y shortcut."""
    if vision_supports_original(candidate):
        return "keep", candidate["token"]
    suggestions = {
        evidence["suggested_token"]
        for evidence in candidate["candidates"]
        if evidence["suggested_token"] is not None
    }
    if len(suggestions) == 1:
        return "correct", next(iter(suggestions))
    return None, None


def vision_target_suggestion(candidate, context_rows):
    """Find a target reading using nearby OCR words, not a global word match.

    Returns None when the visual transcription cannot be aligned uniquely.
    Vision suggestions are only human-review options, never auto-accepted.
    """
    vision_records = [
        evidence.get("vision")
        for evidence in candidate["candidates"]
        if evidence["method"] == "qwen" and evidence.get("vision")
    ]
    if len(vision_records) != 1:
        return None

    vision = vision_records[0]
    transcription = vision.get("transcription") or ""
    word_pattern = r"[^\W_]+(?:['’\-][^\W_]+)*"
    visible = re.findall(word_pattern, transcription, flags=re.UNICODE)
    if not visible:
        return None

    # Token context excludes punctuation; normalize in the same way.
    context = []
    target_index = None
    for token_id, raw, _position in context_rows:
        parts = re.findall(word_pattern, raw, flags=re.UNICODE)
        if token_id == candidate["token_id"]:
            if len(parts) != 1:
                return None
            target_index = len(context)
        context.extend(parts)
    if target_index is None:
        return None

    # Align the target using words immediately before and after it.
    # A unique best match needs at least two nearby anchors, including
    # one on either side, or at least three on one side.
    scored = []
    for index, word in enumerate(visible):
        left = right = 0
        for offset in (-3, -2, -1, 1, 2, 3):
            ci = target_index + offset
            vi = index + offset
            if 0 <= ci < len(context) and 0 <= vi < len(visible):
                if context[ci].casefold() == visible[vi].casefold():
                    if offset < 0:
                        left += 1
                    else:
                        right += 1
        if (left >= 1 and right >= 1) or left >= 3 or right >= 3:
            scored.append((left + right, index, word))

    if not scored:
        return None
    scored.sort(reverse=True)
    best_score = scored[0][0]
    if sum(score == best_score for score, _index, _word in scored) != 1:
        return None

    reading = scored[0][2]
    if reading == candidate["token"]:
        return None  # The existing [n]/[y] keep-original action covers this.
    return reading


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
        "--source-file",
        type=int,
        help="Review candidates from this source_file_id only",
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
    if args.source_file is not None and args.source_file < 1:
        parser.error("--source-file must be positive")
    if args.page is not None and args.page < 1:
        parser.error("--page must be positive")

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

    # Match by permanent page IDs, not by source page numbers, which
    # repeat across different PDFs.
    if args.source_file is not None:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT page_id FROM page WHERE source_file_id = %s",
                    (args.source_file,),
                )
                allowed_page_ids = {row[0] for row in cursor.fetchall()}
        candidates = [c for c in candidates if c["page_id"] in allowed_page_ids]

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

        load_vision_evidence(candidate)
        print_machine_candidates(candidate)

        # -----------------------------------------------------
        # Determine whether machine methods provide one
        # unambiguous correction suggestion.
        # -----------------------------------------------------

        recommended_action, default_suggestion = choose_review_action(candidate)
        vision_suggestion = vision_target_suggestion(candidate, context_rows)
        # [v] is useful only when it offers a distinct alternative.
        if vision_suggestion == default_suggestion:
            vision_suggestion = None

        # -----------------------------------------------------
        # Source image
        # -----------------------------------------------------

        display = show_source_crop(
            candidate
        )

        # =====================================================
        # Commands
        # =====================================================

        if recommended_action == "keep":
            print(f"[y] keep original OCR: {candidate['token']}")
        elif recommended_action == "correct":
            print(f"[y] accept suggestion: {default_suggestion}")

        if vision_suggestion is not None:
            print(f"[v] accept vision reading: {vision_suggestion}")

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

            case "y" if recommended_action == "keep":
                with get_connection() as connection:
                    mark_token_reviewed(connection, token_id=candidate["token_id"])
                    connection.commit()
                print(f"Original OCR preserved: {candidate['token']}")

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
            # Accept the independently transcribed vision reading.
            # The reviewer must still verify it against the scan.
            # -------------------------------------------------

            case "v" if vision_suggestion is not None:
                with get_connection() as connection:
                    insert_token_correction(
                        connection,
                        token_id=candidate["token_id"],
                        corrected_token=vision_suggestion,
                        method="human",
                        confidence=None,
                        notes="Human accepted vision transcription during review.",
                    )
                    mark_token_reviewed(connection, token_id=candidate["token_id"])
                    connection.commit()
                print(f"Vision reading saved: {candidate['token']} -> {vision_suggestion}")

            case "v":
                print("No uniquely aligned vision reading is available; use [e] if needed.")

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
