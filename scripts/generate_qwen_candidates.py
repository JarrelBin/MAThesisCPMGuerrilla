import argparse
import time

from src.database import get_connection
from src.qwen_correction import (
    DEFAULT_MODEL,
    QwenCorrectionModel,
)
from src.repository import (
    delete_correction_candidates_by_method,
    get_tokens_for_contextual_review,
    upsert_correction_candidate,
)


# ============================================================
# Context-window configuration
# ============================================================

WINDOW_SIZE = 60
WINDOW_OVERLAP = 5
CONTEXT_RADIUS = 15


# ============================================================
# Group database tokens by physical page
# ============================================================

def group_by_page(
    rows,
) -> list[dict]:

    pages = {}

    for (
        token_id,
        raw_token,
        token_type,
        page_id,
        source_file_id,
        source_page_number,
        corpus_page_number,
        position_in_page,
    ) in rows:

        if page_id not in pages:

            pages[page_id] = {
                "page_id":
                    page_id,

                "source_file_id":
                    source_file_id,

                "source_page_number":
                    source_page_number,

                "corpus_page_number":
                    corpus_page_number,

                "tokens": [],
            }

        pages[
            page_id
        ]["tokens"].append(
            {
                "token_id":
                    token_id,

                "raw_token":
                    raw_token,

                "token_type":
                    token_type,

                "position_in_page":
                    position_in_page,
            }
        )

    # Preserve physical corpus order.
    return sorted(
        pages.values(),
        key=lambda page: (
            page["corpus_page_number"]
        ),
    )


# ============================================================
# Sliding windows
# ============================================================


def make_windows(
    tokens: list[dict],
):
    """
    Generate overlapping token windows.

    Example with:

        WINDOW_SIZE = 30
        WINDOW_OVERLAP = 5

    windows:

        1-30
        26-55
        51-80
        ...
    """

    if WINDOW_OVERLAP >= WINDOW_SIZE:
        raise ValueError(
            "WINDOW_OVERLAP must be smaller "
            "than WINDOW_SIZE."
        )

    step = (
        WINDOW_SIZE
        - WINDOW_OVERLAP
    )

    for start in range(
        0,
        len(tokens),
        step,
    ):

        window = tokens[
            start:
            start + WINDOW_SIZE
        ]

        if not window:
            break

        yield window

def make_candidate_context(
    tokens: list[dict],
    target_index: int,
) -> list[dict]:

    start = max(
        0,
        target_index - CONTEXT_RADIUS,
    )

    end = min(
        len(tokens),
        target_index + CONTEXT_RADIUS + 1,
    )

    return tokens[start:end]


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Generate contextual OCR correction "
            "candidates using local Qwen via "
            "Hugging Face Transformers."
        )
    )

    parser.add_argument(
        "--source-file",
        type=int,
        help=(
            "Restrict analysis to one "
            "source_file_id."
        ),
    )

    parser.add_argument(
        "--page",
        type=int,
        help=(
            "Restrict analysis to one source "
            "page. Normally used together with "
            "--source-file."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face Qwen model name "
            f"(default: {DEFAULT_MODEL})."
        ),
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        help=(
            "Process only the first N windows "
            "per page. Useful for testing."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if (
        args.page is not None
        and args.source_file is None
    ):
        parser.error(
            "--page requires --source-file"
        )

    if (
        args.max_windows is not None
        and args.max_windows < 1
    ):
        parser.error(
            "--max-windows must be >= 1"
        )

    # ========================================================
    # Retrieve eligible OCR tokens
    # ========================================================

    with get_connection() as connection:

        rows = (
            get_tokens_for_contextual_review(
                connection,
                source_file_id=(
                    args.source_file
                ),
            )
        )

    pages = group_by_page(
        rows
    )

    # --------------------------------------------------------
    # Optional source-page restriction
    # --------------------------------------------------------

    if args.page is not None:

        pages = [
            page
            for page in pages
            if (
                page[
                    "source_page_number"
                ]
                == args.page
            )
        ]

    if not pages:

        print(
            "No eligible tokens found."
        )

        return

    # ========================================================
    # Summary before expensive model loading
    # ========================================================

    print()
    print(
        "=== QWEN CORRECTION "
        "CANDIDATE GENERATION ==="
    )
    print()

    print(
        f"Model:          "
        f"{args.model}"
    )

    if args.source_file is not None:

        print(
            f"Source file ID: "
            f"{args.source_file}"
        )

    if args.page is not None:

        print(
            f"Source page:    "
            f"{args.page:04d}"
        )

    print(
        f"Pages:          "
        f"{len(pages)}"
    )

    print(
        f"Window size:    "
        f"{WINDOW_SIZE}"
    )

    print(
        f"Window overlap: "
        f"{WINDOW_OVERLAP}"
    )

    if args.max_windows is not None:

        print(
            f"Max windows:    "
            f"{args.max_windows} per page"
        )

    print()

    # ========================================================
    # Candidate cleanup
    # ========================================================
    #
    # Only clear ALL Qwen candidates for a complete corpus run.
    #
    # A --source-file or --page test should not erase Qwen
    # results generated elsewhere.
    # ========================================================

    if (
        args.source_file is None
        and args.page is None
    ):

        with get_connection() as connection:

            delete_correction_candidates_by_method(
                connection,
                method="qwen",
            )

            connection.commit()

        print(
            "Cleared previous corpus-wide "
            "Qwen candidates."
        )

        print()

    # ========================================================
    # Load Qwen ONCE
    # ========================================================

    qwen = QwenCorrectionModel(
        model_name=args.model
    )

    print()

    # ========================================================
    # Process pages
    # ========================================================

    stored_candidate_ids = set()

    total_model_candidates = 0
    total_stored_candidates = 0
    total_rejected_candidates = 0
    total_failed_windows = 0
    total_windows = 0

    for page_number, page in enumerate(
        pages,
        start=1,
    ):

        source_page_number = (
            page["source_page_number"]
        )

        print(
            f"Page "
            f"{source_page_number:04d} "
            f"({page_number}/{len(pages)})"
        )

        # ----------------------------------------------------
        # Valid token IDs for this page
        # ----------------------------------------------------
        #
        # Prevent a malformed or hallucinated model response
        # from pointing to a token outside the current page.
        # ----------------------------------------------------

        valid_tokens = {
            token["token_id"]: token
            for token in page["tokens"]
        }

        # ----------------------------------------------------
        # Generate overlapping windows
        # ----------------------------------------------------

        windows = list(
            make_windows(
                page["tokens"]
            )
        )

        if args.max_windows is not None:

            windows = windows[
                :args.max_windows
            ]

        print(
            f"  Tokens:  "
            f"{len(page['tokens'])}"
        )

        print(
            f"  Windows: "
            f"{len(windows)}"
        )

        # ----------------------------------------------------
        # Qwen inference
        # ----------------------------------------------------

        for window_number, window in enumerate(
            windows,
            start=1,
        ):

            total_windows += 1

            print(
                f"  Window "
                f"{window_number}/"
                f"{len(windows)} "
                f"({len(window)} tokens)",
                flush=True,
            )

            start_time = (
                time.perf_counter()
            )

            try:

                suggestions = (
                    qwen.analyze_token_window(
                        window
                    )
                )

            except Exception as error:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                total_failed_windows += 1

                print(
                    f"    FAILED after "
                    f"{elapsed:.1f}s: "
                    f"{error}",
                    flush=True,
                )

                # Do not destroy results from previous
                # successful windows.
                continue

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"    Completed in "
                f"{elapsed:.1f}s "
                f"| candidates="
                f"{len(suggestions)}",
                flush=True,
            )

            total_model_candidates += (
                len(suggestions)
            )

            # ------------------------------------------------
            # Validate Qwen suggestions
            # ------------------------------------------------

            for suggestion in suggestions:
                if not isinstance(
                    suggestion,
                    dict,
                ):
                    total_rejected_candidates += 1

                    print(
                        "    REJECTED: "
                        "candidate is not an object"
                    )

                    continue

                token_id = suggestion.get(
                    "token_id"
                )

                reason = suggestion.get(
                    "reason"
                )
                
                qwen_raw_token = suggestion.get(
                    "raw_token"
                )
                

                # --------------------------------------------
                # token_id validation
                # --------------------------------------------

                try:

                    token_id = int(
                        token_id
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    total_rejected_candidates += 1

                    print(
                        f"    REJECTED: "
                        f"invalid token_id="
                        f"{token_id!r}"
                    )

                    continue

                if token_id not in valid_tokens:

                    total_rejected_candidates += 1

                    print(
                        f"    REJECTED: "
                        f"token_id={token_id} "
                        f"is outside the current page"
                    )

                    continue

                # Validate the ID before looking up its raw token.
                # Restrict it to the current window, not just the page.
                window_tokens = {
                    token["token_id"]: token
                    for token in window
                }
                if token_id not in window_tokens:
                    total_rejected_candidates += 1
                    print(f"    REJECTED: token_id={token_id} outside current window")
                    continue

                raw_token = window_tokens[token_id]["raw_token"]

                # --------------------------------------------
                # Qwen token-alignment validation
                # --------------------------------------------

                if not isinstance(
                    qwen_raw_token,
                    str,
                ):

                    total_rejected_candidates += 1

                    print(
                        f"    REJECTED: "
                        f"token_id={token_id} "
                        f"has no valid raw_token"
                    )

                    continue


                qwen_raw_token = (
                    qwen_raw_token.strip()
                )


                if qwen_raw_token != raw_token:

                    total_rejected_candidates += 1

                    print(
                        f"    REJECTED: "
                        f"token alignment mismatch"
                    )

                    print(
                        f"      Qwen:     "
                        f"[{token_id}] "
                        f"{qwen_raw_token!r}"
                    )

                    print(
                        f"      Database: "
                        f"[{token_id}] "
                        f"{raw_token!r}"
                    )

                    continue

                # --------------------------------------------
                # Reason validation
                # --------------------------------------------
                #
                # A reason is useful as an audit trail, but it
                # is not required for storing a candidate.
                # --------------------------------------------

                if not isinstance(
                    reason,
                    str,
                ):
                    reason = None

                elif not reason.strip():
                    reason = None

                else:
                    reason = reason.strip()

                # --------------------------------------------
                # Duplicate handling
                # --------------------------------------------
                #
                # Overlapping windows can identify the same
                # token more than once.
                #
                # For now, store only the first accepted
                # proposal for a token during this run.
                #
                # Agreement/disagreement across windows can
                # later be retained as separate evidence.
                # --------------------------------------------

                if (
                    token_id
                    in stored_candidate_ids
                ):

                    total_rejected_candidates += 1

                    print(
                        f"    REJECTED: "
                        f"{raw_token!r} "
                        f"(already flagged this run)"
                    )

                    continue

                # --------------------------------------------
                # Store machine candidate
                # --------------------------------------------
                #
                # Qwen does not provide a calibrated numerical
                # confidence score.
                #
                # Store NULL rather than inventing a confidence
                # value.
                # --------------------------------------------

                with get_connection() as connection:

                    upsert_correction_candidate(
                        connection,
                        token_id=token_id,
                        method="qwen",
                        suggested_token=None,
                        details={
                            "reason":
                                reason,

                            "model":
                                args.model,

                            "source_page_number":
                                source_page_number,

                            "window_size":
                                WINDOW_SIZE,

                            "window_overlap":
                                WINDOW_OVERLAP,

                            "window_number":
                                window_number,
                        },
                    )

                    connection.commit()

                stored_candidate_ids.add(
                    token_id
                )

                total_stored_candidates += 1

                print(
                    f"    FLAGGED: "
                    f"[{token_id}] {raw_token!r} "
                    f"(detection only)"
                )

                if reason:

                    print(
                        f"      Reason: "
                        f"{reason}"
                    )

        print()

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "--- Qwen Summary ---"
    )

    print()

    print(
        f"Pages processed:       "
        f"{len(pages)}"
    )

    print(
        f"Windows processed:     "
        f"{total_windows}"
    )

    print(
        f"Windows failed:        "
        f"{total_failed_windows}"
    )

    print(
        f"Candidates returned:   "
        f"{total_model_candidates}"
    )

    print(
        f"Candidates rejected:   "
        f"{total_rejected_candidates}"
    )

    print(
        f"Candidates stored:     "
        f"{total_stored_candidates}"
    )

if __name__ == "__main__":
    main()
