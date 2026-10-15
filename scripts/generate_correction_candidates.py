from src.correction_candidates import (
    get_correction_candidates,
)
from src.database import get_connection
from src.repository import (
    delete_correction_candidates_by_method,
    upsert_correction_candidate,
)


def main():

    # ---------------------------------------------------------
    # Generate candidates using your existing detector.
    # ---------------------------------------------------------

    with get_connection() as connection:

        candidates = get_correction_candidates(
            connection
        )

    # ---------------------------------------------------------
    # Replace previous generated candidates.
    # ---------------------------------------------------------

    with get_connection() as connection:

        delete_correction_candidates_by_method(
            connection,
            method="pattern",
        )

        delete_correction_candidates_by_method(
            connection,
            method="rapidfuzz",
        )

        pattern_count = 0
        rapidfuzz_count = 0

        for candidate in candidates:

            token_id = candidate[
                "token_id"
            ]

            reasons = candidate[
                "reasons"
            ]

            # ---------------------------------------------
            # Pattern candidate
            # ---------------------------------------------

            pattern_reasons = [
                reason
                for reason in reasons
                if reason not in {
                    "lexical_neighbor",
                    "rare",
                }
            ]

            if pattern_reasons:

                upsert_correction_candidate(
                    connection,
                    token_id=token_id,
                    method="pattern",
                    suggested_token=None,
                    details={
                        "reasons":
                            pattern_reasons,

                        "frequency":
                            candidate[
                                "frequency"
                            ],

                        "ocr_confidence":
                            candidate[
                                "confidence"
                            ],
                    },
                )

                pattern_count += 1

            # ---------------------------------------------
            # RapidFuzz candidate
            # ---------------------------------------------

            neighbor = candidate[
                "neighbor"
            ]

            if neighbor is not None:


                upsert_correction_candidate(
                    connection,
                    token_id=token_id,
                    method="rapidfuzz",
                    suggested_token=(
                        neighbor[
                            "candidate"
                        ]
                    ),
                    details={
                        "raw_frequency":
                            candidate[
                                "frequency"
                            ],

                        "candidate_frequency":
                            neighbor[
                                "candidate_frequency"
                            ],

                        "ocr_confidence":
                            candidate[
                                "confidence"
                            ],
                    },
                )

                rapidfuzz_count += 1

        connection.commit()

    print()
    print(
        "=== CORRECTION CANDIDATES ==="
    )
    print()

    print(
        f"Pattern candidates:   "
        f"{pattern_count}"
    )

    print(
        f"RapidFuzz candidates: "
        f"{rapidfuzz_count}"
    )


if __name__ == "__main__":
    main()
