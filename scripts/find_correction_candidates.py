from src.correction_candidates import (
    build_frequency_map,
    find_lexical_neighbor,
    looks_suspicious,
)
from src.database import get_connection


def main():

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
            """
            WITH lexical_tokens AS (
                SELECT
                    t.token_id,
                    t.raw_token,
                    t.token_type,
                    t.position_in_page,
                    r.confidence,
                    p.page_id,
                    p.source_page_number,

                    LAG(t.raw_token, 2) OVER (
                        PARTITION BY p.page_id
                        ORDER BY t.position_in_page
                    ) AS before_2,

                    LAG(t.raw_token, 1) OVER (
                        PARTITION BY p.page_id
                        ORDER BY t.position_in_page
                    ) AS before_1,

                    LEAD(t.raw_token, 1) OVER (
                        PARTITION BY p.page_id
                        ORDER BY t.position_in_page
                    ) AS after_1,

                    LEAD(t.raw_token, 2) OVER (
                        PARTITION BY p.page_id
                        ORDER BY t.position_in_page
                    ) AS after_2

                    FROM token t
                    JOIN ocr_region r
                        ON t.region_id = r.region_id
                    JOIN ocr_run o
                        ON r.ocr_run_id = o.ocr_run_id
                    JOIN page p
                        ON o.page_id = p.page_id

                    WHERE t.token_type = 'word'
                )

                SELECT
                    token_id,
                    raw_token,
                    confidence,
                    source_page_number,
                    position_in_page,
                    before_2,
                    before_1,
                    after_1,
                    after_2
                FROM lexical_tokens
                ORDER BY token_id;
                """
            )

            rows = cursor.fetchall()

    frequencies = build_frequency_map(
        [row[1] for row in rows]
    )

    candidates = []

    for (
        token_id,
        raw_token,
        confidence,
        page_number,
        position,
        before_2,
        before_1,
        after_1,
        after_2,
    ) in rows:

        frequency = frequencies[
            raw_token.casefold()
        ]

        # Rare words deserve inspection, but rarity
        # alone is not enough.
        base_reasons = looks_suspicious(
            raw_token
        )

        if (
            base_reasons == ["alphanumeric"]
            and frequency >= 3
        ):
            base_reasons = []

        neighbor = None

        if frequency <= 3:
            neighbor = find_lexical_neighbor(
                raw_token,
                frequencies,
            )

        # A token becomes a candidate if either:
        #
        # 1. its character pattern looks suspicious, or
        # 2. it closely resembles a more common corpus token.

        if not base_reasons and neighbor is None:
            continue

        reasons = list(base_reasons)

        if neighbor is not None:
            reasons.append("lexical_neighbor")

        if frequency <= 2:
            reasons.append("rare")
            
        if (
            not reasons
        ):
            continue
            
        candidates.append(
            {
                "token_id": token_id,
                "token": raw_token,
                "frequency": frequency,
                "confidence": confidence,
                "page": page_number,
                "position": position,
                "reasons": reasons,
                "neighbor": neighbor,
                
                "context": {
                    "before_2": before_2,
                    "before_1": before_1,
                    "after_1": after_1,
                    "after_2": after_2,
                },
            }
        )
        

    # Most interesting first:
    # multiple signals, then lower confidence.
    candidates.sort(
        key=lambda item: (
            -len(item["reasons"]),
            item["confidence"],
        )
    )

    print()
    print("=== CORRECTION CANDIDATES ===")
    print()

    for item in candidates[:100]:

        reasons = ", ".join(
            item["reasons"]
        )
        
        neighbor = item["neighbor"]

        if neighbor is not None:
            print(
                f"  Possible: "
                f"{neighbor['candidate']!r} "
                f"(similarity="
                f"{neighbor['score']:.1f}, "
                f"freq="
                f"{neighbor['candidate_frequency']})"
            )

        context = item["context"]

        before = " ".join(
            word
            for word in [
                context["before_2"],
                context["before_1"],
            ]
            if word is not None
        )

        after = " ".join(
            word
            for word in [
                context["after_1"],
                context["after_2"],
            ]
            if word is not None
        )

        print(
            f"ID {item['token_id']} "
            f"| Page {item['page']:04d}"
        )

        print(
            f"  Context: "
            f"{before} "
            f">>> {item['token']} <<< "
            f"{after}"
        )

        print(
            f"  Frequency: "
            f"{item['frequency']}"
        )

        print(
            f"  Confidence: "
            f"{item['confidence']:.4f}"
        )

        print(
            f"  Reasons: {reasons}"
        )

        print()


if __name__ == "__main__":
    main()
