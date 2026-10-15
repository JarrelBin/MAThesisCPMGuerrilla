import re
from collections import Counter
from rapidfuzz import fuzz, process


def find_lexical_neighbor(
    token: str,
    frequencies,
    *,
    minimum_frequency: int = 3,
):
    """
    Find a plausible, more frequent corpus token.

    Conservative rules are used because short strings,
    names, and abbreviations otherwise generate many
    meaningless fuzzy matches.
    """

    normalized = token.casefold()

    # ---------------------------------
    # Don't fuzzy-correct very short
    # lexical forms.
    #
    # Examples:
    # CT, OC, WA, MAT, KOK
    # ---------------------------------

    if len(normalized) < 4:
        return None

    # ---------------------------------
    # Length-dependent threshold
    # ---------------------------------

    if len(normalized) <= 5:
        minimum_score = 90.0

    elif len(normalized) <= 8:
        minimum_score = 85.0

    else:
        minimum_score = 82.0

    # ---------------------------------
    # Candidate vocabulary
    # ---------------------------------

    vocabulary = []

    for word, frequency in frequencies.items():

        if word == normalized:
            continue

        if frequency < minimum_frequency:
            continue

        # Don't compare words with wildly
        # different lengths.
        if abs(
            len(word) - len(normalized)
        ) > 2:
            continue

        vocabulary.append(word)

    if not vocabulary:
        return None

    result = process.extractOne(
        normalized,
        vocabulary,
        scorer=fuzz.ratio,
        score_cutoff=minimum_score,
    )

    if result is None:
        return None

    candidate, score, _ = result

    # Suggested form must occur more often.
    if (
        frequencies[candidate]
        <= frequencies[normalized]
    ):
        return None

    return {
        "candidate": candidate,
        "score": float(score),
        "candidate_frequency":
            frequencies[candidate],
    }
    
def looks_suspicious(token: str) -> list[str]:
    """
    Return reasons why a raw OCR token may deserve review.

    These are candidate signals, NOT proof that the token
    is incorrect.
    """
    
    ORDINAL_PATTERN = re.compile(
        r"^\d+(?:st|nd|rd|th)$",
        flags=re.IGNORECASE,
    )
    
    reasons = []

    # Mixed uppercase/lowercase in unusual positions:
    # e.g. eXcept
    if (
        len(token) >= 3
        and token[0].islower()
        and any(char.isupper() for char in token[1:])
    ):
        reasons.append("internal_uppercase")

    # Contains both letters and digits.
    if (
        any(char.isalpha() for char in token)
        and any(char.isdigit() for char in token)
        and not ORDINAL_PATTERN.fullmatch(token)
    ):
        reasons.append("alphanumeric")

    # Unusual symbols inside a lexical token.
    if re.search(
        r"[^A-Za-z0-9.'’-]",
        token,
    ):
        reasons.append("unusual_character")

    return reasons


def build_frequency_map(
    tokens: list[str],
) -> Counter:
    """
    Case-insensitive corpus frequency while preserving
    raw database tokens elsewhere.
    """

    return Counter(
        token.casefold()
        for token in tokens
    )
    
def get_correction_candidates(
    connection,
):
    """
    Build correction candidates from database tokens.

    Returns only tokens that have not already been
    human-reviewed.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            WITH lexical_tokens AS (
                SELECT
                    t.token_id,
                    t.raw_token,
                    t.position_in_page,

                    r.confidence,

                    p.page_id,
                    p.source_page_number,
                    p.image_path,

                    t.x_min,
                    t.y_min,
                    t.x_max,
                    t.y_max,

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

                WHERE
                    t.token_type = 'word'
                    AND t.correction_reviewed = FALSE
            )

            SELECT
                token_id,
                raw_token,
                confidence,
                source_page_number,
                position_in_page,
                page_id,
                image_path,

                x_min,
                y_min,
                x_max,
                y_max,

                before_2,
                before_1,
                after_1,
                after_2

            FROM lexical_tokens

            ORDER BY token_id;
            """
        )

        rows = cursor.fetchall()

    frequencies = (
        get_database_frequency_map(
            connection
        )
    )

    candidates = []

    for (
        token_id,
        raw_token,
        confidence,
        page_number,
        position,
        page_id,
        image_path,
        x_min,
        y_min,
        x_max,
        y_max,
        before_2,
        before_1,
        after_1,
        after_2,
    ) in rows:

        frequency = frequencies[
            raw_token.casefold()
        ]

        base_reasons = looks_suspicious(
            raw_token
        )

        repeated_stable_form = (
            frequency >= 3
            and base_reasons
            == ["alphanumeric"]
        )

        if repeated_stable_form:
            base_reasons = []

        neighbor = None

        if (
            not base_reasons
            and neighbor is None
        ):
            continue

        reasons = list(
            base_reasons
        )

        if neighbor is not None:
            reasons.append(
                "lexical_neighbor"
            )

        if frequency <= 2:
            reasons.append(
                "rare"
            )

        candidates.append(
            {
                "token_id": token_id,
                "token": raw_token,

                "frequency": frequency,
                "confidence": confidence,

                "page": page_number,
                "page_id": page_id,
                "position": position,

                "image_path": image_path,

                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },

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

    # Prioritize tokens with stronger evidence,
    # then lower-confidence OCR regions.
    candidates.sort(
        key=lambda item: (
            -len(item["reasons"]),
            item["confidence"],
        )
    )

    return candidates
    
def get_database_frequency_map(
    connection,
):
    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                LOWER(raw_token),
                COUNT(*)
            FROM token
            WHERE token_type = 'word'
            GROUP BY LOWER(raw_token);
            """
        )

        rows = cursor.fetchall()

    return Counter(
        {
            token: frequency
            for token, frequency in rows
        }
    )

