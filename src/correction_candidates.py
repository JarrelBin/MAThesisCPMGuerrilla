"""Conservative, pattern-only OCR correction candidate detection.

RapidFuzz is intentionally not used here. It remains available to other
modules (for example, report boundary detection).
"""
from __future__ import annotations

import re
from collections import Counter

ORDINAL_PATTERN = re.compile(r"^\d+(?:st|nd|rd|th)$", flags=re.IGNORECASE)
UNUSUAL_CHARACTER_PATTERN = re.compile(r"[^A-Za-z0-9.'’-]")


def looks_suspicious(token: str) -> list[str]:
    """Return pattern signals, not proof of an OCR error."""
    reasons: list[str] = []
    if len(token) >= 3 and token[0].islower() and any(c.isupper() for c in token[1:]):
        reasons.append("internal_uppercase")
    if (any(c.isalpha() for c in token) and any(c.isdigit() for c in token)
            and not ORDINAL_PATTERN.fullmatch(token)):
        reasons.append("alphanumeric")
    if UNUSUAL_CHARACTER_PATTERN.search(token):
        reasons.append("unusual_character")
    return reasons


def build_frequency_map(tokens: list[str]) -> Counter:
    """Case-insensitive frequency counts, without changing raw database tokens."""
    return Counter(token.casefold() for token in tokens)


def get_database_frequency_map(connection) -> Counter:
    """Corpus-wide counts, including already-reviewed tokens."""
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT LOWER(raw_token), COUNT(*)
            FROM token
            WHERE token_type = 'word'
            GROUP BY LOWER(raw_token)
        """)
        return Counter(dict(cursor.fetchall()))


def get_correction_candidates(
    connection,
    *,
    source_file_id: int | None = None,
    source_page_number: int | None = None,
) -> list[dict]:
    """Return unreviewed, pattern-flagged words, optionally scoped to a PDF/page.

    Frequencies always cover the whole corpus. Context uses all word tokens
    on each page (including reviewed ones), then filters unreviewed targets.
    No writes are performed.
    """
    if source_page_number is not None and source_file_id is None:
        raise ValueError("source_page_number requires source_file_id")
    if source_file_id is not None and source_file_id < 1:
        raise ValueError("source_file_id must be positive")
    if source_page_number is not None and source_page_number < 1:
        raise ValueError("source_page_number must be positive")

    with connection.cursor() as cursor:
        cursor.execute("""
            WITH lexical_tokens AS (
                SELECT
                    t.token_id, t.raw_token, t.position_in_page,
                    t.correction_reviewed, r.confidence,
                    p.page_id, p.source_file_id, p.source_page_number, p.image_path,
                    t.x_min, t.y_min, t.x_max, t.y_max,
                    LAG(t.raw_token, 2) OVER (
                        PARTITION BY p.page_id ORDER BY t.position_in_page
                    ) AS before_2,
                    LAG(t.raw_token, 1) OVER (
                        PARTITION BY p.page_id ORDER BY t.position_in_page
                    ) AS before_1,
                    LEAD(t.raw_token, 1) OVER (
                        PARTITION BY p.page_id ORDER BY t.position_in_page
                    ) AS after_1,
                    LEAD(t.raw_token, 2) OVER (
                        PARTITION BY p.page_id ORDER BY t.position_in_page
                    ) AS after_2
                FROM token t
                JOIN ocr_region r ON r.region_id = t.region_id
                JOIN ocr_run o ON o.ocr_run_id = r.ocr_run_id
                JOIN page p ON p.page_id = o.page_id
                WHERE t.token_type = 'word'
            )
            SELECT token_id, raw_token, confidence, source_page_number,
                   position_in_page, page_id, image_path,
                   x_min, y_min, x_max, y_max,
                   before_2, before_1, after_1, after_2
            FROM lexical_tokens
            WHERE correction_reviewed = FALSE
              AND (%s::bigint IS NULL OR source_file_id = %s)
              AND (%s::integer IS NULL OR source_page_number = %s)
            ORDER BY token_id
        """, (source_file_id, source_file_id, source_page_number, source_page_number))
        rows = cursor.fetchall()

    frequencies = get_database_frequency_map(connection)
    candidates: list[dict] = []
    for (token_id, raw_token, confidence, page_number, position, page_id,
         image_path, x_min, y_min, x_max, y_max, before_2, before_1,
         after_1, after_2) in rows:
        frequency = frequencies[raw_token.casefold()]
        reasons = looks_suspicious(raw_token)
        # Preserve the original stable-reference exemption.
        if reasons == ["alphanumeric"] and frequency >= 3:
            reasons = []
        if not reasons:
            continue
        if frequency <= 2:
            reasons = [*reasons, "rare"]
        candidates.append({
            "token_id": token_id,
            "token": raw_token,
            "frequency": frequency,
            "confidence": confidence,
            "page": page_number,
            "page_id": page_id,
            "position": position,
            "image_path": image_path,
            "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            "reasons": reasons,
            "neighbor": None,
            "context": {
                "before_2": before_2, "before_1": before_1,
                "after_1": after_1, "after_2": after_2,
            },
        })
    candidates.sort(key=lambda c: (-len(c["reasons"]),
                                    c["confidence"] if c["confidence"] is not None else 1.0,
                                    c["token_id"]))
    return candidates
