import re
from dataclasses import dataclass
from rapidfuzz.fuzz import ratio

@dataclass
class BoundaryDetection:
    is_candidate: bool

    has_our_ref: bool
    fuzzy_our_ref: bool
    our_ref_score: float | None

    has_statement_by: bool
    has_statement_of: bool
    has_interview_report: bool

    report_ref_raw: str | None


def normalize_detection_text(
    text: str,
) -> str:
    """
    Normalize OCR text for boundary detection only.

    Raw OCR stored elsewhere is never changed.
    """

    text = text.casefold()

    # Collapse arbitrary whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_report_ref(
    regions: list[dict],
) -> str | None:
    """
    Attempt to extract text following 'OUR REF'.

    This is deliberately conservative.
    """

    for region in regions:

        raw_text = region["raw_text"]

        normalized = (
            normalize_detection_text(
                raw_text
            )
        )

        if "our ref" not in normalized:
            continue

        # Examples:
        #
        # OUR REF: ABC/123
        # OUR REF ABC/123
        # Our Ref : ABC/123

        match = re.search(
            r"our\s+ref\s*:?\s*(.+)",
            raw_text,
            flags=re.IGNORECASE,
        )

        if match:
            value = match.group(1).strip()

            if value:
                return value

    return None


def detect_report_boundary(
    regions: list[dict],
) -> BoundaryDetection:

    page_text = "\n".join(
        region["raw_text"]
        for region in regions
    )

    normalized = (
        normalize_detection_text(
            page_text
        )
    )

    has_our_ref = (
        "our ref" in normalized
    )
    
    fuzzy_our_ref, our_ref_score = (
        detect_fuzzy_our_ref(
            regions
        )
    )

    has_statement_by = (
        "statement by" in normalized
    )

    has_statement_of = (
        "statement of" in normalized
    )

    has_interview_report = (
        "interview report" in normalized
    )
    
    has_supporting_signal = (
        has_statement_by
        or has_statement_of
        or has_interview_report
    )

    # Primary rule:
    #
    # OUR REF is sufficient to identify
    # a candidate report front page.
    #
    # Secondary phrases are retained so
    # we can later recover OCR misses.

    is_candidate = (
        has_our_ref
        or (
            fuzzy_our_ref
            and has_supporting_signal
        )
    )

    report_ref_raw = extract_report_ref(
        regions
    )

    return BoundaryDetection(
        is_candidate=is_candidate,

        has_our_ref=has_our_ref,
        fuzzy_our_ref=fuzzy_our_ref,
        our_ref_score=our_ref_score,

        has_statement_by=has_statement_by,
        has_statement_of=has_statement_of,
        has_interview_report=(
            has_interview_report
        ),

        report_ref_raw=report_ref_raw,
    )
    
def detect_fuzzy_our_ref(
    regions: list[dict],
    threshold: float = 75.0,
) -> tuple[bool, float | None]:
    """
    Look for OCR variants of 'OUR REF'.

    Only short OCR regions are considered to reduce
    false positives from ordinary prose.
    """

    best_score = 0.0

    for region in regions:

        text = normalize_detection_text(
            region["raw_text"]
        )

        if not text:
            continue

        # OUR REF is a short label.
        # Avoid fuzzy matching long prose regions.
        if len(text) > 20:
            continue

        # Remove common trailing punctuation/value text.
        words = text.split()

        candidate = " ".join(
            words[:2]
        )

        score = ratio(
            candidate,
            "our ref",
        )

        if score > best_score:
            best_score = score

    if best_score >= threshold:
        return True, best_score

    return False, best_score
