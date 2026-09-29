def classify_from_ocr(
    regions: list[dict],
) -> dict:
    """
    Classify a page from PaddleOCR output.

    This does not modify OCR. It only estimates whether
    the page looks like a sketch/diagram rather than a
    normal prose page.
    """

    region_count = len(regions)

    if not regions:
        return {
            "probable_sketch": True,
            "score": 3,
            "region_count": 0,
            "word_count": 0,
            "long_region_count": 0,
            "short_region_ratio": 1.0,
        }

    word_count = 0
    long_region_count = 0
    short_region_count = 0

    for region in regions:

        text = (
            region["raw_text"]
            .strip()
        )

        words = text.split()

        word_count += len(words)

        if len(words) >= 6:
            long_region_count += 1

        if len(words) <= 2:
            short_region_count += 1

    short_region_ratio = (
        short_region_count
        / region_count
    )

    # ---------------------------------
    # Sketch score
    #
    # These are INITIAL thresholds.
    # Tune against your known sketch and
    # prose pages before production.
    # ---------------------------------

    score = 0

    # Sketches tend to contain much less text.
    if word_count < 100:
        score += 1

    # Few sentence/paragraph-like OCR regions.
    if long_region_count < 5:
        score += 1

    # Most OCR consists of isolated labels.
    if short_region_ratio >= 0.60:
        score += 1

    probable_sketch = (
        score >= 2
    )

    return {
        "probable_sketch":
            probable_sketch,

        "score":
            score,

        "region_count":
            region_count,

        "word_count":
            word_count,

        "long_region_count":
            long_region_count,

        "short_region_ratio":
            short_region_ratio,
    }
