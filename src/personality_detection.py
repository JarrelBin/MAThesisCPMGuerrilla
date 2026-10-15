"""OCR-only appendix heading detection; never modifies OCR artifacts."""
import re
from rapidfuzz.fuzz import ratio


def normalized(text):
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def is_personalities_start(regions):
    lines = [normalized(r.get('raw_text', '')) for r in regions]
    text = ' '.join(lines)
    return (bool(re.search(r'PERSONALIT(?:Y|IES)\s+MENTIONED', text))
            or ('PERSONALITIES' in text and 'PARTICULARS' in text)
            or ('PERSONALITIES' in text and 'IDENTIFYING DETAILS' in text))


def is_appendix_b(regions):
    # Check short OCR regions and adjacent regions, since headings may split.
    lines = [normalized(r.get('raw_text', '')) for r in regions]
    candidates = lines + [' '.join(lines[i:i+2]) for i in range(len(lines)-1)]
    for candidate in candidates:
        if len(candidate) > 45:
            continue
        words = candidate.split()
        for i, word in enumerate(words):
            if ratio(word, 'APPENDIX') < 75:
                continue
            if i+1 < len(words) and words[i+1] in ('B', '8'):
                return True
    return False
