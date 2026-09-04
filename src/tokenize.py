import re


TOKEN_PATTERN = re.compile(
    r"""
    # Dotted initialisms
    # D.S.B. / H.S.B.
    (?:[^\W\d_]\.){2,}

    |

    # Words, ordinals, apostrophes and hyphens
    # Inspector-General
    # don't
    # 2nd
    [^\W_]+
    (?:['’-][^\W_]+)*

    |

    # Numbers
    \d+(?:['’-]\d+)*

    |

    # Remaining punctuation, including /
    [^\w\s]
    """,
    flags=re.UNICODE | re.VERBOSE,
)


def classify_token(token: str) -> str:
    """
    Classify a token without modifying it.
    """

    if token.isdigit():
        return "number"

    if re.fullmatch(
        r"[^\w\s]",
        token,
        flags=re.UNICODE,
    ):
        return "punctuation"

    return "word"


def tokenize_text(text: str) -> list[dict]:
    """
    Tokenize one Paddle component.

    Returns token text plus its character offsets
    within the component.
    """

    tokens = []

    for match in TOKEN_PATTERN.finditer(text):

        raw_token = match.group()

        tokens.append(
            {
                "raw_token": raw_token,
                "token_type": classify_token(
                    raw_token
                ),
                "char_start": match.start(),
                "char_end": match.end(),
            }
        )

    return tokens


def tokenize_regions(
    regions: list[dict],
) -> list[dict]:
    """
    Convert PaddleOCR regions/components into
    database-ready raw tokens.

    OCR spelling and capitalization are preserved.
    Whitespace components are excluded.
    """

    tokens = []

    page_position = 1

    for region in regions:

        region_position = 1

        for part in region["paddle_parts"]:

            part_text = part["text"]

            if not part_text.strip():
                continue

            part_tokens = tokenize_text(
                part_text
            )

            for item in part_tokens:

                token = {
                    "region_index":
                        region["region_index"],

                    "part_index":
                        part["part_index"],

                    "position_in_region":
                        region_position,

                    "position_in_page":
                        page_position,

                    "raw_token":
                        item["raw_token"],

                    "token_type":
                        item["token_type"],

                    "source_bbox":
                        part["bbox"],

                    "char_start":
                        item["char_start"],

                    "char_end":
                        item["char_end"],
                }

                tokens.append(token)

                region_position += 1
                page_position += 1

    return tokens
