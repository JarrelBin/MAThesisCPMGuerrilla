# src/tokenize.py

import re


def tokenize_document(pages):
    tokens = []

    for page in pages:
        text = page["raw_text"]

        for position, match in enumerate(
            re.finditer(r"\S+", text),
            start=1
        ):
            tokens.append({
                "page_number": page["page_number"],
                "position": position,
                "raw_token": match.group(),
                "char_start": match.start(),
                "char_end": match.end(),
            })

    return tokens
