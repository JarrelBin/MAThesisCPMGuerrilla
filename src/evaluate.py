import re

from jiwer import cer, wer


def normalize_for_evaluation(text: str) -> str:
    """
    Normalize layout-related whitespace without changing
    the actual textual content.

    - tabs become spaces
    - repeated spaces collapse to one
    - leading/trailing whitespace is removed
    """

    text = text.replace("\t", " ")

    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ ]+", " ", line)
        line = line.strip()

        if line:
            lines.append(line)

    return " ".join(lines)


def evaluate_text(
    reference: str,
    hypothesis: str,
) -> dict:

    reference = normalize_for_evaluation(
        reference
    )

    hypothesis = normalize_for_evaluation(
        hypothesis
    )

    return {
        "cer": cer(
            reference,
            hypothesis,
        ),
        "wer": wer(
            reference,
            hypothesis,
        ),
        "reference": reference,
        "hypothesis": hypothesis,
    }
