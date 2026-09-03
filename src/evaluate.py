from jiwer import cer, wer


def evaluate_regions(regions):
    """
    Evaluate reviewed OCR regions against ground truth.

    Illegible regions are excluded.
    """

    references = []
    hypotheses = []

    for region in regions:

        if region["annotation_status"] != "correct":
            continue

        gt = region["gt_text"]
        ocr = region["ocr_text"]

        if gt is None:
            continue

        references.append(gt)
        hypotheses.append(ocr)

    reference_text = "\n".join(references)
    hypothesis_text = "\n".join(hypotheses)

    return {
        "cer": cer(
            reference_text,
            hypothesis_text,
        ),
        "wer": wer(
            reference_text,
            hypothesis_text,
        ),
        "evaluated_regions": len(references),
    }
