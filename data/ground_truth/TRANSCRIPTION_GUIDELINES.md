# Ground Truth Transcription Guidelines

## 1. General principle

Ground truth represents the text visible in the historical source.

Do not:
- correct spelling
- expand abbreviations
- modernize language
- infer missing words
- copy corrections from OCR output

Always transcribe from the source image.

## 2. Capitalization

Preserve capitalization as printed.

Examples:

SECRET
Distribution:
External

## 3. Punctuation

Preserve visible punctuation.

Examples:

Inspector-General
(H.S.B.)
Kedah/Perlis

Do not add punctuation that is not visible.

## 4. Abbreviations

Preserve abbreviations exactly as printed.

Do not expand abbreviations.

Example:

Comdr

not:

Commander

## 5. Numbers

Preserve all printed numbers.

Numbers in structured fields such as the Copies column are
transcribed as document content.

## 6. Headers and security markings

Printed headers, classifications, and markings such as:

SECRET

are included in the transcription.

## 7. Printed/typewritten text

All visible printed or typewritten textual content is included,
including stop words.

## 8. Handwritten annotations

Handwritten annotations are not included in the primary OCR
ground truth.

They should be recorded separately if they are required for
later historical analysis.

## 9. Illegible characters

Do not guess illegible text.

Mark genuinely unreadable material using a consistent notation:

[illegible]

The use of this marker must be excluded or handled separately
during CER/WER evaluation.

## 10. Source errors

If the historical source itself contains a spelling or
typographical error, reproduce the error exactly.

Ground truth represents the source, not corrected language.

## 11. Whitespace

Do not attempt to reproduce exact horizontal spacing between
typewritten words.

Use a single space between words for textual evaluation.

Document layout is represented separately through region
coordinates and structural metadata.

## 12. Line structure

Preserve meaningful textual lines where possible.

Columns and other layout structures are represented separately
rather than reconstructed using spaces.
