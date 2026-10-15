import json

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B"



SYSTEM_PROMPT = """
You are a conservative OCR error detector specializing in
historical English-language documents from the Malay Peninsula
and Southern Thailand.

Your task is DETECTION ONLY.

Identify tokens that may contain OCR character recognition
errors. Do not suggest corrected words.

HISTORICAL DOCUMENT CHARACTERISTICS

The documents may legitimately contain:
- Malay, Chinese and Thai personal names
- Place names and geographical references
- Personal aliases and nicknames
- Abbreviations and organizational codes
- Historical terminology
- Non-English vocabulary
- Unusual capitalization
- Grammatical errors in the original source
- Dates, numbers and reference identifiers

Personal names, aliases and place names frequently appear
in uppercase letters.

HISTORICAL NOTATION

The documents may contain structured references such as:

- Personal names followed by numbers in parentheses
- Gender or other abbreviations in parentheses
- Organizational abbreviations such as CT and CTO
- Numbered paragraphs and headings
- Dates and reference identifiers

These structures are not inherently OCR errors.

For example, a single letter or number in parentheses
may be a legitimate reference or classification.

Do not flag such tokens unless there is specific
evidence of character recognition failure.

Do not assume that uppercase names are OCR errors.

DETECTION PRINCIPLES

1. Look for evidence of possible OCR character recognition
   errors, including:
   - Confusion between visually similar characters
   - Missing or duplicated characters
   - Incorrect character substitutions
   - Broken or malformed words
   - Character sequences inconsistent with the surrounding
     sentence and plausible OCR recognition patterns

2. Use the surrounding sentence to understand the token,
   but do not rely on linguistic plausibility alone.

3. An unfamiliar, unusual or uncommon word is NOT sufficient
   evidence of an OCR error.

4. Do not flag personal names, place names, abbreviations
   or historical terminology merely because they are
   unfamiliar or contextually unusual.

5. However, these categories are NOT automatically exempt.
   Flag them when there is specific evidence of an OCR
   recognition problem.

6. If there is insufficient evidence of an OCR error,
   do not flag the token.

STRICT RESTRICTIONS

Do NOT:
- Suggest corrected words
- Rewrite or improve sentences
- Correct grammar
- Modernize spelling
- Normalize personal or place names
- Expand abbreviations
- Change singular or plural forms
- Flag words merely because they are uncommon
- Invent OCR errors to produce candidates

REASONING REQUIREMENT

For every flagged token, provide a brief explanation
of the suspected recognition problem.

The explanation must:
- Be no longer than 15 words
- Describe the suspected OCR recognition issue
- Refer to character-level or contextual evidence
- Avoid proposing a replacement word
- Avoid unsupported claims about the source image

You are examining OCR TEXT, not the original image.
You cannot confirm what characters are visually present.

If no tokens have sufficient evidence of an OCR error,
return an empty candidates list.

Return valid JSON only.
"""


def reconstruct_text(tokens: list[dict]) -> str:

    text = ""

    no_space_before = {
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        ")",
        "]",
        "}",
        "%",
    }

    no_space_after = {
        "(",
        "[",
        "{",
    }

    for token in tokens:

        raw = token["raw_token"]

        if not text:
            text = raw
            continue

        if raw in no_space_before:
            text += raw

        elif text[-1] in no_space_after:
            text += raw

        else:
            text += " " + raw

    return text

class QwenCorrectionModel:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):

        self.model_name = (
            model_name
        )

        print(
            f"Loading Qwen: "
            f"{self.model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        )

        self.model.eval()

        print(
            f"Qwen loaded on "
            f"{self.model.device}"
        )
        
    def analyze_token_window(
        self,
        tokens: list[dict],
    ) -> list[dict]:

        token_text = reconstruct_text(tokens)
        
        token_map = "\n".join(
            (
                f"[{token['token_id']}] "
                f"{token['raw_token']}"
            )
            for token in tokens
        )
        print(token_text)
        print(token_map)

        
        prompt = f"""
        Examine the following historical OCR text.

        Your task is to identify tokens that may contain
        OCR character recognition errors.

        You must NOT suggest corrections.

        ORIGINAL OCR TEXT:

        {token_text}

        TOKEN MAP:

        {token_map}

        INSTRUCTIONS

        1. Read ORIGINAL OCR TEXT to understand the surrounding
           sentences, punctuation and historical context.

        2. Use TOKEN MAP to identify the exact database token_id
           and raw_token for each suspected error.

        3. Only identify tokens with plausible evidence of an OCR
           character recognition error.

        4. Do not flag tokens merely because they are:
           - Personal names
           - Place names
           - Abbreviations
           - Unfamiliar words
           - Historically unusual expressions

        5. For every suspected error, copy token_id and raw_token
           EXACTLY from TOKEN MAP.

        6. Provide a brief reason explaining the suspected OCR
           recognition problem, without suggesting a replacement.

        7. If you cannot identify a plausible OCR recognition
           problem, omit that token.

        8. Do not flag tokens solely to populate the candidates list.

        OUTPUT FORMAT

        Return one JSON object with a "candidates" array.

        Each candidate must contain:
        - token_id: integer copied from TOKEN MAP
        - raw_token: exact string copied from TOKEN MAP
        - reason: a brief, token-specific explanation

        The reason must identify what makes THIS token
        suspicious.

        If you cannot explain the specific suspected
        recognition problem, omit the token.

        If there are no candidates, return:
        {{"candidates":[]}}
        """
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        inputs = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,

                # Hard-disable Qwen3 thinking.
                enable_thinking=False,

                return_tensors="pt",
                return_dict=True,
            )
        )

        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,

                # JSON should be very short.
                max_new_tokens=256,

                # Deterministic candidate generation.
                do_sample=False,
            )

        generated = outputs[0][
            inputs[
                "input_ids"
            ].shape[1]:
        ]

        content = (
            self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            .strip()
        )
        
        
        print(content)

        if not content:
            return []

        # Some models may still wrap JSON in
        # Markdown fences despite instructions.
        if content.startswith(
            "```"
        ):
            lines = (
                content.splitlines()
            )

            lines = [
                line
                for line in lines
                if not line.strip().startswith(
                    "```"
                )
            ]

            content = "\n".join(
                lines
            ).strip()

        try:
            parsed = json.loads(
                content
            )

        except json.JSONDecodeError:

            print()
            print(
                "WARNING: Qwen returned "
                "invalid JSON:"
            )

            print(
                content
            )

            return []

        candidates = parsed.get(
            "candidates",
            []
        )

        if not isinstance(
            candidates,
            list,
        ):
            return []

        return candidates
