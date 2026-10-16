"""Conservative page-level OCR detection and correction proposals via llama.cpp.

Qwen3-8B Q6_K is served locally using llama-server's chat API.
No Transformers/PyTorch dependencies are required for this module.
"""
from __future__ import annotations

import json
import re
from urllib import error, request

DEFAULT_MODEL = "qwen3-8b-q6"
DEFAULT_SERVER = "http://127.0.0.1:8080"
PROMPT_VERSION = "historical-page-detect-propose-v2"

SYSTEM_PROMPT = """You review OCR TEXT (not source images) from historical documents of
Malaya, Singapore, and southern Thailand. Preserve historical spelling,
Malay/Thai/Chinese vocabulary, names, aliases, capitalization, abbreviations,
reference codes, and original grammar. A rare word is not necessarily wrong.

Identify only probable CHARACTER RECOGNITION errors supported by specific
textual evidence. Do not flag words merely because a hypothetical character
confusion is possible. Do not invent image evidence. If uncertain, omit.

For each probable OCR error, propose ONE minimally changed replacement token.
The replacement must correct an apparent OCR transcription error, not rewrite,
modernize, expand an abbreviation, or change meaning. Do not alter surrounding
words. If no defensible replacement exists, omit the token.

Explain each proposal in at most 20 words. Reasons must identify a specific
anomaly and why the proposed reading fits. Avoid generic speculation.
Never claim a character was seen in the image.

Return ONLY JSON, exactly this shape:
{"candidates":[{"token_id":123,"raw_token":"Distri ot","suggested_token":"District","reason":"Specific explanation"}]}
If no convincing errors: {"candidates":[]}.
"""


def reconstruct_text(tokens: list[dict]) -> str:
    text = ""
    no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "}", "%"}
    no_space_after = {"(", "[", "{"}
    for token in tokens:
        raw = str(token["raw_token"])
        if not text:
            text = raw
        elif raw in no_space_before or text[-1] in no_space_after:
            text += raw
        else:
            text += " " + raw
    return text


class QwenCorrectionModel:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        server_url: str = DEFAULT_SERVER,
        context_size: int = 8192,
        max_new_tokens: int = 2048,
        timeout: int = 900,
        verbose: bool = False,
    ) -> None:
        if context_size <= max_new_tokens + 256:
            raise ValueError("Context must exceed output budget by at least 256 tokens")
        self.model_name = model_name
        self.server_url = server_url.rstrip("/")
        self.context_size = context_size
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.verbose = verbose
        self.last_response: dict | None = None
        models = self._request("GET", "/v1/models")
        ids = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
        if ids and model_name not in ids:
            raise RuntimeError(f"Model {model_name!r} not found on server; available: {ids}")
        print(f"Connected to llama.cpp: {self.server_url} | model={model_name}")

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.server_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail[:1200]}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"llama.cpp unavailable: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected response from {path}")
        return result

    def _prompt(self, tokens: list[dict]) -> str:
        text = reconstruct_text(tokens)
        mapping = "\n".join(f"[{t['token_id']}] {t['raw_token']}" for t in tokens)
        return (
            "Review ONE complete OCR page. The token map provides authoritative token IDs.\n"
            "Use the page text for context; only return tokens from the map.\n"
            "Return one minimal replacement per genuine suspected OCR error.\n"
            "Do not flag names such as SAMNAK OK or words such as padi merely for unfamiliarity.\n"
            "A proposal must DIFFER from the original token. Do not invent changes.\n\n"
            f"PAGE TEXT:\n{text}\n\nTOKEN MAP:\n{mapping}\n\n"
            'Return JSON only: {"candidates":[{"token_id":1,"raw_token":"...",'
            '"suggested_token":"...","reason":"..."}]} or {"candidates":[]}.'
        )

    def analyze_page(self, tokens: list[dict]) -> list[dict]:
        self.last_response = None
        if not tokens:
            return []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._prompt(tokens)},
        ]
        if self.verbose:
            print(messages[1]["content"])
        # /tokenize is not chat-template-aware; keep a conservative safety margin.
        tokenized = self._request("POST", "/tokenize", {
            "content": SYSTEM_PROMPT + "\n" + messages[1]["content"],
            "add_special": False,
        })
        count = tokenized.get("tokens")
        if not isinstance(count, list):
            raise ValueError("Server /tokenize returned no token list")
        estimated = len(count)
        budget = self.context_size - self.max_new_tokens - 256
        print(f"  Estimated prompt tokens: {estimated}; safe budget: {budget}")
        if estimated > budget:
            raise ValueError(
                f"Page too long ({estimated} estimated tokens > {budget}); "
                "increase server context or use an explicit segmentation strategy"
            )
        result = self._request("POST", "/v1/chat/completions", {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        })
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("No completion choices")
        choice = choices[0]
        content = (choice.get("message") or {}).get("content")
        self.last_response = {
            "prompt_version": PROMPT_VERSION,
            "model": self.model_name,
            "estimated_prompt_tokens": estimated,
            "finish_reason": choice.get("finish_reason"),
            "usage": result.get("usage"),
            "raw_response": content,
        }
        if choice.get("finish_reason") == "length":
            raise ValueError("Qwen response truncated; no candidates stored for page")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen returned empty response")
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(
                line for line in content.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        if self.verbose:
            print(content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {content[:800]!r}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("candidates"), list):
            raise ValueError("Response must be an object with candidates array")
        return parsed["candidates"]

    # Compatibility with previous call sites.
    def analyze_token_window(self, tokens: list[dict]) -> list[dict]:
        return self.analyze_page(tokens)
