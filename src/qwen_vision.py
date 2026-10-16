"""Neutral, image-only transcription via a local llama.cpp multimodal server."""
from __future__ import annotations
import base64
import json
from urllib import request, error

VISION_MODEL = "qwen3-vl-8b"

def transcribe_crop(image_bytes: bytes, *, url: str = "http://127.0.0.1:8080", model: str = VISION_MODEL, timeout: int = 900) -> dict:
    image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe exactly the printed text in this cropped historical document image. Preserve spelling, capitalization, names, and punctuation. Do not correct or guess. Return only the transcription."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        "temperature": 0, "max_tokens": 256, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = request.Request(url.rstrip('/') + '/v1/chat/completions', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(f"Vision HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:800]}") from exc
    choice = data['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise RuntimeError('Vision transcription truncated')
    content = choice['message']['content']
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('Empty vision transcription')
    return {'transcription': content.strip(), 'usage': data.get('usage'), 'finish_reason': choice.get('finish_reason')}
