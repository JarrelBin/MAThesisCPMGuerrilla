
import argparse
import base64
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3-vl-8b"


def verify_candidate(token_id: int, page: int):

    image_path = (
        PROJECT_ROOT
        / "data"
        / "experiments"
        / "vision_crops"
        / f"token_{token_id}_page_{page:04d}.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Crop not found: {image_path}"
        )

    image_bytes = image_path.read_bytes()

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("ascii")

    image_url = (
        "data:image/png;base64,"
        + image_base64
    )

    prompt = (
        "You are transcribing a historical document image.\n"
        "Read the text visible in the image exactly as printed.\n"
        "Preserve capitalization, spelling, names and punctuation.\n"
        "Do not correct grammar or modernize language.\n"
        "If a character is unclear, indicate uncertainty.\n"
        "Return only the transcription."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }

    print(f"Sending crop: {image_path.name}")

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=900,
    )

    response.raise_for_status()

    result = response.json()

    transcription = (
        result["choices"][0]["message"]["content"]
    )

    print()
    print("=== VISION TRANSCRIPTION ===")
    print()
    print(transcription)

    return transcription


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--token-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--page",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    verify_candidate(
        token_id=args.token_id,
        page=args.page,
    )


if __name__ == "__main__":
    main()
