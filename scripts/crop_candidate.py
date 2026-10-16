"""Create a visual verification crop for an OCR token.

Run from repository root:
  python -m scripts.crop_candidate --token-id 1466 --padding 160

Requires existing src.database.get_connection and PaddleOCR page JSON.
Does not modify PostgreSQL. Fails closed if the token cannot be uniquely
aligned to a PaddleOCR word box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.database import get_connection

ROOT = Path(__file__).resolve().parents[1]


def find_token(token_id: int) -> tuple[str, str, int]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT t.raw_token, sf.filename, p.source_page_number
                FROM token AS t
                JOIN ocr_region AS r ON r.region_id = t.region_id
                JOIN ocr_run AS o ON o.ocr_run_id = r.ocr_run_id
                JOIN page AS p ON p.page_id = o.page_id
                JOIN source_file AS sf ON sf.source_file_id = p.source_file_id
                WHERE t.token_id = %s
                """,
                (token_id,),
            )
            row = cursor.fetchone()
    if row is None:
        raise ValueError(f"No token found for token_id={token_id}")
    return str(row[0]), Path(row[1]).stem, int(row[2])


def bounds(box: object) -> tuple[int, int, int, int]:
    if not isinstance(box, (list, tuple)):
        raise ValueError(f"Unexpected word box: {box!r}")
    if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        x1, y1, x2, y2 = box
        return int(x1), int(y1), int(x2), int(y2)
    if len(box) >= 4 and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in box):
        return (
            int(min(pt[0] for pt in box)), int(min(pt[1] for pt in box)),
            int(max(pt[0] for pt in box)), int(max(pt[1] for pt in box)),
        )
    raise ValueError(f"Unsupported word box format: {box!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-id", type=int, required=True)
    parser.add_argument(
        "--padding-x",
        type=int,
        default=450,
        help="Horizontal padding around target word",
    )

    parser.add_argument(
        "--padding-y",
        type=int,
        default=100,
        help="Vertical padding around target word",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.padding_x < 0 or args.padding_y < 0:
        parser.error("--padding must be nonnegative")

    padding_x = args.padding_x
    padding_y = args.padding_y

    raw, source, page = find_token(args.token_id)
    stem = f"page_{page:04d}"
    ocr_path = ROOT / "data" / "ocr" / "paddle" / source / f"{stem}.json"
    image_path = ROOT / "data" / "images" / source / f"{stem}.png"
    regions = json.loads(ocr_path.read_text(encoding="utf-8"))
    matches = []
    for region in regions:
        for part in region.get("paddle_parts", []):
            if part.get("text") == raw:
                matches.append(part)

    if len(matches) != 1:
        raise RuntimeError(
            f"Found {len(matches)} matching PaddleOCR word boxes for {raw!r} "
            f"on {stem}. Cannot safely associate token_id={args.token_id} "
            "with a box using text alone. Need region/part ID alignment."
        )

    x1, y1, x2, y2 = bounds(matches[0]["bbox"])
    output = args.output or (ROOT / "data" / "experiments" / "vision_crops" / f"token_{args.token_id}_{stem}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as original:
        image = original.convert("RGB")
        left = max(0, x1 - padding_x)
        top = max(0, y1 - padding_y)

        right = min(image.width, x2 + padding_x)
        bottom = min(image.height, y2 + padding_y)

        crop = image.crop((left, top, right, bottom))
        crop.save(output)
    print(f"Token ID: {args.token_id} | OCR: {raw!r}")
    print(f"Source: {source} | page: {page}")
    print(f"Word box: {(x1, y1, x2, y2)}")
    print(f"Crop saved: {output}")
    print("IMPORTANT: This is a text-unique alignment only; verify crop visually.")


if __name__ == "__main__":
    main()
