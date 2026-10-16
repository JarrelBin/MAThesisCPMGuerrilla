"""Exact token_id -> stored PostgreSQL word bounding box -> horizontal PNG crop."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from PIL import Image
from src.paths import from_project_relative


def fetch_token_geometry(connection, token_id: int) -> dict:
    with connection.cursor() as cur:
        cur.execute('''
            SELECT t.token_id, t.raw_token, t.x_min, t.y_min, t.x_max, t.y_max,
                   t.part_index, t.region_id, p.image_path, p.source_page_number
            FROM token t JOIN ocr_region r ON r.region_id=t.region_id
            JOIN ocr_run o ON o.ocr_run_id=r.ocr_run_id
            JOIN page p ON p.page_id=o.page_id
            WHERE t.token_id=%s
        ''', (token_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f'Token {token_id} not found')
    keys = ('token_id','raw_token','x_min','y_min','x_max','y_max','part_index','region_id','image_path','source_page_number')
    return dict(zip(keys, row))


def make_crop(geometry: dict, *, padding_x: int = 650, padding_y: int = 100) -> tuple[bytes, tuple[int,int,int,int]]:
    if min(padding_x, padding_y) < 0:
        raise ValueError('Padding must be nonnegative')
    coords = [geometry[k] for k in ('x_min','y_min','x_max','y_max')]
    if any(v is None for v in coords):
        raise ValueError(f'Token {geometry["token_id"]} has no word-level bounding box')
    x1,y1,x2,y2 = map(int, coords)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid token bounding box: {coords}')
    image_path = from_project_relative(geometry['image_path'])
    with Image.open(image_path) as source:
        image = source.convert('RGB')
        box = (max(0,x1-padding_x), max(0,y1-padding_y), min(image.width,x2+padding_x), min(image.height,y2+padding_y))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f'Invalid crop bounds {box}')
        out = BytesIO()
        image.crop(box).save(out, format='PNG')
    return out.getvalue(), box
