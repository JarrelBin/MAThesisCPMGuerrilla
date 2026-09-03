import json
from pathlib import Path

from paddleocr import PaddleOCR


class PaddleOCREngine:
    def __init__(self, lang: str = "en"):
        """
        Initialise PaddleOCR.

        MKLDNN is disabled because of the compatibility issue
        encountered with the current PaddlePaddle CPU environment.
        """

        self.ocr = PaddleOCR(
            lang=lang,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize_page(
        self,
        image_path: Path,
        json_output_path: Path,
        text_output_path: Path,
    ) -> list[dict]:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        json_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        text_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        results = self.ocr.predict(
            input=str(image_path)
        )

        regions = []

        for result in results:

            data = result.json["res"]

            texts = data["rec_texts"]
            scores = data["rec_scores"]
            boxes = data["rec_boxes"]
            polygons = data["rec_polys"]

            for sequence, (
                text,
                score,
                box,
                polygon,
            ) in enumerate(
                zip(
                    texts,
                    scores,
                    boxes,
                    polygons,
                ),
                start=1,
            ):

                x_min, y_min, x_max, y_max = [
                    int(value) for value in box
                ]

                region = {
                    "region_index": sequence,
                    "raw_text": str(text),
                    "confidence": float(score),

                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    },

                    "polygon": [
                        [int(x), int(y)]
                        for x, y in polygon
                    ],
                }

                regions.append(region)

        # Save structured OCR
        with json_output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                regions,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # Save plain OCR
        with text_output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            for region in regions:
                file.write(
                    region["raw_text"] + "\n"
                )

        return regions
