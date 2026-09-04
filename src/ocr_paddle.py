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
            input=str(image_path),
            return_word_box=True,
        )

        regions = []

        for result in results:
            #result.print()
            data = result.json["res"]
            
            region_index = 4
            
            #tiny diagnostic code to confirm semantic alignment
            parts = data["text_word"][region_index]
            boxes = data["text_word_boxes"][region_index]

            print("TEXT:")
            print(data["rec_texts"][region_index])

            print()
            print("PARTS:")
            print(parts)

            print()
            print("NUMBER OF PARTS:")
            print(len(parts))

            print()
            print("NUMBER OF BOXES:")
            print(len(boxes))

            texts = data["rec_texts"]
            scores = data["rec_scores"]
            boxes = data["rec_boxes"]
            polygons = data["rec_polys"]
            
            word_parts = data["text_word"]
            word_boxes = data["text_word_boxes"]

            for sequence, (
                text,
                score,
                box,
                polygon,
                parts,
                part_boxes,
            ) in enumerate(
                zip(
                    texts,
                    scores,
                    boxes,
                    polygons,
                    word_parts,
                    word_boxes,
                ),
                start=1,
            ):
                paddle_parts = []

                for part_index, (
                    part,
                    part_box,
                ) in enumerate(
                    zip(
                        parts,
                        part_boxes,
                    ),
                    start=1,
                ):
                    paddle_parts.append(
                        {
                            "part_index": part_index,
                            "text": str(part),
                            "bbox": [
                                int(value)
                                for value in part_box
                            ],
                        }
                    )
    
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
                    
                    "paddle_parts": paddle_parts,
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
