import json
from pathlib import Path

from paddleocr import PaddleOCR


class PaddleOCREngine:
    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        enable_mkldnn: bool = False,
    ):
        """
        Initialise PaddleOCR.

        MKLDNN is disabled because of the compatibility
        issue encountered with the current PaddlePaddle
        CPU environment.
        """

        self.lang = lang
        self.device = device
        self.enable_mkldnn = enable_mkldnn

        self.ocr = PaddleOCR(
            lang=self.lang,
            device=self.device,
            enable_mkldnn=self.enable_mkldnn,
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

            texts = data["rec_texts"]
            scores = data["rec_scores"]
            boxes = data["rec_boxes"]
            polygons = data["rec_polys"]
            
            word_parts = data["text_word"]
            word_boxes = data["text_word_boxes"]

            lengths = {
                "texts": len(texts),
                "scores": len(scores),
                "boxes": len(boxes),
                "polygons": len(polygons),
                "word_parts": len(word_parts),
                "word_boxes": len(word_boxes),
            }

            if len(set(lengths.values())) != 1:
                raise ValueError(
                    "PaddleOCR returned mismatched "
                    f"region counts: {lengths}"
                )

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
