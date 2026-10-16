"""Run the source-file OCR and correction pipeline.

From the repository root:
    python -m scripts.process_source data/raw/source_02.pdf --sequence 2
    python -m scripts.process_source data/raw/source_02.pdf --sequence 2 --with-ensemble

This script only registers NEW source files. If a stage fails after registration,
resume the remaining commands manually; do not register the same PDF again.
Interactive review stages use the current terminal.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.database import get_connection


def source_ids() -> set[int]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT source_file_id FROM source_file")
            return {row[0] for row in cursor.fetchall()}


def source_record(source_file_id: int):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT filename, source_path FROM source_file WHERE source_file_id = %s",
                (source_file_id,),
            )
            return cursor.fetchone()


def run_stage(label: str, module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(arg) for arg in arguments)]
    print(f"\n{'=' * 68}\n{label}\n$ {' '.join(command)}\n{'=' * 68}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process one NEW PDF through OCR, correction proposals, and human review."
    )
    parser.add_argument("pdf", type=Path, help="Path to a new source PDF")
    parser.add_argument("--sequence", type=int, required=True, help="Corpus file sequence")

    parser.add_argument(
        "--skip-vision", action="store_true",
        help="Run Qwen text correction only (default includes vision verification)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if args.pdf.suffix.lower() != ".pdf":
        parser.error("Input must be a PDF")
    if args.sequence < 1:
        parser.error("--sequence must be positive")

    before = source_ids()
    run_stage("1. Register source PDF", "scripts.ingest_source", args.pdf, "--sequence", args.sequence)
    new_ids = source_ids() - before
    if len(new_ids) != 1:
        raise RuntimeError(
            "Could not identify exactly one newly registered source_file_id "
            f"(new IDs: {sorted(new_ids)}). Stop to avoid processing the wrong PDF."
        )

    source_file_id = new_ids.pop()
    record = source_record(source_file_id)
    if record is None or Path(record[0]).name != args.pdf.name:
        raise RuntimeError(
            f"Source ID {source_file_id} does not match PDF filename {args.pdf.name!r}."
        )

    print(f"\nNew source_file_id = {source_file_id}", flush=True)

    stages: list[tuple[str, str, list[object]]] = [
        ("2. Run PaddleOCR", "scripts.run_paddle", ["--source-file", source_file_id, "--all"]),
        ("3. Preliminary sketch classification", "scripts.classify_pages", ["--source-file", source_file_id, "--all"]),
        ("4. Human review: sketches", "scripts.review_page_types", ["--source-file", source_file_id]),
        ("5. Detect report boundaries", "scripts.detect_reports", ["--source-file", source_file_id, "--all"]),
        ("6. Segment reports", "scripts.segment_reports", ["--source-file", source_file_id]),
        ("7. Classify personalities", "scripts.classify_report_pages", ["--source-file", source_file_id, "--all"]),
        ("8. Ingest OCR regions and tokens", "scripts.ingest_ocr", ["--source-file", source_file_id, "--all"]),
        ("9. Generate lexical correction candidates", "scripts.generate_correction_candidates", ["--source-file", source_file_id]),
    ]

    qwen_args: list[object] = ["--source-file", source_file_id]
    if not args.skip_vision:
        qwen_args.append("--vision")
    stages.append(("10. Generate Qwen correction candidates", "scripts.generate_qwen_candidates", qwen_args))

    try:
        for label, module, arguments in stages:
            run_stage(label, module, *arguments)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as exc:
        print(
            f"\nSTOPPED. Source file ID: {source_file_id}. Completed stages remain saved.",
            file=sys.stderr,
        )
        print(
            "Do not rerun the entire pipeline for this already-registered PDF. "
            "Resume remaining stages manually.",
            file=sys.stderr,
        )
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else exc.returncode) from exc

    print(f"\nPipeline complete for source_file_id={source_file_id}.")


if __name__ == "__main__":
    main()
