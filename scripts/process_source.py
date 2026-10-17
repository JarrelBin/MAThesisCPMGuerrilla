"""Register and process a new PDF, or explicitly resume an existing source.

New source:
    python -m scripts.process_source data/raw/source_02.pdf --sequence 2

Resume an existing source at a chosen stage:
    python -m scripts.process_source --resume-source-file 2 --start-stage 9

Resume skips registration. It does not automatically detect completed stages.
Use --start-stage carefully: repeating non-idempotent ingestion or segmentation
may create duplicate records. Final OCR correction review is separate.
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
                "SELECT filename, source_path, file_sequence FROM source_file WHERE source_file_id = %s",
                (source_file_id,),
            )
            return cursor.fetchone()


def run_stage(label: str, module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(arg) for arg in arguments)]
    print(f"\n{'=' * 68}\n{label}\n$ {' '.join(command)}\n{'=' * 68}", flush=True)
    subprocess.run(command, check=True)


def build_stages(source_file_id: int, skip_vision: bool):
    stages = [
        (2, "Run PaddleOCR", "scripts.run_paddle", ["--source-file", source_file_id, "--all"]),
        (3, "Preliminary sketch classification", "scripts.classify_pages", ["--source-file", source_file_id, "--all"]),
        (4, "Human review: sketches", "scripts.review_page_types", ["--source-file", source_file_id]),
        (5, "Detect report boundaries", "scripts.detect_reports", ["--source-file", source_file_id, "--all"]),
        (6, "Segment reports", "scripts.segment_reports", ["--source-file", source_file_id]),
        (7, "Classify personalities", "scripts.classify_report_pages", ["--source-file", source_file_id, "--all"]),
        (8, "Ingest OCR regions and tokens", "scripts.ingest_ocr", ["--source-file", source_file_id, "--all"]),
        (9, "Generate pattern correction candidates", "scripts.generate_correction_candidates", ["--source-file", source_file_id]),
    ]
    qwen_args = ["--source-file", source_file_id]
    if not skip_vision:
        qwen_args.append("--vision")
    stages.append((10, "Generate Qwen correction candidates", "scripts.generate_qwen_candidates", qwen_args))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a new source PDF or resume an existing source from a chosen stage.")
    parser.add_argument("pdf", nargs="?", type=Path, help="New PDF path (omit when resuming)")
    parser.add_argument("--sequence", type=int, help="Corpus sequence (required for a new PDF)")
    parser.add_argument("--resume-source-file", type=int, metavar="ID", help="Existing source_file_id; skip registration")
    parser.add_argument("--start-stage", type=int, choices=range(2, 11), metavar="2-10", help="First stage to execute when resuming")
    parser.add_argument("--skip-vision", action="store_true", help="Run Qwen text correction only")
    args = parser.parse_args()

    resume = args.resume_source_file is not None
    if resume:
        if args.resume_source_file < 1:
            parser.error("--resume-source-file must be positive")
        if args.pdf is not None or args.sequence is not None:
            parser.error("Do not supply a PDF or --sequence when using --resume-source-file")
        if args.start_stage is None:
            parser.error("--start-stage is required when resuming; choose the first unfinished stage")
        source_file_id = args.resume_source_file
        record = source_record(source_file_id)
        if record is None:
            parser.error(f"Source file ID {source_file_id} does not exist")
        print(f"Resuming source_file_id={source_file_id}: {record[0]} (sequence {record[2]})", flush=True)
        print(f"Starting at stage {args.start_stage}. Earlier stages will NOT run.", flush=True)
        if args.start_stage <= 8:
            print("WARNING: Verify that repeating this stage will not duplicate database records.", flush=True)
        first_stage = args.start_stage
    else:
        if args.start_stage is not None:
            parser.error("--start-stage requires --resume-source-file")
        if args.pdf is None or args.sequence is None:
            parser.error("New source requires PDF and --sequence")
        if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
            parser.error(f"PDF not found or not a PDF: {args.pdf}")
        if args.sequence < 1:
            parser.error("--sequence must be positive")
        before = source_ids()
        run_stage("1. Register source PDF", "scripts.ingest_source", args.pdf, "--sequence", args.sequence)
        new_ids = source_ids() - before
        if len(new_ids) != 1:
            raise RuntimeError(
                "Could not identify exactly one newly registered source_file_id "
                f"(new IDs: {sorted(new_ids)}). If already registered, use --resume-source-file."
            )
        source_file_id = new_ids.pop()
        record = source_record(source_file_id)
        if record is None or Path(record[0]).name != args.pdf.name:
            raise RuntimeError(f"Source ID {source_file_id} does not match PDF filename {args.pdf.name!r}")
        print(f"New source_file_id={source_file_id}", flush=True)
        first_stage = 2

    stages = [stage for stage in build_stages(source_file_id, args.skip_vision) if stage[0] >= first_stage]
    try:
        for number, label, module, arguments in stages:
            run_stage(f"{number}. {label}", module, *arguments)
    except KeyboardInterrupt as exc:
        print(f"\nSTOPPED. Source file ID: {source_file_id}. Check completed stages before resuming.", file=sys.stderr)
        raise SystemExit(130) from exc
    except subprocess.CalledProcessError as exc:
        print(f"\nFAILED. Source file ID: {source_file_id}. Resume with --start-stage after diagnosing the failure.", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    print(f"\nPipeline complete for source_file_id={source_file_id}.")
    print("Human correction review remains a separate command.")


if __name__ == "__main__":
    main()
