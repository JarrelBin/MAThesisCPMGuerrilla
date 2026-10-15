"""Run the existing source-file pipeline without changing its component scripts.

Run from the project root:
    python -m scripts.process_source data/raw/source_02.pdf --sequence 2

Interactive review stages use the current terminal. A failed stage stops the
pipeline; completed database operations are not rolled back automatically.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.database import get_connection


def source_ids():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT source_file_id FROM source_file")
            return {row[0] for row in cursor.fetchall()}


def source_record(source_file_id):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT filename, source_path FROM source_file WHERE source_file_id = %s",
                (source_file_id,),
            )
            return cursor.fetchone()


def run_stage(label, module, *arguments):
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print(f"\n{'=' * 68}\n{label}\n$ {' '.join(command)}\n{'=' * 68}", flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Process one NEW PDF through the existing OCR and correction workflow.")
    parser.add_argument("pdf", type=Path, help="Path to the new source PDF")
    parser.add_argument("--sequence", type=int, required=True, help="Corpus file sequence")
    parser.add_argument("--with-ensemble", action="store_true", help="Run the optional ensemble generator before correction review")
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
            "Could not unambiguously identify one newly registered source_file_id "
            f"(new IDs: {sorted(new_ids)}). Stopping to avoid processing the wrong file."
        )
    source_file_id = new_ids.pop()
    record = source_record(source_file_id)
    if record is None or Path(record[0]).name != args.pdf.name:
        raise RuntimeError(
            f"New source ID {source_file_id} does not match PDF filename {args.pdf.name!r}. "
            "Stopping for manual verification."
        )

    print(f"\nNew source_file_id = {source_file_id}", flush=True)
    stages = [
        ("2. Run PaddleOCR", "scripts.run_paddle", ["--source-file", source_file_id, "--all"]),
        ("3. Preliminary sketch classification", "scripts.classify_pages", ["--source-file", source_file_id, "--all"]),
        ("4. Human review: sketches", "scripts.review_page_types", ["--source-file", source_file_id]),
        ("5. Detect report boundaries", "scripts.detect_reports", ["--source-file", source_file_id, "--all"]),
        ("6. Segment reports", "scripts.segment_reports", ["--source-file", source_file_id]),
        ("7. Classify personalities", "scripts.classify_report_pages", ["--source-file", source_file_id, "--all"]),
        ("8. Ingest OCR regions and tokens", "scripts.ingest_ocr", ["--source-file", source_file_id, "--all"]),
        ("9. Generate lexical correction candidates (existing corpus-wide command)", "scripts.generate_correction_candidates", []),
        ("10. Generate Qwen correction candidates", "scripts.generate_qwen_candidates", ["--source-file", source_file_id]),
    ]
    if args.with_ensemble:
        stages.append(("11. Generate ensemble candidates", "scripts.generate_ensemble_candidates", ["--source-file", source_file_id]))
    stages.append(("Human review: corrections (existing command; may cover multiple sources)", "scripts.review_corrections", []))

    try:
        for label, module, arguments in stages:
            run_stage(label, module, *arguments)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as exc:
        print(f"\nSTOPPED. Source file ID: {source_file_id}. Completed stages remain saved.", file=sys.stderr)
        print("Do not rerun the entire pipeline on this already-ingested PDF; resume the remaining commands manually.", file=sys.stderr)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else exc.returncode) from exc

    print(f"\nPipeline complete for source_file_id={source_file_id}.")


if __name__ == "__main__":
    main()
