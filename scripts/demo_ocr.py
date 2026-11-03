"""Temporary OCR course demonstration. Run: python -m scripts.demo_ocr"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import readline
import glob
from pathlib import Path

from src.database import get_connection
from src.ingest import inspect_source_file
from src.repository import get_source_file_by_sha256

ENGINES = {
    "1": ("PaddleOCR", "paddle"),
    "2": ("Kraken - English", "kraken_en"),
    "3": ("Kraken - McCATMuS", "kraken_mccatmus"),
}
CORRECTIONS = {
    "1": ("Pattern-based", "pattern"),
    "2": ("OCR text GenAI", "textual_genai"),
    "3": ("OCR vision GenAI", "vision_genai"),
}


def run_module(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *(str(a) for a in args)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def select_engine() -> str | None:
    while True:
        print("\n=== OCR ENGINE SELECTION ===")
        for key, (label, _) in ENGINES.items():
            print(f"{key}. {label}")
        print("0. Quit")
        choice = input("Select engine: ").strip()
        if choice == "0":
            return None
        if choice in ENGINES:
            engine = ENGINES[choice][1]
            if engine != "paddle":
                print("This engine is not connected yet. Choose PaddleOCR.")
                continue
            return engine
        print("Invalid selection.")

def pdf_path_completer(text, state):
    """
    Autocomplete directories and PDF files.
    """
    expanded = str(Path(text).expanduser())

    matches = glob.glob(
        glob.escape(expanded) + "*"
    )

    matches = [
        match + "/" if Path(match).is_dir() else match
        for match in matches
        if Path(match).is_dir()
        or Path(match).suffix.lower() == ".pdf"
    ]

    matches.sort()

    if state < len(matches):
        return matches[state]

    return None


readline.set_completer(pdf_path_completer)
readline.parse_and_bind("tab: complete")
readline.set_completer_delims("\t\n")

def select_pdf() -> Path | None:
    while True:
        value = input(
            "\nPDF path (0 to quit): "
        ).strip()

        if value == "0":
            return None

        pdf = Path(value).expanduser().resolve()

        if pdf.is_file() and pdf.suffix.lower() == ".pdf":
            return pdf

        print("PDF not found. Try again.")

def select_pages() -> list[str] | None:
    while True:
        print("\n=== PAGE SELECTION ===")
        print("1. One page")
        print("2. All pages")
        print("0. Quit")
        choice = input("Select: ").strip()
        if choice == "0":
            return None
        if choice == "2":
            return ["--all"]
        if choice == "1":
            while True:
                page = input("Page number (0 to quit): ").strip()
                if page == "0":
                    return None
                if page.isdigit() and int(page) > 0:
                    return ["--page", page]
                print("Enter a positive page number.")
        else:
            print("Invalid selection.")


def register_source(pdf: Path, sequence: int) -> int:
    metadata = inspect_source_file(pdf)
    with get_connection() as connection:
        existing = get_source_file_by_sha256(connection, metadata["sha256"])
    if existing is not None:
        print(f"\n[DB] Reusing source_file_id={existing[0]}")
        return existing[0]
    run_module("scripts.ingest_source", pdf, "--sequence", sequence)
    with get_connection() as connection:
        existing = get_source_file_by_sha256(connection, metadata["sha256"])
    if existing is None:
        raise RuntimeError("Source registration failed")
    return existing[0]


def run_paddle(source_id: int, page_args: list[str]) -> None:
    print("\n=== PADDLEOCR ===")
    run_module("scripts.run_paddle", "--source-file", source_id, *page_args)
    print("\n=== DATABASE INGESTION ===")
    run_module("scripts.ingest_ocr", "--source-file", source_id, *page_args)


def select_corrections() -> set[str] | None:
    while True:
        print("\n=== OCR ERROR CORRECTION ===")
        for key, (label, _) in CORRECTIONS.items():
            print(f"{key}. {label}")
        print("0. Quit without corrections")
        selection = input("Select one or more (e.g. 1,2): ").strip()
        if selection == "0":
            return None
        choices = [part.strip() for part in selection.split(",")]
        if not choices or any(choice not in CORRECTIONS for choice in choices):
            print("Invalid selection. Use numbers separated by commas.")
            continue
        selected = {CORRECTIONS[choice][1] for choice in choices}
        unavailable = selected - {"pattern"}
        if unavailable:
            print("Textual and vision GenAI are not connected yet.")
            print("For this demonstration, choose 1 (pattern-based).")
            continue
        return selected


def run_corrections(source_id: int, selected: set[str]) -> None:
    if "pattern" in selected:
        print("\n=== PATTERN-BASED CORRECTION CANDIDATES ===")
        run_module(
            "scripts.generate_correction_candidates",
            "--source-file", source_id,
        )

def select_review() -> bool:
    print("\n=== HUMAN CORRECTION REVIEW ===")
    print("1. Review correction candidates")
    print("0. Quit demonstration")

    while True:
        choice = input("\nSelect: ").strip()

        if choice == "1":
            return True

        if choice == "0":
            return False

        print("Invalid selection.")

def run_review(source_id: int, page_args: list[str]) -> None:
    print("\n=== HUMAN CORRECTION REVIEW ===")

    run_module(
        "scripts.review_corrections_demo",
        "--source-file",
        source_id,
        *([] if "--all" in page_args else page_args),
    )

def show_summary(source_id: int) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(DISTINCT o.ocr_run_id),
                       COUNT(DISTINCT r.region_id), COUNT(t.token_id)
                FROM page p
                LEFT JOIN ocr_run o ON o.page_id = p.page_id
                LEFT JOIN ocr_region r ON r.ocr_run_id = o.ocr_run_id
                LEFT JOIN token t ON t.region_id = r.region_id
                WHERE p.source_file_id = %s
                """,
                (source_id,),
            )
            runs, regions, tokens = cursor.fetchone()
            print(f"\n[DB] OCR runs: {runs}; regions: {regions}; tokens: {tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-minute OCR demonstration")
    parser.add_argument("--sequence", type=int, default=1)
    args = parser.parse_args()
    if args.sequence < 1:
        parser.error("--sequence must be positive")
    if os.environ.get("DB_NAME") != "digital_history_demo":
        parser.error("Set DB_NAME=digital_history_demo before running the demo")

    print("\n=== HISTORICAL DOCUMENT OCR DEMONSTRATION ===")
    if select_engine() is None:
        return
    pdf = select_pdf()
    if pdf is None:
        return
    page_args = select_pages()
    if page_args is None:
        return

    source_id = register_source(pdf, args.sequence)
    print(f"\n[DB] Source ID: {source_id}")
    run_paddle(source_id, page_args)
    show_summary(source_id)

    selected = select_corrections()
    if selected is None:
        print("\nDemonstration ended after OCR ingestion.")
        return
    run_corrections(source_id, selected)
    if select_review():
        run_review(source_id, page_args)
    print("\n=== DEMONSTRATION COMPLETE ===")
    print("Correction candidates are suggestions; raw OCR tokens remain unchanged.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nDemonstration cancelled.")
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        print(f"\nStage failed (exit code {exc.returncode}).", file=sys.stderr)
        sys.exit(exc.returncode)
