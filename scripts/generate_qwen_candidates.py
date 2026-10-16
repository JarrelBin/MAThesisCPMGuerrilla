"""Generate page-level OCR correction proposals with Qwen3-8B Q6_K.

Run: python -m scripts.generate_qwen_candidates --source-file 1 --page 4
Uses existing src.database and src.repository APIs. Never accepts corrections.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from src.database import get_connection
from src.qwen_correction import (
    DEFAULT_MODEL, DEFAULT_SERVER, PROMPT_VERSION, QwenCorrectionModel,
)
from src.repository import get_tokens_for_contextual_review, upsert_correction_candidate


def group_by_page(rows) -> list[dict]:
    pages = {}
    for (
        token_id, raw_token, token_type, page_id, source_file_id,
        source_page_number, corpus_page_number, position_in_page,
    ) in rows:
        page = pages.setdefault(page_id, {
            "page_id": page_id,
            "source_file_id": source_file_id,
            "source_page_number": source_page_number,
            "corpus_page_number": corpus_page_number,
            "tokens": [],
        })
        page["tokens"].append({
            "token_id": token_id,
            "raw_token": raw_token,
            "token_type": token_type,
            "position_in_page": position_in_page,
        })
    for page in pages.values():
        page["tokens"].sort(key=lambda t: t["position_in_page"])
    return sorted(pages.values(), key=lambda p: (p["corpus_page_number"], p["page_id"]))


# Catch clearly self-contradictory substitutions, not every weak reason.
SUBSTITUTION = re.compile(
    r"['\"]([^'\"]+)['\"]\s+(?:instead of|replaced by|rather than)\s+['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
GENERIC_REASONS = {
    "possible confusion with similar-looking characters",
    "potential character substitution",
    "possible missing or substituted character",
    "possible confusion with similar-looking characters in word",
}


def validate_candidate(item: object, token_map: dict, seen: set) -> tuple[dict | None, str | None]:
    if not isinstance(item, dict):
        return None, "candidate is not an object"
    token_id = item.get("token_id")
    if isinstance(token_id, bool):
        return None, "boolean token_id"
    try:
        token_id = int(token_id)
    except (ValueError, TypeError):
        return None, "invalid token_id"
    token = token_map.get(token_id)
    if token is None:
        return None, "token_id outside page"
    if token_id in seen:
        return None, "duplicate token_id"
    raw = item.get("raw_token")
    suggestion = item.get("suggested_token")
    reason = item.get("reason")
    if not isinstance(raw, str) or raw != token["raw_token"]:
        return None, "raw_token mismatch"
    if not isinstance(suggestion, str) or not suggestion.strip():
        return None, "missing replacement"
    # Keep one-token proposals; punctuation-only or multi-token corrections
    # need a separate alignment-aware workflow.
    suggestion = suggestion.strip()
    if any(c.isspace() for c in suggestion):
        return None, "multi-token replacement requires separate workflow"
    if suggestion == raw:
        return None, "replacement identical to OCR token"
    if not isinstance(reason, str) or not reason.strip():
        return None, "missing explanation"
    reason = reason.strip()
    if reason.casefold() in GENERIC_REASONS:
        return None, "generic unsupported explanation"
    match = SUBSTITUTION.search(reason)
    if match and match.group(1) == match.group(2):
        return None, "contradictory substitution explanation"
    seen.add(token_id)
    return {
        "token_id": token_id,
        "raw_token": raw,
        "suggested_token": suggestion,
        "reason": reason,
    }, None


def write_audit(directory: Path, page: dict, record: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"source_{page['source_file_id']}_page_{page['source_page_number']:04d}.jsonl"
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-8B Q6_K page-level OCR correction proposals")
    parser.add_argument("--source-file", type=int)
    parser.add_argument("--page", type=int, help="Requires --source-file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server-url", default=DEFAULT_SERVER)
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Audit proposals without database writes")
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("data/experiments/qwen_q6_page"),
        help="Append raw model responses and validation outcomes to per-page JSONL files",
    )
    args = parser.parse_args()
    if args.page is not None and args.source_file is None:
        parser.error("--page requires --source-file")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be >= 1")

    with get_connection() as connection:
        rows = get_tokens_for_contextual_review(connection, source_file_id=args.source_file)
    pages = group_by_page(rows)
    if args.page is not None:
        pages = [p for p in pages if p["source_page_number"] == args.page]
    if args.max_pages is not None:
        pages = pages[:args.max_pages]
    if not pages:
        print("No eligible pages/tokens found.")
        return

    print("\n=== QWEN Q6_K PAGE-LEVEL OCR CORRECTION PROPOSALS ===")
    print(f"Model: {args.model} | server: {args.server_url}")
    print(f"Pages: {len(pages)} | context: {args.context_size} | output: {args.max_new_tokens}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'STORE PROPOSALS'}")
    print("No existing candidates will be deleted. Upserts MAY replace prior qwen proposals.\n")
    qwen = QwenCorrectionModel(
        model_name=args.model,
        server_url=args.server_url,
        context_size=args.context_size,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )
    returned = rejected = stored = failed = 0
    for i, page in enumerate(pages, 1):
        tokens = page["tokens"]
        page_no = page["source_page_number"]
        print(f"Page {page_no:04d} ({i}/{len(pages)}): {len(tokens)} tokens", flush=True)
        start = time.perf_counter()
        audit = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_file_id": page["source_file_id"],
            "page_id": page["page_id"],
            "source_page_number": page_no,
            "page_token_count": len(tokens),
            "model": args.model,
            "quantization": "Q6_K",
            "backend": "llama.cpp",
            "prompt_version": PROMPT_VERSION,
            "dry_run": args.dry_run,
        }
        try:
            suggestions = qwen.analyze_page(tokens)
            audit["model_response"] = qwen.last_response
            audit["returned_candidates"] = suggestions
            returned += len(suggestions)
            token_map = {t["token_id"]: t for t in tokens}
            seen = set()
            valid, invalid = [], []
            for item in suggestions:
                candidate, reason = validate_candidate(item, token_map, seen)
                if reason:
                    rejected += 1
                    invalid.append({"candidate": item, "rejection_reason": reason})
                    print(f"  REJECTED: {reason} | {item!r}")
                else:
                    valid.append(candidate)
            audit["validated_candidates"] = valid
            audit["rejected_candidates"] = invalid
            if not args.dry_run:
                # Atomic per-page transaction; never mark corrections accepted.
                with get_connection() as connection:
                    for item in valid:
                        upsert_correction_candidate(
                            connection,
                            token_id=item["token_id"],
                            method="qwen",
                            suggested_token=item["suggested_token"],
                            details={
                                "reason": item["reason"],
                                "model": args.model,
                                "backend": "llama.cpp",
                                "quantization": "Q6_K",
                                "prompt_version": PROMPT_VERSION,
                                "analysis_unit": "page",
                                "source_page_number": page_no,
                                "page_token_count": len(tokens),
                                "context_size": args.context_size,
                                "max_new_tokens": args.max_new_tokens,
                            },
                        )
                    connection.commit()
                stored += len(valid)
            for item in valid:
                print(f"  PROPOSED [{item['token_id']}] {item['raw_token']!r} -> {item['suggested_token']!r}: {item['reason']}")
            audit["status"] = "ok"
            print(f"  Completed in {time.perf_counter() - start:.1f}s | {len(valid)} valid, {len(invalid)} rejected")
        except Exception as exc:
            failed += 1
            audit["status"] = "failed"
            audit["error"] = str(exc)
            audit["model_response"] = qwen.last_response
            print(f"  FAILED: {exc}", flush=True)
        finally:
            audit["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            try:
                path = write_audit(args.audit_dir, page, audit)
                print(f"  Audit: {path}")
            except OSError as exc:
                print(f"  WARNING: audit write failed: {exc}")
    print("\n--- Qwen Summary ---")
    print(f"Pages attempted: {len(pages)} | failed: {failed}")
    print(f"Candidates returned: {returned} | rejected: {rejected} | stored: {stored}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
