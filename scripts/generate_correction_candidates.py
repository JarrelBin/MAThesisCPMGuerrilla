"""Generate pattern-only OCR candidates.

Examples:
  python -m scripts.generate_correction_candidates --source-file 1
  python -m scripts.generate_correction_candidates --source-file 1 --page 4 --dry-run

Does not delete existing candidates, change accepted corrections, or overwrite
existing pattern proposals. Raw OCR remains unchanged.
"""
from __future__ import annotations

import argparse

from src.correction_candidates import get_correction_candidates
from src.database import get_connection
from src.repository import upsert_correction_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=int, help="Only this source_file_id")
    parser.add_argument("--page", type=int, help="Source page (requires --source-file)")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates without database writes")
    args = parser.parse_args()
    if args.source_file is not None and args.source_file < 1:
        parser.error("--source-file must be positive")
    if args.page is not None and (args.source_file is None or args.page < 1):
        parser.error("--page requires --source-file and must be positive")

    with get_connection() as connection:
        candidates = get_correction_candidates(
            connection,
            source_file_id=args.source_file,
            source_page_number=args.page,
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT token_id FROM correction_candidate WHERE method = 'pattern'")
            existing = {row[0] for row in cursor.fetchall()}

    print(f"Pattern candidates: {len(candidates)} | source: {args.source_file or 'all'} "
          f"| page: {args.page or 'all'} | dry-run: {args.dry_run}", flush=True)
    stored = skipped = 0
    if args.dry_run:
        for item in candidates[:20]:
            print(f"  [{item['token_id']}] {item['token']!r}: {', '.join(item['reasons'])}")
    else:
        with get_connection() as connection:
            for item in candidates:
                if item["token_id"] in existing:
                    skipped += 1
                    continue
                upsert_correction_candidate(
                    connection,
                    token_id=item["token_id"],
                    method="pattern",
                    suggested_token=None,
                    details={
                        "reasons": item["reasons"],
                        "frequency": item["frequency"],
                        "ocr_confidence": item["confidence"],
                        "source_page_number": item["page"],
                        "position_in_page": item["position"],
                        "neighbor": None,
                        "context": item["context"],
                        "detector": "pattern_only_v1",
                    },
                )
                stored += 1
            connection.commit()
    print(f"Stored: {stored} | existing skipped: {skipped}", flush=True)


if __name__ == "__main__":
    main()
