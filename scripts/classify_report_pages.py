"""Classify personalities appendices after report segmentation.

Only changes page.page_type on unreviewed pages.
"""
import argparse
import json

from src.database import get_connection
from src.paths import paddle_directory
from src.repository import get_source_file
from src.personality_detection import is_personalities_start, is_appendix_b


def classify_sequence(pages):
    """Yield (page, proposed_type, reason), ordered within reports.

    Each page is a dict with report_id, page_type, reviewed, front,
    regions. An existing sketch ends the personalities section.
    """
    current_report = None
    active = False
    for page in pages:
        report_id = page['report_id']
        if report_id != current_report or page['front']:
            active = False
        current_report = report_id
        if report_id is None:
            yield page, None, 'unassigned'
            continue
        if page['reviewed'] and page['page_type'] == 'sketch':
            active = False
            yield page, None, 'reviewed sketch'
            continue
        if is_appendix_b(page['regions']):
            active = False
            yield page, None, 'appendix B'
            continue
        if is_personalities_start(page['regions']):
            active = True
        if page['reviewed']:
            yield page, None, 'human reviewed'
        elif page['page_type'] == 'sketch':
            # Keep existing sketch candidates, but do not propagate
            # personalities beyond them. Review sketches first.
            active = False
            yield page, None, 'existing sketch candidate'
        elif active:
            yield page, 'personalities', 'heading or continuation'
        else:
            yield page, None, 'no personalities evidence'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-file', type=int, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--page', type=int)
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()

    with get_connection() as connection:
        source = get_source_file(connection, source_file_id=args.source_file)
        if source is None:
            raise RuntimeError('Source file not found')
        ocr_dir = paddle_directory(source[2])
        with connection.cursor() as cursor:
            cursor.execute('''
                SELECT page_id, source_page_number, report_id, page_type,
                       page_type_reviewed, is_report_front_page
                FROM page WHERE source_file_id = %s
                ORDER BY source_page_number
            ''', (args.source_file,))
            rows = cursor.fetchall()
        if not rows:
            raise RuntimeError('No pages registered')
        if not any(row[2] is not None for row in rows):
            raise RuntimeError('No report IDs assigned: run segment_reports first')

        pages = []
        for page_id, number, report_id, page_type, reviewed, front in rows:
            path = ocr_dir / f'page_{number:04d}.json'
            with path.open('r', encoding='utf-8') as handle:
                regions = json.load(handle)
            pages.append(dict(page_id=page_id, number=number,
                              report_id=report_id, page_type=page_type,
                              reviewed=reviewed, front=front, regions=regions))

        if args.page is not None and not any(p['number'] == args.page for p in pages):
            raise RuntimeError(f'Page {args.page} not found')

        # Compute the complete sequence even for --page, so preceding
        # appendix headings remain available as context.
        with connection.cursor() as cursor:
            for page, proposed, reason in classify_sequence(pages):
                if args.page is not None and page['number'] != args.page:
                    continue
                if proposed == 'personalities':
                    cursor.execute('''
                        UPDATE page SET page_type = 'personalities'
                        WHERE page_id = %s AND page_type_reviewed = FALSE
                          AND page_type <> 'sketch'
                    ''', (page['page_id'],))
                elif (not page['reviewed'] and
                      page['page_type'] == 'personalities'):
                    # Clear stale algorithmic personalities labels only.
                    cursor.execute('''
                        UPDATE page SET page_type = 'unknown'
                        WHERE page_id = %s AND page_type = 'personalities'
                          AND page_type_reviewed = FALSE
                    ''', (page['page_id'],))
                print(f"Page {page['number']:04d} | report={page['report_id']} "
                      f"| {proposed or 'KEEP'} | {reason}")
        connection.commit()


if __name__ == '__main__':
    main()
