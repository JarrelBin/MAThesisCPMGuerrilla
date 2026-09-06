from pathlib import Path

from src.database import get_connection

import json

def get_ocr_run(
    connection,
    *,
    page_id: int,
    engine: str,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT ocr_run_id
            FROM ocr_run
            WHERE page_id = %s
              AND engine = %s
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (
                page_id,
                engine,
            ),
        )

        result = cursor.fetchone()

        return result[0] if result else None


def insert_ocr_run(
    connection,
    *,
    page_id: int,
    engine: str,
    engine_version: str | None,
    framework_version: str | None,
    model_name: str | None,
    language: str,
    parameters: dict,
    raw_output_path: str,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ocr_run (
                page_id,
                engine,
                engine_version,
                framework_version,
                model_name,
                language,
                parameters,
                raw_output_path
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING ocr_run_id;
            """,
            (
                page_id,
                engine,
                engine_version,
                framework_version,
                model_name,
                language,
                json.dumps(parameters),
                raw_output_path,
            ),
        )

        return cursor.fetchone()[0]


def insert_ocr_region(
    connection,
    *,
    ocr_run_id: int,
    region: dict,
):
    bbox = region["bbox"]

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ocr_region (
                ocr_run_id,
                region_index,
                raw_text,
                confidence,
                x_min,
                y_min,
                x_max,
                y_max,
                polygon
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            RETURNING region_id;
            """,
            (
                ocr_run_id,
                region["region_index"],
                region["raw_text"],
                region["confidence"],
                bbox["x_min"],
                bbox["y_min"],
                bbox["x_max"],
                bbox["y_max"],
                json.dumps(
                    region["polygon"]
                ),
            ),
        )

        return cursor.fetchone()[0]


def insert_token(
    connection,
    *,
    region_id: int,
    token: dict,
):
    bbox = token["source_bbox"]

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO token (
                region_id,
                part_index,
                position_in_region,
                position_in_page,
                raw_token,
                token_type,
                x_min,
                y_min,
                x_max,
                y_max,
                char_start,
                char_end
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
            RETURNING token_id;
            """,
            (
                region_id,
                token["part_index"],
                token["position_in_region"],
                token["position_in_page"],
                token["raw_token"],
                token["token_type"],
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
                token["char_start"],
                token["char_end"],
            ),
        )

        return cursor.fetchone()[0]

def get_corpus_by_name(
    connection,
    name: str,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT corpus_id, name
            FROM corpus
            WHERE name = %s;
            """,
            (name,),
        )

        return cursor.fetchone()


def get_source_file_by_sha256(
    connection,
    sha256: str,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                source_file_id,
                corpus_id,
                filename,
                source_path,
                sha256,
                page_count,
                file_sequence
            FROM source_file
            WHERE sha256 = %s;
            """,
            (sha256,),
        )

        return cursor.fetchone()


def insert_source_file(
    connection,
    *,
    corpus_id: int,
    filename: str,
    source_path: str,
    sha256: str,
    page_count: int,
    file_sequence: int,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO source_file (
                corpus_id,
                filename,
                source_path,
                sha256,
                page_count,
                file_sequence
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING source_file_id;
            """,
            (
                corpus_id,
                filename,
                source_path,
                sha256,
                page_count,
                file_sequence,
            ),
        )

        return cursor.fetchone()[0]


def insert_page(
    connection,
    *,
    source_file_id: int,
    source_page_number: int,
    corpus_page_number: int,
    image_path: str,
    width: int,
    height: int,
    dpi: int,
):
    """
    Insert a physical page if it does not already exist.

    Report assignment is deliberately left NULL.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO page (
                source_file_id,
                report_id,
                source_page_number,
                corpus_page_number,
                report_page_number,
                image_path,
                width,
                height,
                dpi,
                page_type,
                is_report_front_page
            )
            VALUES (
                %s,
                NULL,
                %s,
                %s,
                NULL,
                %s,
                %s,
                %s,
                %s,
                'unknown',
                FALSE
            )
            ON CONFLICT (
                source_file_id,
                source_page_number
            )
            DO NOTHING
            RETURNING page_id;
            """,
            (
                source_file_id,
                source_page_number,
                corpus_page_number,
                image_path,
                width,
                height,
                dpi,
            ),
        )

        result = cursor.fetchone()

        if result is None:
            return None

        return result[0]

def get_page(
    connection,
    *,
    source_file_id: int,
    source_page_number: int,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                page_id,
                corpus_page_number,
                image_path
            FROM page
            WHERE source_file_id = %s
              AND source_page_number = %s;
            """,
            (
                source_file_id,
                source_page_number,
            ),
        )

        return cursor.fetchone()
        
def recalculate_corpus_page_numbers(
    connection,
):
    """
    Recalculate corpus-wide page numbers according to:

        source_file.file_sequence
        then
        page.source_page_number

    This allows source PDFs to be ingested in any order.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                p.page_id
            FROM page p
            JOIN source_file sf
                ON p.source_file_id = sf.source_file_id
            ORDER BY
                sf.file_sequence,
                p.source_page_number;
            """
        )

        page_ids = [
            row[0]
            for row in cursor.fetchall()
        ]

        # Temporarily move numbers outside the normal
        # range to avoid UNIQUE conflicts.
        cursor.execute(
            """
            UPDATE page
            SET corpus_page_number =
                corpus_page_number + 1000000000;
            """
        )

        for corpus_page_number, page_id in enumerate(
            page_ids,
            start=1,
        ):
            cursor.execute(
                """
                UPDATE page
                SET corpus_page_number = %s
                WHERE page_id = %s;
                """,
                (
                    corpus_page_number,
                    page_id,
                ),
            )

def insert_report(
    connection,
    *,
    corpus_id: int,
    report_ref_raw: str | None,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO report (
                corpus_id,
                report_ref_raw
            )
            VALUES (%s, %s)
            RETURNING report_id;
            """,
            (
                corpus_id,
                report_ref_raw,
            ),
        )

        return cursor.fetchone()[0]


def assign_page_to_report(
    connection,
    *,
    page_id: int,
    report_id: int,
    report_page_number: int,
    is_front_page: bool,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE page
            SET
                report_id = %s,
                report_page_number = %s,
                page_type = 'report',
                is_report_front_page = %s
            WHERE page_id = %s;
            """,
            (
                report_id,
                report_page_number,
                is_front_page,
                page_id,
            ),
        )


def recalculate_report_sequences(
    connection,
    *,
    corpus_id: int,
):
    """
    Recalculate report_sequence according to the
    earliest corpus page assigned to each report.

    report_id never changes.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                r.report_id,
                MIN(p.corpus_page_number)
                    AS first_page
            FROM report r
            JOIN page p
                ON p.report_id = r.report_id
            WHERE r.corpus_id = %s
            GROUP BY r.report_id
            ORDER BY first_page;
            """,
            (corpus_id,),
        )

        report_ids = [
            row[0]
            for row in cursor.fetchall()
        ]

        for sequence, report_id in enumerate(
            report_ids,
            start=1,
        ):
            cursor.execute(
                """
                UPDATE report
                SET report_sequence = %s
                WHERE report_id = %s;
                """,
                (
                    sequence,
                    report_id,
                ),
            )
            
def recalculate_report_page_numbers(
    connection,
    *,
    report_id: int,
):
    """
    Recalculate report_page_number according to
    physical corpus page order.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT page_id
            FROM page
            WHERE report_id = %s
            ORDER BY corpus_page_number;
            """,
            (report_id,),
        )

        page_ids = [
            row[0]
            for row in cursor.fetchall()
        ]

        for report_page_number, page_id in enumerate(
            page_ids,
            start=1,
        ):
            cursor.execute(
                """
                UPDATE page
                SET report_page_number = %s
                WHERE page_id = %s;
                """,
                (
                    report_page_number,
                    page_id,
                ),
            )


def recalculate_report_front_pages(
    connection,
    *,
    report_id: int,
):
    """
    Mark only the first physical page of a report
    as its front page.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            UPDATE page
            SET is_report_front_page = FALSE
            WHERE report_id = %s;
            """,
            (report_id,),
        )

        cursor.execute(
            """
            UPDATE page
            SET is_report_front_page = TRUE
            WHERE page_id = (
                SELECT page_id
                FROM page
                WHERE report_id = %s
                ORDER BY corpus_page_number
                LIMIT 1
            );
            """,
            (report_id,),
        )
        
def split_report(
    connection,
    *,
    report_id: int,
    split_page_id: int,
    corpus_id: int,
    report_ref_raw: str | None = None,
):
    """
    Split an existing report immediately before
    split_page_id.

    The split page and every subsequent page currently
    belonging to the old report are assigned to a newly
    created report.

    Returns the new report_id.
    """

    with connection.cursor() as cursor:

        # Find physical position of split page.
        cursor.execute(
            """
            SELECT corpus_page_number
            FROM page
            WHERE page_id = %s
              AND report_id = %s;
            """,
            (
                split_page_id,
                report_id,
            ),
        )

        result = cursor.fetchone()

        if result is None:
            raise ValueError(
                "Split page does not belong "
                "to the specified report."
            )

        split_corpus_page = result[0]

        # Don't allow splitting before the first page.
        cursor.execute(
            """
            SELECT MIN(corpus_page_number)
            FROM page
            WHERE report_id = %s;
            """,
            (report_id,),
        )

        first_page = cursor.fetchone()[0]

        if split_corpus_page == first_page:
            raise ValueError(
                "Cannot split before the first page "
                "of a report."
            )

    # Create new permanent report.
    new_report_id = insert_report(
        connection,
        corpus_id=corpus_id,
        report_ref_raw=report_ref_raw,
    )

    # Move the split page and everything after it
    # from the OLD report into the new report.
    with connection.cursor() as cursor:

        cursor.execute(
            """
            UPDATE page
            SET report_id = %s
            WHERE report_id = %s
              AND corpus_page_number >= %s;
            """,
            (
                new_report_id,
                report_id,
                split_corpus_page,
            ),
        )

    # Rebuild logical numbering.
    recalculate_report_page_numbers(
        connection,
        report_id=report_id,
    )

    recalculate_report_page_numbers(
        connection,
        report_id=new_report_id,
    )

    recalculate_report_front_pages(
        connection,
        report_id=report_id,
    )

    recalculate_report_front_pages(
        connection,
        report_id=new_report_id,
    )

    recalculate_report_sequences(
        connection,
        corpus_id=corpus_id,
    )

    return new_report_id
    
def merge_reports(
    connection,
    *,
    keep_report_id: int,
    merge_report_id: int,
    corpus_id: int,
):
    """
    Merge merge_report_id into keep_report_id.

    All pages belonging to merge_report_id are reassigned
    to keep_report_id. The empty report is then deleted.

    Permanent IDs of all other reports remain unchanged.
    """

    if keep_report_id == merge_report_id:
        raise ValueError(
            "Cannot merge a report into itself."
        )

    with connection.cursor() as cursor:

        # ---------------------------------
        # Verify both reports exist and
        # belong to the same corpus.
        # ---------------------------------

        cursor.execute(
            """
            SELECT report_id
            FROM report
            WHERE corpus_id = %s
              AND report_id IN (%s, %s);
            """,
            (
                corpus_id,
                keep_report_id,
                merge_report_id,
            ),
        )

        found = {
            row[0]
            for row in cursor.fetchall()
        }

        expected = {
            keep_report_id,
            merge_report_id,
        }

        if found != expected:
            raise ValueError(
                "Both reports must exist and "
                "belong to the same corpus."
            )
        
        # ---------------------------------
        # Will not accidentally merge two
        # reports that are nowhere
        # near each other.
        # ---------------------------------
        
        cursor.execute(
            """
            SELECT
                report_id,
                MIN(corpus_page_number),
                MAX(corpus_page_number)
            FROM page
            WHERE report_id IN (%s, %s)
            GROUP BY report_id;
            """,
            (
                keep_report_id,
                merge_report_id,
            ),
        )

        ranges = {
            row[0]: (row[1], row[2])
            for row in cursor.fetchall()
        }

        keep_range = ranges.get(
            keep_report_id
        )

        merge_range = ranges.get(
            merge_report_id
        )

        if (
            keep_range is None
            or merge_range is None
        ):
            raise ValueError(
                "Both reports must contain pages."
            )

        are_adjacent = (
            keep_range[1] + 1
            == merge_range[0]
            or
            merge_range[1] + 1
            == keep_range[0]
        )

        if not are_adjacent:
            raise ValueError(
                "Reports are not physically adjacent. "
                "Refusing automatic merge."
            )
            
        # ---------------------------------
        # Move pages
        # ---------------------------------

        cursor.execute(
            """
            UPDATE page
            SET report_id = %s
            WHERE report_id = %s;
            """,
            (
                keep_report_id,
                merge_report_id,
            ),
        )

        # ---------------------------------
        # Delete now-empty report
        # ---------------------------------

        cursor.execute(
            """
            DELETE FROM report
            WHERE report_id = %s;
            """,
            (merge_report_id,),
        )

    # ---------------------------------
    # Rebuild logical numbering
    # ---------------------------------

    recalculate_report_page_numbers(
        connection,
        report_id=keep_report_id,
    )

    recalculate_report_front_pages(
        connection,
        report_id=keep_report_id,
    )

    recalculate_report_sequences(
        connection,
        corpus_id=corpus_id,
    )

def get_source_file(
    connection,
    *,
    source_file_id: int,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                source_file_id,
                corpus_id,
                filename,
                source_path,
                file_sequence,
                page_count
            FROM source_file
            WHERE source_file_id = %s;
            """,
            (source_file_id,),
        )

        return cursor.fetchone()

def get_pages_for_source_file(
    connection,
    *,
    source_file_id: int,
):
    """
    Return all physical pages belonging to a source file
    in source-page order.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                page_id,
                source_page_number,
                corpus_page_number,
                image_path
            FROM page
            WHERE source_file_id = %s
            ORDER BY source_page_number;
            """,
            (source_file_id,),
        )

        return cursor.fetchall()
        
def mark_token_reviewed(
    connection,
    *,
    token_id: int,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE token
            SET
                correction_reviewed = TRUE,
                correction_reviewed_at =
                    CURRENT_TIMESTAMP
            WHERE token_id = %s;
            """,
            (token_id,),
        )


def insert_token_correction(
    connection,
    *,
    token_id: int,
    corrected_token: str,
    method: str = "human",
    confidence: float | None = None,
    notes: str | None = None,
):
    """
    Insert an accepted correction.

    raw_token is never modified.
    """

    with connection.cursor() as cursor:

        # Ensure only one currently accepted correction
        # exists for this token.
        cursor.execute(
            """
            UPDATE token_correction
            SET accepted = FALSE
            WHERE token_id = %s
              AND accepted = TRUE;
            """,
            (token_id,),
        )

        cursor.execute(
            """
            INSERT INTO token_correction (
                token_id,
                corrected_token,
                method,
                confidence,
                accepted,
                notes
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                TRUE,
                %s
            )
            RETURNING correction_id;
            """,
            (
                token_id,
                corrected_token,
                method,
                confidence,
                notes,
            ),
        )

        return cursor.fetchone()[0]
        
def get_or_create_corpus(
    connection,
    *,
    name: str,
    description: str | None = None,
):
    """
    Return the corpus_id, creating the corpus
    if it does not already exist.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            INSERT INTO corpus (
                name,
                description
            )
            VALUES (%s, %s)

            ON CONFLICT (name)
            DO UPDATE
            SET name = EXCLUDED.name

            RETURNING corpus_id;
            """,
            (
                name,
                description,
            ),
        )

        return cursor.fetchone()[0]
