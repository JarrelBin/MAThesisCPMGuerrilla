from src.database import get_connection


def get_report_tokens(
    report_id: int,
    *,
    include_punctuation: bool = True,
) -> list[str]:
    """
    Return the effective analytical tokens for one report
    in logical reading order.

    Accepted corrections are automatically used through
    the analytical_token view.
    """

    conditions = [
        "p.report_id = %s",
    ]

    if not include_punctuation:
        conditions.append(
            "a.token_type != 'punctuation'"
        )

    where_clause = " AND ".join(
        conditions
    )

    query = f"""
        SELECT
            a.token
        FROM analytical_token a

        JOIN ocr_region rg
            ON a.region_id = rg.region_id

        JOIN ocr_run o
            ON rg.ocr_run_id = o.ocr_run_id

        JOIN page p
            ON o.page_id = p.page_id

        WHERE {where_clause}

        ORDER BY
            p.report_page_number,
            a.position_in_page;
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (report_id,),
            )

            return [
                row[0]
                for row in cursor.fetchall()
            ]


def get_report_text(
    report_id: int,
    *,
    include_punctuation: bool = False,
) -> str:
    """
    Reconstruct one report as analytical text.
    """

    tokens = get_report_tokens(
        report_id,
        include_punctuation=include_punctuation,
    )

    return " ".join(tokens)
    
def get_all_reports(
    *,
    corpus_id: int = 1,
    include_punctuation: bool = False,
) -> list[dict]:
    """
    Return every report in corpus order.

    Each dictionary contains:
        report_id
        report_sequence
        report_ref
        tokens
        text
    """

    token_filter = ""

    if not include_punctuation:
        token_filter = (
            "AND a.token_type != 'punctuation'"
        )

    query = f"""
        SELECT
            r.report_id,
            r.report_sequence,
            COALESCE(
                r.report_ref,
                r.report_ref_raw
            ) AS report_ref,
            a.token

        FROM report r

        JOIN page p
            ON p.report_id = r.report_id

        JOIN ocr_run o
            ON o.page_id = p.page_id

        JOIN ocr_region rg
            ON rg.ocr_run_id = o.ocr_run_id

        JOIN analytical_token a
            ON a.region_id = rg.region_id

        WHERE
            r.corpus_id = %s
            {token_filter}

        ORDER BY
            r.report_sequence,
            p.report_page_number,
            a.position_in_page;
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (corpus_id,),
            )

            rows = cursor.fetchall()

    reports = {}

    for (
        report_id,
        report_sequence,
        report_ref,
        token,
    ) in rows:

        if report_id not in reports:

            reports[report_id] = {
                "report_id": report_id,
                "report_sequence":
                    report_sequence,
                "report_ref": report_ref,
                "tokens": [],
            }

        reports[
            report_id
        ]["tokens"].append(token)

    result = list(
        reports.values()
    )

    for report in result:

        report["text"] = " ".join(
            report["tokens"]
        )

    return result
