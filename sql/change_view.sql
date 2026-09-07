DROP VIEW IF EXISTS analytical_token;

CREATE VIEW analytical_token AS

SELECT
    -- --------------------------------------------------------
    -- Token identity
    -- --------------------------------------------------------

    t.token_id,
    t.region_id,

    -- Immutable PaddleOCR token
    t.raw_token,

    -- Effective analytical token
    COALESCE(
        c.corrected_token,
        t.raw_token
    ) AS token,

    t.token_type,

    -- --------------------------------------------------------
    -- Token ordering
    -- --------------------------------------------------------

    t.position_in_region,
    t.position_in_page,

    -- --------------------------------------------------------
    -- Physical source structure
    -- --------------------------------------------------------

    p.page_id,
    p.source_file_id,
    p.source_page_number,
    p.corpus_page_number,

    -- --------------------------------------------------------
    -- Historical report structure
    -- --------------------------------------------------------

    p.report_id,
    r.report_sequence,
    p.report_page_number,

    r.report_ref_raw,
    r.report_ref,

    -- --------------------------------------------------------
    -- Correction provenance
    -- --------------------------------------------------------

    t.correction_reviewed,

    (c.correction_id IS NOT NULL)
        AS was_corrected,

    c.correction_id,
    c.method AS correction_method

FROM token t

JOIN ocr_region rg
    ON t.region_id = rg.region_id

JOIN ocr_run o
    ON rg.ocr_run_id = o.ocr_run_id

JOIN page p
    ON o.page_id = p.page_id

LEFT JOIN report r
    ON p.report_id = r.report_id

LEFT JOIN token_correction c
    ON c.token_id = t.token_id
    AND c.accepted = TRUE;
