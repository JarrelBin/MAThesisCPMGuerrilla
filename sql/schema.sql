-- ============================================================
-- Defector Reports Digital History Database
-- Schema v1
-- ============================================================


-- ============================================================
-- DEVELOPMENT RESET
--
-- Remove this section once the database contains research data
-- that must be preserved.
-- ============================================================

DROP TABLE IF EXISTS token_correction CASCADE;
DROP TABLE IF EXISTS token CASCADE;
DROP TABLE IF EXISTS ocr_region CASCADE;
DROP TABLE IF EXISTS ocr_run CASCADE;
DROP TABLE IF EXISTS page CASCADE;
DROP TABLE IF EXISTS report CASCADE;
DROP TABLE IF EXISTS source_file CASCADE;
DROP TABLE IF EXISTS corpus CASCADE;
DROP TABLE IF EXISTS document CASCADE;


-- ============================================================
-- CORPUS
--
-- The complete Defector Reports collection.
-- The four PDFs together form one corpus.
-- ============================================================

CREATE TABLE corpus (
    corpus_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    name TEXT NOT NULL UNIQUE,

    description TEXT,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- SOURCE FILE
--
-- One physical archival PDF.
--
-- The four PDFs are containers and are NOT equivalent to
-- historical reports.
-- ============================================================

CREATE TABLE source_file (
    source_file_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    corpus_id BIGINT NOT NULL
        REFERENCES corpus(corpus_id)
        ON DELETE CASCADE,

    filename TEXT NOT NULL,

    source_path TEXT NOT NULL,

    sha256 CHAR(64) NOT NULL UNIQUE,

    page_count INTEGER
        CHECK (
            page_count IS NULL
            OR page_count > 0
        ),

    -- Physical ordering of the four PDFs.
    file_sequence INTEGER NOT NULL
        CHECK (file_sequence > 0),

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (
        corpus_id,
        file_sequence
    )
);


-- ============================================================
-- REPORT
--
-- One logical/historical defector report.
--
-- report_id:
--     Permanent internal PostgreSQL identifier.
--     NEVER renumber this.
--
-- report_sequence:
--     Position of the report in the corpus.
--     May be recalculated after split/merge operations.
--
-- report_ref_raw:
--     Reference extracted from OCR.
--
-- report_ref:
--     Verified/corrected historical reference.
-- ============================================================

CREATE TABLE report (
    report_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    corpus_id BIGINT NOT NULL
        REFERENCES corpus(corpus_id)
        ON DELETE CASCADE,

    report_sequence INTEGER
        CHECK (
            report_sequence IS NULL
            OR report_sequence > 0
        ),

    report_ref_raw TEXT,

    report_ref TEXT,

    title TEXT,

    report_date DATE,

    notes TEXT,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- PAGE
--
-- Critical bridge between:
--
--     physical structure → source_file
--     historical structure → report
--
-- report_id is intentionally nullable.
--
-- Pages such as archival covers/front matter can exist before
-- being assigned to a historical report.
--
-- A page can later be reassigned to another report without
-- changing OCR regions or tokens.
-- ============================================================

CREATE TABLE page (
    page_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    source_file_id BIGINT NOT NULL
        REFERENCES source_file(source_file_id)
        ON DELETE CASCADE,

    report_id BIGINT
        REFERENCES report(report_id)
        ON DELETE SET NULL,

    -- Page number inside its physical PDF.
    source_page_number INTEGER NOT NULL
        CHECK (source_page_number > 0),

    -- Sequential page number across all four PDFs.
    corpus_page_number INTEGER NOT NULL
        CHECK (corpus_page_number > 0),

    -- Logical page number inside a historical report.
    --
    -- Can be recalculated after split/merge.
    report_page_number INTEGER
        CHECK (
            report_page_number IS NULL
            OR report_page_number > 0
        ),

    image_path TEXT NOT NULL,

    width INTEGER
        CHECK (
            width IS NULL
            OR width > 0
        ),

    height INTEGER
        CHECK (
            height IS NULL
            OR height > 0
        ),

    dpi INTEGER
        CHECK (
            dpi IS NULL
            OR dpi > 0
        ),

    page_type TEXT NOT NULL
        DEFAULT 'unknown'
        CHECK (
            page_type IN (
                'report',
                'front_matter',
                'unknown'
            )
        ),

    is_report_front_page BOOLEAN NOT NULL
        DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    -- A physical PDF cannot contain the same physical page twice.
    UNIQUE (
        source_file_id,
        source_page_number
    ),

    -- Corpus-wide physical ordering must be unique.
    UNIQUE (
        corpus_page_number
    )
);


-- ============================================================
-- OCR RUN
--
-- A particular OCR operation performed on a page.
--
-- Allows OCR to be rerun later without destroying provenance.
-- ============================================================

CREATE TABLE ocr_run (
    ocr_run_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    page_id BIGINT NOT NULL
        REFERENCES page(page_id)
        ON DELETE CASCADE,

    engine TEXT NOT NULL,

    engine_version TEXT,

    framework_version TEXT,

    model_name TEXT,

    model_version TEXT,

    language TEXT,

    parameters JSONB NOT NULL
        DEFAULT '{}'::jsonb,

    raw_output_path TEXT,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- OCR REGION
--
-- PaddleOCR detected text region.
--
-- raw_text is immutable OCR output.
-- ============================================================

CREATE TABLE ocr_region (
    region_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    ocr_run_id BIGINT NOT NULL
        REFERENCES ocr_run(ocr_run_id)
        ON DELETE CASCADE,

    region_index INTEGER NOT NULL
        CHECK (region_index > 0),

    raw_text TEXT NOT NULL,

    confidence DOUBLE PRECISION
        CHECK (
            confidence IS NULL
            OR confidence BETWEEN 0 AND 1
        ),

    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,

    polygon JSONB,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (
        ocr_run_id,
        region_index
    ),

    CHECK (
        x_min IS NULL
        OR x_max IS NULL
        OR x_min <= x_max
    ),

    CHECK (
        y_min IS NULL
        OR y_max IS NULL
        OR y_min <= y_max
    )
);


-- ============================================================
-- TOKEN
--
-- Analytical token generated from raw PaddleOCR output.
--
-- Stop words are retained.
-- Raw OCR spelling and capitalization are retained.
--
-- token_type:
--     word
--     number
--     punctuation
-- ============================================================

CREATE TABLE token (
    token_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    region_id BIGINT NOT NULL
        REFERENCES ocr_region(region_id)
        ON DELETE CASCADE,

    part_index INTEGER
        CHECK (
            part_index IS NULL
            OR part_index > 0
        ),

    position_in_region INTEGER NOT NULL
        CHECK (position_in_region > 0),

    position_in_page INTEGER NOT NULL
        CHECK (position_in_page > 0),

    raw_token TEXT NOT NULL,

    token_type TEXT NOT NULL
        CHECK (
            token_type IN (
                'word',
                'number',
                'punctuation'
            )
        ),

    correction_reviewed BOOLEAN NOT NULL
    DEFAULT FALSE,
    
    correction_reviewed_at TIMESTAMPTZ,

    -- Coordinates inherited from Paddle word/component box.
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,

    char_start INTEGER,

    char_end INTEGER,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (
        region_id,
        position_in_region
    ),

    CHECK (
        char_start IS NULL
        OR char_start >= 0
    ),

    CHECK (
        char_end IS NULL
        OR char_end >= 0
    ),

    CHECK (
        char_start IS NULL
        OR char_end IS NULL
        OR char_start <= char_end
    ),

    CHECK (
        x_min IS NULL
        OR x_max IS NULL
        OR x_min <= x_max
    ),

    CHECK (
        y_min IS NULL
        OR y_max IS NULL
        OR y_min <= y_max
    )
);


-- ============================================================
-- TOKEN CORRECTION
--
-- OCR corrections are separate from raw tokens.
--
-- raw_token is NEVER overwritten.
-- ============================================================

CREATE TABLE token_correction (
    correction_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    token_id BIGINT NOT NULL
        REFERENCES token(token_id)
        ON DELETE CASCADE,

    corrected_token TEXT NOT NULL,

    method TEXT NOT NULL,

    confidence DOUBLE PRECISION
        CHECK (
            confidence IS NULL
            OR confidence BETWEEN 0 AND 1
        ),

    accepted BOOLEAN NOT NULL
        DEFAULT FALSE,

    notes TEXT,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX idx_source_file_corpus
    ON source_file(corpus_id);


CREATE INDEX idx_report_corpus
    ON report(corpus_id);


CREATE INDEX idx_report_sequence
    ON report(
        corpus_id,
        report_sequence
    );


CREATE INDEX idx_report_ref
    ON report(report_ref);


CREATE INDEX idx_page_source_file
    ON page(source_file_id);


CREATE INDEX idx_page_report
    ON page(report_id);


CREATE INDEX idx_page_report_number
    ON page(
        report_id,
        report_page_number
    );


CREATE INDEX idx_page_corpus_number
    ON page(corpus_page_number);


CREATE INDEX idx_ocr_run_page
    ON ocr_run(page_id);


CREATE INDEX idx_ocr_region_run
    ON ocr_region(ocr_run_id);


CREATE INDEX idx_token_region
    ON token(region_id);


CREATE INDEX idx_token_raw
    ON token(raw_token);


CREATE INDEX idx_token_type
    ON token(token_type);


CREATE INDEX idx_token_page_position
    ON token(position_in_page);


CREATE INDEX idx_correction_token
    ON token_correction(token_id);


CREATE INDEX idx_correction_accepted
    ON token_correction(token_id)
    WHERE accepted = TRUE;
    
-- ============================================================
-- ANALYTICAL TOKEN VIEW
--
-- Analysis-facing representation of corpus tokens.
--
-- Uses an accepted correction when one exists;
-- otherwise preserves the raw OCR token.
--
-- Includes report/page structure so downstream analytical
-- tools do not need to reconstruct the database hierarchy.
--
-- IMPORTANT:
-- SQL views do not guarantee row order.
-- Queries must still use ORDER BY when sequence matters.
-- ============================================================

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
