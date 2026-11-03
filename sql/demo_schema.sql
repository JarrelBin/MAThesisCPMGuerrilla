-- OCR demonstration schema. Run ONLY against the separate demo database.
-- Compatible with existing source registration, Paddle ingestion and correction repositories.
BEGIN;

-- Safety guard: refuse to reset any database other than the demo database.
DO $$
BEGIN
    IF current_database() <> 'digital_history_demo' THEN
        RAISE EXCEPTION 'Refusing reset: connected to %, expected digital_history_demo', current_database();
    END IF;
END $$;

DROP VIEW IF EXISTS analytical_token;
DROP TABLE IF EXISTS token_correction CASCADE;
DROP TABLE IF EXISTS correction_candidate CASCADE;
DROP TABLE IF EXISTS token CASCADE;
DROP TABLE IF EXISTS ocr_region CASCADE;
DROP TABLE IF EXISTS ocr_run CASCADE;
DROP TABLE IF EXISTS page CASCADE;
DROP TABLE IF EXISTS source_file CASCADE;
DROP TABLE IF EXISTS corpus CASCADE;


CREATE TABLE corpus (
    corpus_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE source_file (
    source_file_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    corpus_id BIGINT NOT NULL REFERENCES corpus(corpus_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    source_path TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL UNIQUE,
    page_count INTEGER CHECK (page_count IS NULL OR page_count > 0),
    file_sequence INTEGER NOT NULL CHECK (file_sequence > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (corpus_id, file_sequence)
);

CREATE TABLE page (
    page_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_file_id BIGINT NOT NULL REFERENCES source_file(source_file_id) ON DELETE CASCADE,
    report_id BIGINT, -- compatibility with existing insert_page(); no report table
    source_page_number INTEGER NOT NULL CHECK (source_page_number > 0),
    corpus_page_number INTEGER NOT NULL UNIQUE CHECK (corpus_page_number > 0),
    report_page_number INTEGER, -- compatibility only
    image_path TEXT NOT NULL,
    width INTEGER CHECK (width IS NULL OR width > 0),
    height INTEGER CHECK (height IS NULL OR height > 0),
    dpi INTEGER CHECK (dpi IS NULL OR dpi > 0),
    page_type TEXT NOT NULL DEFAULT 'unknown',
    page_type_reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    is_report_front_page BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_file_id, source_page_number)
);

CREATE TABLE ocr_run (
    ocr_run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    page_id BIGINT NOT NULL REFERENCES page(page_id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    engine_version TEXT,
    framework_version TEXT,
    model_name TEXT,
    model_version TEXT,
    language TEXT,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_output_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ocr_region (
    region_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ocr_run_id BIGINT NOT NULL REFERENCES ocr_run(ocr_run_id) ON DELETE CASCADE,
    region_index INTEGER NOT NULL CHECK (region_index > 0),
    raw_text TEXT NOT NULL,
    confidence DOUBLE PRECISION CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    polygon JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ocr_run_id, region_index),
    CHECK (x_min IS NULL OR x_max IS NULL OR x_min <= x_max),
    CHECK (y_min IS NULL OR y_max IS NULL OR y_min <= y_max)
);

CREATE TABLE token (
    token_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    region_id BIGINT NOT NULL REFERENCES ocr_region(region_id) ON DELETE CASCADE,
    part_index INTEGER CHECK (part_index IS NULL OR part_index > 0),
    position_in_region INTEGER NOT NULL CHECK (position_in_region > 0),
    position_in_page INTEGER NOT NULL CHECK (position_in_page > 0),
    raw_token TEXT NOT NULL,
    token_type TEXT NOT NULL CHECK (token_type IN ('word', 'number', 'punctuation')),
    correction_reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    correction_reviewed_at TIMESTAMPTZ,
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (region_id, position_in_region),
    CHECK (char_start IS NULL OR char_start >= 0),
    CHECK (char_end IS NULL OR char_end >= 0),
    CHECK (char_start IS NULL OR char_end IS NULL OR char_start <= char_end)
);

CREATE TABLE correction_candidate (
    candidate_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    token_id BIGINT NOT NULL REFERENCES token(token_id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    suggested_token TEXT,
    confidence DOUBLE PRECISION CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (token_id, method)
);

CREATE TABLE token_correction (
    correction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    token_id BIGINT NOT NULL REFERENCES token(token_id) ON DELETE CASCADE,
    corrected_token TEXT NOT NULL,
    method TEXT NOT NULL,
    confidence DOUBLE PRECISION CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    accepted BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_page_source ON page(source_file_id);
CREATE INDEX idx_ocr_run_page ON ocr_run(page_id);
CREATE INDEX idx_ocr_region_run ON ocr_region(ocr_run_id);
CREATE INDEX idx_token_region ON token(region_id);
CREATE INDEX idx_token_raw ON token(raw_token);
CREATE INDEX idx_candidate_token ON correction_candidate(token_id);
CREATE INDEX idx_candidate_method ON correction_candidate(method);
CREATE UNIQUE INDEX idx_one_accepted_correction ON token_correction(token_id) WHERE accepted = TRUE;

CREATE VIEW analytical_token AS
SELECT t.token_id, t.region_id, t.raw_token,
       COALESCE(c.corrected_token, t.raw_token) AS token,
       t.token_type, t.position_in_region, t.position_in_page,
       p.page_id, p.source_file_id, p.source_page_number,
       p.corpus_page_number, p.page_type,
       o.ocr_run_id, o.engine, o.model_name,
       t.correction_reviewed,
       (c.correction_id IS NOT NULL) AS was_corrected,
       c.correction_id, c.method AS correction_method
FROM token t
JOIN ocr_region r ON r.region_id = t.region_id
JOIN ocr_run o ON o.ocr_run_id = r.ocr_run_id
JOIN page p ON p.page_id = o.page_id
LEFT JOIN token_correction c ON c.token_id = t.token_id AND c.accepted = TRUE;


-- Give the application role ownership of demo objects and access to the database.
-- Run this file as postgres (or another sufficiently privileged role).
DO $$
DECLARE
    object_name TEXT;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'digital_history') THEN
        RAISE EXCEPTION 'Role digital_history does not exist';
    END IF;

    FOREACH object_name IN ARRAY ARRAY[
        'corpus', 'source_file', 'page', 'ocr_run', 'ocr_region',
        'token', 'correction_candidate', 'token_correction'
    ] LOOP
        EXECUTE format('ALTER TABLE public.%I OWNER TO digital_history', object_name);
    END LOOP;
END $$;

ALTER VIEW public.analytical_token OWNER TO digital_history;
GRANT CONNECT, TEMPORARY ON DATABASE digital_history_demo TO digital_history;
GRANT USAGE, CREATE ON SCHEMA public TO digital_history;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO digital_history;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO digital_history;

COMMIT;
