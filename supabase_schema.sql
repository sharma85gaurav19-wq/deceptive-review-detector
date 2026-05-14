-- ──────────────────────────────────────────────────────────────────────────────
-- Deceptive Review Detector — Supabase database schema
-- Run once in the Supabase SQL editor: Database → SQL Editor → New query
-- ──────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predictions (
    id                    UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    review_text           TEXT          NOT NULL,
    predicted_label       TEXT          NOT NULL CHECK (predicted_label IN ('Genuine', 'Deceptive')),
    confidence_score      NUMERIC(6, 4) NOT NULL,
    trust_score           NUMERIC(5, 1) NOT NULL,
    model_used            TEXT          NOT NULL,
    dataset_used          TEXT          NOT NULL,
    source                TEXT          NOT NULL DEFAULT 'manual',
    product_asin          TEXT,
    rating                INTEGER,
    verified_purchase     BOOLEAN,
    reviewer_review_count INTEGER,
    user_feedback         TEXT          CHECK (user_feedback IN ('correct', 'incorrect')),
    created_at            TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_label
    ON predictions (predicted_label);
CREATE INDEX IF NOT EXISTS idx_predictions_feedback
    ON predictions (user_feedback)
    WHERE user_feedback IS NOT NULL;

-- Row-Level Security — allow public read/write via the anon key
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon_insert"          ON predictions;
DROP POLICY IF EXISTS "anon_select"          ON predictions;
DROP POLICY IF EXISTS "anon_update_feedback" ON predictions;

CREATE POLICY "anon_insert" ON predictions
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "anon_select" ON predictions
    FOR SELECT TO anon USING (true);

CREATE POLICY "anon_update_feedback" ON predictions
    FOR UPDATE TO anon USING (true) WITH CHECK (true);
