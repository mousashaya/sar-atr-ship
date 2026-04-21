-- setup_db.sql
-- =============
-- Creates the defense schema and all tables.
-- Run once: psql -d fungidb -f scripts/setup_db.sql
--
-- Reuses the same Postgres server as lichen, in a separate schema.

CREATE SCHEMA IF NOT EXISTS defense;

-- ── ATRNet image metadata ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS defense.atrnet_images (
    id               SERIAL PRIMARY KEY,
    local_path       TEXT NOT NULL,
    target_type      TEXT,          -- 40-class leaf: e.g. 'Toyota_Camry_2012'
    subclass         TEXT,          -- 21-class: e.g. 'Medium_SUV'
    main_class       TEXT,          -- 4-class: e.g. 'Car'
    scene            TEXT,          -- city / factory / sandstone / woodland / bare_soil
    split            TEXT,          -- train / test (ATRNet pre-defined splits)
    experiment_setting TEXT,        -- ATRNet experimental setting name (e.g. 'Setting1')
    depression_angle FLOAT,
    azimuth_angle    FLOAT,
    polarization     TEXT,
    band             TEXT
);

CREATE INDEX IF NOT EXISTS idx_atrnet_target_type ON defense.atrnet_images(target_type);
CREATE INDEX IF NOT EXISTS idx_atrnet_main_class  ON defense.atrnet_images(main_class);
CREATE INDEX IF NOT EXISTS idx_atrnet_scene        ON defense.atrnet_images(scene);

-- ── Experiments ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS defense.experiments (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    model_type  TEXT,
    config_name TEXT,
    quality_min INT DEFAULT 0,
    epochs      INT,
    notes       TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- ── Per-class metrics (precision / recall / F1) ────────────────────────────────
CREATE TABLE IF NOT EXISTS defense.experiment_metrics (
    id            SERIAL PRIMARY KEY,
    experiment_id INT REFERENCES defense.experiments(id) ON DELETE CASCADE,
    split         TEXT,          -- 'val' or 'test'
    class_name    TEXT,          -- NULL = overall weighted average
    accuracy      FLOAT,
    precision     FLOAT,
    recall        FLOAT,
    f1            FLOAT,
    support       INT
);

-- ── Per-image predictions (for confusion matrix drill-down) ────────────────────
CREATE TABLE IF NOT EXISTS defense.experiment_predictions (
    id              SERIAL PRIMARY KEY,
    experiment_id   INT REFERENCES defense.experiments(id) ON DELETE CASCADE,
    image_id        INT REFERENCES defense.atrnet_images(id),
    split           TEXT,
    true_class      TEXT,
    predicted_class TEXT,
    confidence      FLOAT
);

CREATE INDEX IF NOT EXISTS idx_def_preds_exp_split
    ON defense.experiment_predictions(experiment_id, split);
