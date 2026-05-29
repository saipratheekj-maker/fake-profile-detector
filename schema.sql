-- ============================================================
--  ProfileSentinel — MySQL Schema
--  Run once:  mysql -u root -p < schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS profile_sentinel
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE profile_sentinel;

-- ── Main profiles table ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS profiles (
  id                    INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

  -- Identity
  username              VARCHAR(100)  NOT NULL,
  display_name          VARCHAR(150)  DEFAULT '',
  bio                   TEXT,

  -- Raw stats
  followers_count       INT UNSIGNED  DEFAULT 0,
  following_count       INT UNSIGNED  DEFAULT 0,
  posts_count           INT UNSIGNED  DEFAULT 0,
  account_age_days      INT UNSIGNED  DEFAULT 0,
  avg_likes_per_post    FLOAT         DEFAULT 0,
  avg_comments_per_post FLOAT         DEFAULT 0,
  has_profile_pic       TINYINT(1)    DEFAULT 0,
  has_external_url      TINYINT(1)    DEFAULT 0,
  is_private            TINYINT(1)    DEFAULT 0,

  -- ML result
  prediction            ENUM('fake','real') NOT NULL,
  fake_probability      FLOAT         NOT NULL,
  real_probability      FLOAT         NOT NULL,
  trust_score           INT           NOT NULL,
  confidence            ENUM('high','medium','low') NOT NULL,
  risk_factors          JSON,
  top_features          JSON,

  -- Derived features stored for auditing
  followers_following_ratio FLOAT     DEFAULT 0,
  post_frequency        FLOAT         DEFAULT 0,
  username_digit_ratio  FLOAT         DEFAULT 0,
  bio_length            INT           DEFAULT 0,

  -- Meta
  analyzed_at           DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  notes                 TEXT,

  INDEX idx_username    (username),
  INDEX idx_prediction  (prediction),
  INDEX idx_analyzed_at (analyzed_at),
  INDEX idx_trust_score (trust_score)
) ENGINE=InnoDB;


-- ── Summary stats view ────────────────────────────────────────
CREATE OR REPLACE VIEW vw_stats AS
SELECT
  COUNT(*)                                        AS total_analyzed,
  SUM(prediction = 'fake')                        AS total_fake,
  SUM(prediction = 'real')                        AS total_real,
  ROUND(AVG(fake_probability), 2)                 AS avg_fake_probability,
  ROUND(AVG(trust_score), 1)                      AS avg_trust_score,
  SUM(confidence = 'high')                        AS high_confidence_count,
  MAX(analyzed_at)                                AS last_analyzed_at
FROM profiles;
