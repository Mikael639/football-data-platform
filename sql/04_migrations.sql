ALTER TABLE fact_match
ADD COLUMN IF NOT EXISTS season TEXT;

CREATE INDEX IF NOT EXISTS idx_fact_match_competition_season
    ON fact_match (competition_id, season);
