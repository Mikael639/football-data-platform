ALTER TABLE dim_player
ADD COLUMN IF NOT EXISTS photo_url VARCHAR(500);

ALTER TABLE dim_team
ADD COLUMN IF NOT EXISTS crest_url TEXT;

ALTER TABLE dim_team
ADD COLUMN IF NOT EXISTS short_name TEXT;

ALTER TABLE fact_match
ADD COLUMN IF NOT EXISTS status TEXT;

ALTER TABLE fact_match
ADD COLUMN IF NOT EXISTS matchday INT;

ALTER TABLE fact_match
ADD COLUMN IF NOT EXISTS kickoff_utc TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS fact_standings_snapshot (
    competition_id INT NOT NULL REFERENCES dim_competition(competition_id),
    season INT NOT NULL,
    matchday INT,
    team_id INT NOT NULL REFERENCES dim_team(team_id),
    position INT,
    points INT,
    played_games INT,
    won INT,
    draw INT,
    lost INT,
    goals_for INT,
    goals_against INT,
    goal_difference INT,
    snapshot_ts TIMESTAMPTZ,
    PRIMARY KEY (competition_id, season, matchday, team_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_match_competition_kickoff_utc
    ON fact_match(competition_id, kickoff_utc);

CREATE INDEX IF NOT EXISTS idx_fact_match_home_team_kickoff_utc
    ON fact_match(home_team_id, kickoff_utc);

CREATE INDEX IF NOT EXISTS idx_fact_match_away_team_kickoff_utc
    ON fact_match(away_team_id, kickoff_utc);

CREATE INDEX IF NOT EXISTS idx_fact_standings_snapshot_competition_season_matchday
    ON fact_standings_snapshot(competition_id, season, matchday);
