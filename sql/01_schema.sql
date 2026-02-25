-- ETL pipeline execution log (technical monitoring)
CREATE TABLE pipeline_run_log (
    run_id UUID PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    status VARCHAR(20),
    extracted_count INT,
    loaded_count INT,
    error_message TEXT
);

-- Team dimension
CREATE TABLE dim_team (
    team_id INT PRIMARY KEY,
    team_name VARCHAR(100),
    country VARCHAR(100)
);

-- Player dimension
CREATE TABLE dim_player (
    player_id INT PRIMARY KEY,
    full_name VARCHAR(150),
    position VARCHAR(50),
    nationality VARCHAR(100),
    birth_date DATE,
    photo_url VARCHAR(500),
    team_id INT REFERENCES dim_team(team_id)
);

-- Competition dimension
CREATE TABLE dim_competition (
    competition_id INT PRIMARY KEY,
    competition_name VARCHAR(150),
    country VARCHAR(100)
);

-- Date dimension
CREATE TABLE dim_date (
    date_id DATE PRIMARY KEY,
    year INT,
    month INT,
    day INT
);

-- Match fact table
CREATE TABLE fact_match (
    match_id INT PRIMARY KEY,
    date_id DATE REFERENCES dim_date(date_id),
    competition_id INT REFERENCES dim_competition(competition_id),
    home_team_id INT REFERENCES dim_team(team_id),
    away_team_id INT REFERENCES dim_team(team_id),
    home_score INT,
    away_score INT
);

-- Player stats fact table (player-match grain)
CREATE TABLE fact_player_match_stats (
    match_id INT NOT NULL REFERENCES fact_match(match_id),
    player_id INT NOT NULL REFERENCES dim_player(player_id),
    minutes INT NOT NULL CHECK (minutes >= 0 AND minutes <= 130),
    goals INT NOT NULL DEFAULT 0 CHECK (goals >= 0),
    assists INT NOT NULL DEFAULT 0 CHECK (assists >= 0),
    shots INT NOT NULL DEFAULT 0 CHECK (shots >= 0),
    passes INT NOT NULL DEFAULT 0 CHECK (passes >= 0),
    pass_accuracy FLOAT CHECK (pass_accuracy >= 0 AND pass_accuracy <= 1),
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS data_quality_check (
    check_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pipeline_run_log(run_id),
    check_name VARCHAR(200) NOT NULL,
    status VARCHAR(20) NOT NULL, -- PASS/FAIL
    metric_value FLOAT,
    threshold FLOAT,
    details TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
