-- Journal des exécutions du pipeline ETL (suivi technique)
CREATE TABLE pipeline_run_log (
    run_id UUID PRIMARY KEY,          -- Identifiant unique de l'exécution
    started_at TIMESTAMP NOT NULL,    -- Date/heure de début
    ended_at TIMESTAMP,               -- Date/heure de fin (NULL si en cours / interrompu)
    status VARCHAR(20),               -- Statut (ex: SUCCESS, FAILED, RUNNING)
    extracted_count INT,              -- Nombre de lignes extraites
    loaded_count INT,                 -- Nombre de lignes chargées
    error_message TEXT                -- Message d'erreur si échec
);

-- Dimension des équipes
CREATE TABLE dim_team (
    team_id INT PRIMARY KEY,          -- Identifiant unique de l'équipe
    team_name VARCHAR(100),           -- Nom de l'équipe
    country VARCHAR(100)              -- Pays de l'équipe
);

-- Dimension des joueurs
CREATE TABLE dim_player (
    player_id INT PRIMARY KEY,        -- Identifiant unique du joueur
    full_name VARCHAR(150),           -- Nom complet du joueur
    position VARCHAR(50),             -- Poste (attaquant, milieu, etc.)
    nationality VARCHAR(100),         -- Nationalité
    birth_date DATE,                  -- Date de naissance
    team_id INT REFERENCES dim_team(team_id) -- Équipe actuelle du joueur (clé étrangère)
);

-- Dimension des compétitions
CREATE TABLE dim_competition (
    competition_id INT PRIMARY KEY,   -- Identifiant unique de la compétition
    competition_name VARCHAR(150),    -- Nom de la compétition
    country VARCHAR(100)              -- Pays/zone de la compétition
);

-- Dimension temps (date)
CREATE TABLE dim_date (
    date_id DATE PRIMARY KEY,         -- Date (utilisée comme clé primaire)
    year INT,                         -- Année
    month INT,                        -- Mois
    day INT                           -- Jour
);

-- Table de faits des matchs (niveau match)
CREATE TABLE fact_match (
    match_id INT PRIMARY KEY,         -- Identifiant unique du match
    date_id DATE REFERENCES dim_date(date_id), -- Date du match
    competition_id INT REFERENCES dim_competition(competition_id), -- Compétition
    home_team_id INT REFERENCES dim_team(team_id), -- Équipe à domicile
    away_team_id INT REFERENCES dim_team(team_id), -- Équipe à l'extérieur
    home_score INT,                   -- Score de l'équipe à domicile
    away_score INT                    -- Score de l'équipe à l'extérieur
);

-- Table de faits des statistiques joueurs par match (niveau joueur-match)
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
