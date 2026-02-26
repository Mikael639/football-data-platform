CREATE TABLE IF NOT EXISTS public.study_fbref_standard_season_raw (
    raw_id BIGSERIAL PRIMARY KEY,
    season_start INTEGER NOT NULL,
    source_file TEXT NOT NULL,
    rk INTEGER NULL,
    player_name TEXT NOT NULL,
    nation_raw TEXT NULL,
    nation_code TEXT NULL,
    position_raw TEXT NULL,
    position_group TEXT NULL,
    team_name TEXT NULL,
    age INTEGER NULL,
    birth_year INTEGER NULL,
    matches_played INTEGER NULL,
    starts INTEGER NULL,
    minutes_total INTEGER NULL,
    nineties NUMERIC NULL,
    goals_total INTEGER NULL,
    assists_total INTEGER NULL,
    ga_total INTEGER NULL,
    goals_non_pk_total INTEGER NULL,
    pk_goals INTEGER NULL,
    pk_attempts INTEGER NULL,
    yellow_cards INTEGER NULL,
    red_cards INTEGER NULL,
    goals_p90 NUMERIC NULL,
    assists_p90 NUMERIC NULL,
    ga_p90 NUMERIC NULL,
    goals_non_pk_p90 NUMERIC NULL,
    ga_non_pk_p90 NUMERIC NULL,
    ingested_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_study_fbref_raw_season ON public.study_fbref_standard_season_raw (season_start);
CREATE INDEX IF NOT EXISTS idx_study_fbref_raw_player ON public.study_fbref_standard_season_raw (player_name);
ALTER TABLE public.study_fbref_standard_season_raw
    ADD COLUMN IF NOT EXISTS player_key TEXT NULL;

CREATE TABLE IF NOT EXISTS public.study_fbref_player_season (
    season_start INTEGER NOT NULL,
    player_id BIGINT NOT NULL,
    player_key TEXT NOT NULL,
    player_name TEXT NOT NULL,
    team_id BIGINT NOT NULL,
    team_name TEXT NOT NULL,
    position_group TEXT NOT NULL,
    position_raw TEXT NULL,
    clubs_count INTEGER NOT NULL DEFAULT 1,
    clubs_list TEXT NULL,
    is_multi_club_season BOOLEAN NOT NULL DEFAULT FALSE,
    nation_code TEXT NULL,
    age INTEGER NULL,
    birth_year INTEGER NULL,
    matches_played INTEGER NOT NULL DEFAULT 0,
    starts INTEGER NOT NULL DEFAULT 0,
    sub_apps INTEGER NOT NULL DEFAULT 0,
    minutes_total INTEGER NOT NULL DEFAULT 0,
    goals_total INTEGER NOT NULL DEFAULT 0,
    assists_total INTEGER NOT NULL DEFAULT 0,
    ga_total INTEGER NOT NULL DEFAULT 0,
    goals_non_pk_total INTEGER NOT NULL DEFAULT 0,
    pk_goals_total INTEGER NOT NULL DEFAULT 0,
    pk_attempts_total INTEGER NOT NULL DEFAULT 0,
    yellow_cards_total INTEGER NOT NULL DEFAULT 0,
    red_cards_total INTEGER NOT NULL DEFAULT 0,
    shots_total INTEGER NOT NULL DEFAULT 0,
    passes_total INTEGER NOT NULL DEFAULT 0,
    pass_acc_mean NUMERIC NOT NULL DEFAULT 0,
    goals_p90 NUMERIC NOT NULL DEFAULT 0,
    assists_p90 NUMERIC NOT NULL DEFAULT 0,
    ga_p90 NUMERIC NOT NULL DEFAULT 0,
    shots_p90 NUMERIC NOT NULL DEFAULT 0,
    passes_p90 NUMERIC NOT NULL DEFAULT 0,
    eligible_600 BOOLEAN NOT NULL DEFAULT FALSE,
    eligible_900 BOOLEAN NOT NULL DEFAULT FALSE,
    source_mode TEXT NOT NULL DEFAULT 'manual_standard_csv',
    source_file TEXT NULL,
    ingested_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season_start, player_id)
);
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS clubs_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS clubs_list TEXT NULL;
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS is_multi_club_season BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS goals_non_pk_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS pk_goals_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS pk_attempts_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS yellow_cards_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE public.study_fbref_player_season ADD COLUMN IF NOT EXISTS red_cards_total INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_study_fbref_player_season_player ON public.study_fbref_player_season (player_id);
CREATE INDEX IF NOT EXISTS idx_study_fbref_player_season_team ON public.study_fbref_player_season (team_name);
CREATE INDEX IF NOT EXISTS idx_study_fbref_player_season_season_pos ON public.study_fbref_player_season (season_start, position_group);

CREATE TABLE IF NOT EXISTS public.study_fbref_player_match (
    season_start INTEGER NOT NULL,
    match_id TEXT NOT NULL,
    date_id TEXT NOT NULL,
    competition TEXT NULL,
    team_id BIGINT NOT NULL,
    team_name TEXT NOT NULL,
    player_id BIGINT NOT NULL,
    player_name TEXT NOT NULL,
    player_key TEXT NULL,
    position TEXT NULL,
    position_group TEXT NULL,
    is_starting BOOLEAN NOT NULL DEFAULT FALSE,
    minutes INTEGER NOT NULL DEFAULT 0,
    goals INTEGER NOT NULL DEFAULT 0,
    assists INTEGER NOT NULL DEFAULT 0,
    shots INTEGER NOT NULL DEFAULT 0,
    passes INTEGER NOT NULL DEFAULT 0,
    pass_accuracy NUMERIC NOT NULL DEFAULT 0,
    played_flag INTEGER NOT NULL DEFAULT 0,
    start_flag INTEGER NOT NULL DEFAULT 0,
    sub_in_flag INTEGER NOT NULL DEFAULT 0,
    ga INTEGER NOT NULL DEFAULT 0,
    goals_p90_match NUMERIC NOT NULL DEFAULT 0,
    assists_p90_match NUMERIC NOT NULL DEFAULT 0,
    ga_p90_match NUMERIC NOT NULL DEFAULT 0,
    shots_p90_match NUMERIC NOT NULL DEFAULT 0,
    passes_p90_match NUMERIC NOT NULL DEFAULT 0,
    source_mode TEXT NOT NULL DEFAULT 'manual_matchlog_csv',
    source_file TEXT NULL,
    ingested_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season_start, match_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_study_fbref_player_match_player ON public.study_fbref_player_match (player_id);
CREATE INDEX IF NOT EXISTS idx_study_fbref_player_match_season ON public.study_fbref_player_match (season_start);
CREATE INDEX IF NOT EXISTS idx_study_fbref_player_match_season_player ON public.study_fbref_player_match (season_start, player_id);

CREATE TABLE IF NOT EXISTS public.study_fbref_regularity (
    season_start INTEGER NOT NULL,
    player_id BIGINT NOT NULL,
    player_name TEXT NOT NULL,
    team_id BIGINT NOT NULL,
    team_name TEXT NOT NULL,
    position_group TEXT NOT NULL,
    minutes_total INTEGER NOT NULL DEFAULT 0,
    matches_played INTEGER NOT NULL DEFAULT 0,
    ga_p90_mean NUMERIC NOT NULL DEFAULT 0,
    ga_p90_std NUMERIC NOT NULL DEFAULT 0,
    shots_p90_mean NUMERIC NOT NULL DEFAULT 0,
    shots_p90_std NUMERIC NOT NULL DEFAULT 0,
    passes_p90_mean NUMERIC NOT NULL DEFAULT 0,
    passes_p90_std NUMERIC NOT NULL DEFAULT 0,
    ga_p90_cv NUMERIC NULL,
    shots_p90_cv NUMERIC NULL,
    passes_p90_cv NUMERIC NULL,
    stability_proxy NUMERIC NOT NULL DEFAULT 0,
    perf_z NUMERIC NOT NULL DEFAULT 0,
    stab_z NUMERIC NOT NULL DEFAULT 0,
    regularity_score NUMERIC NOT NULL DEFAULT 0,
    regularity_rank_pos INTEGER NULL,
    podium TEXT NULL,
    source_mode TEXT NOT NULL DEFAULT 'manual_standard_csv',
    note TEXT NULL,
    ingested_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season_start, player_id)
);

CREATE TABLE IF NOT EXISTS public.study_fbref_progression (
    season_start INTEGER NOT NULL,
    player_id BIGINT NOT NULL,
    player_name TEXT NOT NULL,
    team_name TEXT NOT NULL,
    position_group TEXT NOT NULL,
    minutes_total INTEGER NOT NULL DEFAULT 0,
    goals_p90 NUMERIC NOT NULL DEFAULT 0,
    assists_p90 NUMERIC NOT NULL DEFAULT 0,
    ga_p90 NUMERIC NOT NULL DEFAULT 0,
    shots_p90 NUMERIC NOT NULL DEFAULT 0,
    passes_p90 NUMERIC NOT NULL DEFAULT 0,
    pass_acc_mean NUMERIC NOT NULL DEFAULT 0,
    player_name_prev TEXT NULL,
    team_name_prev TEXT NULL,
    position_group_prev TEXT NULL,
    minutes_total_prev INTEGER NULL,
    goals_p90_prev NUMERIC NULL,
    assists_p90_prev NUMERIC NULL,
    ga_p90_prev NUMERIC NULL,
    shots_p90_prev NUMERIC NULL,
    passes_p90_prev NUMERIC NULL,
    pass_acc_mean_prev NUMERIC NULL,
    delta_goals_p90 NUMERIC NULL,
    delta_assists_p90 NUMERIC NULL,
    delta_ga_p90 NUMERIC NULL,
    delta_shots_p90 NUMERIC NULL,
    delta_passes_p90 NUMERIC NULL,
    delta_pass_acc_mean NUMERIC NULL,
    delta_minutes_total INTEGER NULL,
    progress_score NUMERIC NULL,
    progress_rank_pos INTEGER NULL,
    podium TEXT NULL,
    source_mode TEXT NOT NULL DEFAULT 'manual_standard_csv',
    ingested_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season_start, player_id)
);

CREATE INDEX IF NOT EXISTS idx_study_fbref_progression_season_pos ON public.study_fbref_progression (season_start, position_group);

CREATE TABLE IF NOT EXISTS public.study_fbref_meta (
    dataset_name TEXT PRIMARY KEY,
    league TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    seasons_start_years JSONB NOT NULL,
    season_labels JSONB NOT NULL,
    files JSONB NULL,
    notes TEXT NULL,
    generated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
