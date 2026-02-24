CREATE INDEX idx_fact_match_date ON fact_match(date_id);
CREATE INDEX idx_fact_match_competition ON fact_match(competition_id);
CREATE INDEX idx_fact_player_match_player ON fact_player_match_stats(player_id);
CREATE INDEX idx_dim_player_team ON dim_player(team_id);