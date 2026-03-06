from src.extract import extract_from_mock
from src.transform import merge_transformed_data, transform, transform_csv_to_tables, transform_football_data


def test_transform_shapes():
    payload = extract_from_mock()
    transformed = transform(payload)

    assert len(transformed["fact_match"]) == 2
    assert len(transformed["fact_player_match_stats"]) == 3
    assert len(transformed["fact_standings_snapshot"]) == 2

    assert len(transformed["dim_team"]) >= 2
    assert len(transformed["dim_player"]) >= 3
    assert len(transformed["dim_date"]) == 2

    assert "photo_url" in transformed["dim_player"][0]


def test_transform_accepts_squad_players_without_player_match_stats():
    payload = {
        "fixtures": [
            {
                "match_id": 2001,
                "date": "2024-08-18",
                "competition_id": 1,
                "competition_name": "LaLiga",
                "home_team": {"team_id": 86, "team_name": "Real Madrid", "country": "Spain"},
                "away_team": {"team_id": 81, "team_name": "Barcelona", "country": "Spain"},
                "score": {"home": 1, "away": 0},
                "player_stats": [],
            }
        ],
        "squad_players": [
            {
                "player_id": 123,
                "full_name": "Test Player",
                "position": "MID",
                "nationality": "Spain",
                "birth_date": "2000-01-01",
                "photo_url": None,
                "team_id": 86,
            }
        ],
    }

    transformed = transform(payload)
    assert len(transformed["fact_match"]) == 1
    assert len(transformed["fact_player_match_stats"]) == 0
    assert any(player["player_id"] == 123 for player in transformed["dim_player"])


def test_fact_match_has_new_fields():
    payload = extract_from_mock()
    transformed = transform(payload)

    match_row = transformed["fact_match"][0]
    assert match_row["status"] == "FINISHED"
    assert match_row["matchday"] == 4
    assert "stage" in match_row
    assert "group_name" in match_row
    assert match_row["kickoff_utc"].isoformat() == "2024-09-01T19:00:00+00:00"


def test_dim_team_crest_url_nullable():
    payload = {
        "fixtures": [
            {
                "match_id": 3001,
                "date": "2024-08-18",
                "competition_id": 1,
                "competition_name": "LaLiga",
                "home_team": {"team_id": 86, "team_name": "Real Madrid", "country": "Spain", "crest_url": None},
                "away_team": {"team_id": 81, "team_name": "Barcelona", "country": "Spain"},
                "score": {"home": 1, "away": 0},
                "player_stats": [],
            }
        ]
    }

    transformed = transform(payload)
    team_row = next(team for team in transformed["dim_team"] if team["team_id"] == 86)
    assert "crest_url" in team_row
    assert team_row["crest_url"] is None


def test_standings_transform_shape():
    payload = {
        "season": 2025,
        "competition_code": "PD",
        "competition": {"id": 2014, "name": "Primera Division", "area": {"name": "Spain"}},
        "teams": [
            {"id": 86, "name": "Real Madrid", "shortName": "Real Madrid", "area": {"name": "Spain"}, "crest": None},
            {"id": 81, "name": "Barcelona", "shortName": "Barca", "area": {"name": "Spain"}, "crest": "crest.png"},
        ],
        "squads_by_team": [],
        "matches": [],
        "standings": {
            "season": {"currentMatchday": 26},
            "standings": [
                {
                    "type": "TOTAL",
                    "table": [
                        {
                            "position": 1,
                            "team": {"id": 86, "name": "Real Madrid"},
                            "points": 60,
                            "playedGames": 26,
                            "won": 19,
                            "draw": 3,
                            "lost": 4,
                            "goalsFor": 55,
                            "goalsAgainst": 20,
                            "goalDifference": 35,
                        },
                        {
                            "position": 2,
                            "team": {"id": 81, "name": "Barcelona"},
                            "points": 58,
                            "playedGames": 26,
                            "won": 18,
                            "draw": 4,
                            "lost": 4,
                            "goalsFor": 53,
                            "goalsAgainst": 24,
                            "goalDifference": 29,
                        },
                    ],
                }
            ],
        },
        "extracted_at_utc": "2025-03-02T09:00:00Z",
    }

    transformed = transform_football_data(payload)
    standings_rows = transformed["fact_standings_snapshot"]

    assert len(standings_rows) == 2
    assert {row["team_id"] for row in standings_rows} == {81, 86}
    assert all(row["competition_id"] == 2014 for row in standings_rows)
    assert all(row["season"] == 2025 for row in standings_rows)
    assert all(row["matchday"] == 26 for row in standings_rows)
    assert all(row["snapshot_ts"] is not None for row in standings_rows)


def test_transform_football_data_keeps_stage_and_group_name():
    payload = {
        "season": 2025,
        "competition_code": "CL",
        "competition": {"id": 2001, "name": "UEFA Champions League", "area": {"name": None}},
        "teams": [
            {"id": 86, "name": "Real Madrid CF", "shortName": "Real Madrid", "area": {"name": "Spain"}, "crest": None},
            {"id": 81, "name": "FC Barcelona", "shortName": "Barcelona", "area": {"name": "Spain"}, "crest": None},
        ],
        "squads_by_team": [],
        "matches": [
            {
                "id": 9001,
                "utcDate": "2025-10-01T19:00:00Z",
                "status": "SCHEDULED",
                "matchday": 2,
                "stage": "LEAGUE_STAGE",
                "group": "League phase",
                "homeTeam": {"id": 86, "name": "Real Madrid CF", "shortName": "Real Madrid"},
                "awayTeam": {"id": 81, "name": "FC Barcelona", "shortName": "Barcelona"},
                "score": {"fullTime": {"home": None, "away": None}},
            }
        ],
        "standings": {"season": {"currentMatchday": None}, "standings": []},
        "extracted_at_utc": "2025-10-01T09:00:00Z",
    }

    transformed = transform_football_data(payload)

    assert len(transformed["fact_match"]) == 1
    match_row = transformed["fact_match"][0]
    assert match_row["stage"] == "LEAGUE_STAGE"
    assert match_row["group_name"] == "League phase"


def test_transform_football_data_consumes_player_match_candidates():
    payload = {
        "season": 2025,
        "competition_code": "PD",
        "competition": {"id": 2014, "name": "Primera Division", "area": {"name": "Spain"}},
        "teams": [
            {"id": 86, "name": "Real Madrid", "shortName": "Real Madrid", "area": {"name": "Spain"}, "crest": None},
            {"id": 81, "name": "Barcelona", "shortName": "Barcelona", "area": {"name": "Spain"}, "crest": None},
        ],
        "squads_by_team": [],
        "matches": [
            {
                "id": 1001,
                "utcDate": "2026-01-10T20:00:00Z",
                "status": "FINISHED",
                "matchday": 20,
                "homeTeam": {"id": 86, "name": "Real Madrid", "shortName": "Real Madrid"},
                "awayTeam": {"id": 81, "name": "Barcelona", "shortName": "Barcelona"},
                "score": {"fullTime": {"home": 2, "away": 1}},
            }
        ],
        "player_match_candidates": [
            {
                "match_id": 1001,
                "player_id": 7000001,
                "player_name": "Kylian Mbappe",
                "position": "FW",
                "minutes": 90,
                "goals": 1,
                "assists": 0,
                "shots": 4,
                "passes": 21,
                "pass_accuracy": 0.84,
                "team_id": 86,
            }
        ],
        "standings": {"season": {"currentMatchday": 20}, "standings": []},
        "extracted_at_utc": "2026-01-10T21:00:00Z",
    }

    transformed = transform_football_data(payload)

    assert len(transformed["fact_player_match_stats"]) == 1
    assert transformed["fact_player_match_stats"][0]["match_id"] == 1001
    assert transformed["fact_player_match_stats"][0]["player_id"] == 7000001
    assert transformed["fact_player_match_stats"][0]["minutes"] == 90
    assert any(player["player_id"] == 7000001 for player in transformed["dim_player"])


def test_transform_football_data_skips_matches_with_missing_team_ids():
    payload = {
        "season": 2025,
        "competition_code": "CL",
        "competition": {"id": 2001, "name": "UEFA Champions League", "area": {"name": None}},
        "teams": [],
        "squads_by_team": [],
        "matches": [
            {
                "id": 9002,
                "utcDate": "2025-10-02T19:00:00Z",
                "status": "TIMED",
                "matchday": 2,
                "stage": "LEAGUE_STAGE",
                "group": "League phase",
                "homeTeam": {"id": None, "name": "TBD"},
                "awayTeam": {"id": 81, "name": "FC Barcelona", "shortName": "Barcelona"},
                "score": {"fullTime": {"home": None, "away": None}},
            }
        ],
        "standings": {"season": {"currentMatchday": None}, "standings": []},
        "extracted_at_utc": "2025-10-01T09:00:00Z",
    }

    transformed = transform_football_data(payload)

    assert transformed["fact_match"] == []


def test_merge_transformed_data_prefers_richer_team_rows_and_combines_matches():
    csv_transformed = {
        "dim_date": [{"date_id": "2024-08-18", "year": 2024, "month": 8, "day": 18}],
        "dim_team": [{"team_id": 101, "team_name": "Real Madrid", "country": "Spain", "crest_url": None, "short_name": None}],
        "dim_competition": [{"competition_id": 2014, "competition_name": "La Liga", "country": "Spain"}],
        "dim_player": [],
        "fact_match": [
            {
                "match_id": 1,
                "date_id": "2024-08-18",
                "competition_id": 2014,
                "home_team_id": 101,
                "away_team_id": 202,
                "status": "FINISHED",
                "matchday": 1,
                "kickoff_utc": None,
                "season": "2024-2025",
                "home_score": 2,
                "away_score": 0,
            }
        ],
        "fact_player_match_stats": [],
        "fact_standings_snapshot": [],
    }
    api_transformed = {
        "dim_date": [{"date_id": "2025-08-17", "year": 2025, "month": 8, "day": 17}],
        "dim_team": [
            {"team_id": 86, "team_name": "Real Madrid", "country": "Spain", "crest_url": "crest.png", "short_name": "Real Madrid"},
            {"team_id": 81, "team_name": "Barcelona", "country": "Spain", "crest_url": "barca.png", "short_name": "Barca"},
        ],
        "dim_competition": [{"competition_id": 2014, "competition_name": "Primera Division", "country": "Spain"}],
        "dim_player": [],
        "fact_match": [
            {
                "match_id": 2,
                "date_id": "2025-08-17",
                "competition_id": 2014,
                "home_team_id": 86,
                "away_team_id": 81,
                "status": "SCHEDULED",
                "matchday": 1,
                "kickoff_utc": None,
                "season": "2025-2026",
                "home_score": None,
                "away_score": None,
            }
        ],
        "fact_player_match_stats": [],
        "fact_standings_snapshot": [],
    }

    merged = merge_transformed_data(csv_transformed, api_transformed)

    real_madrid = next(team for team in merged["dim_team"] if team["team_name"] == "Real Madrid")
    assert real_madrid["team_id"] == 86
    assert real_madrid["crest_url"] == "crest.png"
    assert len(merged["fact_match"]) == 2
    csv_match = next(match for match in merged["fact_match"] if match["match_id"] == 1)
    assert csv_match["home_team_id"] == 86


def test_transform_csv_to_tables_builds_players_and_player_match_stats(tmp_path):
    payload = {
        "competition_code": "PD",
        "competition": {"id": 2014, "name": "La Liga", "area": {"name": "Spain"}},
        "teams": [
            {"id": 86, "name": "Real Madrid", "shortName": "Real Madrid", "area": {"name": "Spain"}, "crest": None},
            {"id": 92, "name": "Real Sociedad", "shortName": "Real Sociedad", "area": {"name": "Spain"}, "crest": None},
        ],
        "match_candidates": [
            {
                "match_id": 1001,
                "date_id": "2020-09-20",
                "home_team_id": 92,
                "away_team_id": 86,
                "status": "FINISHED",
                "matchday": 2,
                "kickoff_utc": "2020-09-20T12:00:00Z",
                "season": "2020-2021",
                "home_score": 0,
                "away_score": 0,
            },
            {
                "match_id": 1002,
                "date_id": "2020-09-30",
                "home_team_id": 86,
                "away_team_id": 99,
                "status": "FINISHED",
                "matchday": 4,
                "kickoff_utc": "2020-09-30T12:00:00Z",
                "season": "2020-2021",
                "home_score": 1,
                "away_score": 0,
            },
        ],
        "player_match_candidates": [
            {
                "match_id": 1001,
                "player_id": 9,
                "player_name": "Karim Benzema",
                "position": "FW",
                "minutes": 90,
                "goals": 0,
                "assists": 0,
                "shots": 2,
                "passes": 20,
                "pass_accuracy": 0.84,
                "team_id": 86,
            },
            {
                "match_id": 1002,
                "player_id": 9,
                "player_name": "Karim Benzema",
                "position": "FW",
                "minutes": 90,
                "goals": 1,
                "assists": 0,
                "shots": 4,
                "passes": 23,
                "pass_accuracy": 0.88,
                "team_id": 86,
            },
        ],
    }
    transformed = transform_csv_to_tables(payload)

    assert len(transformed["dim_player"]) == 1
    assert len(transformed["fact_player_match_stats"]) == 2
    player_row = transformed["dim_player"][0]
    assert player_row["full_name"] == "Karim Benzema"
    assert player_row["team_id"] != 0
    assert all(row["minutes"] > 0 for row in transformed["fact_player_match_stats"])
