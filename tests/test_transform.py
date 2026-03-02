from src.extract import extract_from_mock
from src.transform import transform, transform_football_data


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
