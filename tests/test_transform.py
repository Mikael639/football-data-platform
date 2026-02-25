from src.extract import extract_from_mock
from src.transform import transform

def test_transform_shapes():
    payload = extract_from_mock()
    t = transform(payload)

    assert len(t["fact_match"]) == 2
    assert len(t["fact_player_match_stats"]) == 3

    # dim tables should not be empty
    assert len(t["dim_team"]) >= 2
    assert len(t["dim_player"]) >= 3
    assert len(t["dim_date"]) == 2

    # player enrichment
    assert "photo_url" in t["dim_player"][0]


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

    t = transform(payload)
    assert len(t["fact_match"]) == 1
    assert len(t["fact_player_match_stats"]) == 0
    assert any(p["player_id"] == 123 for p in t["dim_player"])
