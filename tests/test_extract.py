import requests
import sys
import types

import pandas as pd

from src.extract import (
    _fetch_standings_payload,
    extract_football_data_competition,
    extract_football_data_live_competitions,
)
from src.config import Settings
from src.player_stats_provider import fetch_player_match_candidates


def test_fetch_standings_payload_returns_empty_snapshot_for_missing_endpoint(monkeypatch):
    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError(response=response)

    def fake_fd_get(path, token, base_url, params=None):
        raise error

    monkeypatch.setattr("src.extract._fd_get", fake_fd_get)

    payload = _fetch_standings_payload("EL", 2025, "token", "https://api.football-data.org/v4")

    assert payload == {"season": {"currentMatchday": None}, "standings": []}


def test_extract_football_data_live_competitions_skips_forbidden_competitions(monkeypatch):
    def fake_extract(settings=None, competition_code=None, today=None):
        if competition_code == "EL":
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError(response=response)
        return {"competition_code": competition_code}

    monkeypatch.setattr("src.extract.extract_football_data_competition", fake_extract)

    settings = Settings.from_env(
        {
            "FOOTBALL_DATA_TOKEN": "token",
            "LIVE_COMPETITION_CODES": "PD,CL,EL",
            "DATA_MODE": "api",
        }
    )

    payloads = extract_football_data_live_competitions(settings=settings)

    assert [payload["competition_code"] for payload in payloads] == ["PD", "CL"]


def test_extract_football_data_competition_includes_enriched_player_stats(monkeypatch):
    def fake_fd_get(path, token, base_url, params=None):
        if path.endswith("/teams"):
            return {
                "competition": {"id": 2014, "name": "Primera Division"},
                "teams": [
                    {"id": 86, "name": "Real Madrid", "shortName": "Real Madrid"},
                    {"id": 81, "name": "Barcelona", "shortName": "Barcelona"},
                ],
            }
        if path.startswith("/teams/"):
            team_id = int(path.split("/")[-1])
            return {"team": {"id": team_id, "name": "Team"}, "squad": []}
        if path.endswith("/matches"):
            return {
                "matches": [
                    {
                        "id": 1001,
                        "utcDate": "2026-01-10T20:00:00Z",
                        "homeTeam": {"id": 86, "name": "Real Madrid"},
                        "awayTeam": {"id": 81, "name": "Barcelona"},
                    }
                ]
            }
        if path.endswith("/standings"):
            return {"season": {"currentMatchday": 20}, "standings": []}
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr("src.extract._fd_get", fake_fd_get)
    monkeypatch.setattr(
        "src.extract.fetch_player_match_candidates",
        lambda provider, competition_code, season_start, teams, matches, token=None, base_url=None, timeout_sec=30: [
            {"match_id": 1001, "player_id": 7, "minutes": 90}
        ],
    )

    settings = Settings.from_env(
        {
            "DATA_MODE": "api",
            "FOOTBALL_DATA_TOKEN": "token",
            "ENRICH_PLAYER_STATS": "true",
            "PLAYER_STATS_PROVIDER": "fbref",
            "COMPETITION_CODE": "PD",
        }
    )
    payload = extract_football_data_competition(settings=settings, competition_code="PD")

    assert payload["player_stats_provider"] == "fbref"
    assert payload["player_match_candidates"] == [{"match_id": 1001, "player_id": 7, "minutes": 90}]


def test_player_stats_provider_fbref_maps_rows_to_matches(monkeypatch):
    fake_soccerdata = types.SimpleNamespace()

    class FakeFBref:
        def __init__(self, leagues, seasons, proxy=None, no_cache=False):
            self.leagues = leagues
            self.seasons = seasons

        def read_player_match_stats(self, stat_type="summary"):
            return pd.DataFrame(
                [
                    {
                        "game": "2026-01-10 Real Madrid-Barcelona",
                        "team": "Real Madrid",
                        "player": "Kylian Mbappe",
                        "min": 90,
                        "gls": 1,
                        "ast": 0,
                        "sh": 4,
                        "cmp": 21,
                        "cmp_pct": "84",
                        "pos": "FW",
                        "starts": 1,
                    }
                ]
            )

    fake_soccerdata.FBref = FakeFBref
    monkeypatch.setitem(sys.modules, "soccerdata", fake_soccerdata)

    rows = fetch_player_match_candidates(
        provider="fbref",
        competition_code="PD",
        season_start=2025,
        teams=[
            {"id": 86, "name": "Real Madrid", "shortName": "Real Madrid"},
            {"id": 81, "name": "Barcelona", "shortName": "Barcelona"},
        ],
        matches=[
            {
                "id": 1001,
                "utcDate": "2026-01-10T20:00:00Z",
                "status": "FINISHED",
                "homeTeam": {"id": 86, "name": "Real Madrid"},
                "awayTeam": {"id": 81, "name": "Barcelona"},
            }
        ],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["match_id"] == 1001
    assert row["team_id"] == 86
    assert row["player_name"] == "Kylian Mbappe"
    assert row["minutes"] == 90
    assert row["goals"] == 1


def test_player_stats_provider_custom_http_with_token(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "date": "2026-01-10",
                        "team_name": "Real Madrid",
                        "player_name": "Kylian Mbappe",
                        "minutes": 90,
                        "goals": 1,
                        "assists": 0,
                    }
                ]
            }

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("src.player_stats_provider.requests.get", fake_get)

    rows = fetch_player_match_candidates(
        provider="custom_http",
        competition_code="PD",
        season_start=2025,
        teams=[{"id": 86, "name": "Real Madrid"}],
        matches=[
            {
                "id": 1001,
                "utcDate": "2026-01-10T20:00:00Z",
                "status": "FINISHED",
                "homeTeam": {"id": 86, "name": "Real Madrid"},
                "awayTeam": {"id": 81, "name": "Barcelona"},
            }
        ],
        token="abc123",
        base_url="https://player-stats.example.test",
        timeout_sec=15,
    )

    assert captured["url"] == "https://player-stats.example.test/player-match-stats"
    assert captured["params"] == {"competition_code": "PD", "season_start": 2025}
    assert captured["headers"]["Authorization"] == "Bearer abc123"
    assert len(rows) == 1
    assert rows[0]["match_id"] == 1001
