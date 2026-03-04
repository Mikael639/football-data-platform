import requests

from src.extract import _fetch_standings_payload, extract_football_data_live_competitions
from src.config import Settings


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
