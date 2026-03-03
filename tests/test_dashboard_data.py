import sys
import types

import pandas as pd

if "streamlit" not in sys.modules:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_data = lambda **kwargs: (lambda fn: fn)
    sys.modules["streamlit"] = streamlit_stub

from dashboard.data.dashboard_data import (
    DashboardFilters,
    build_local_league_table,
    build_match_where_clause,
    build_perspective_table,
    build_team_match_view,
    compute_team_kpis,
    describe_season_source,
    get_current_standings,
    get_live_league_tables,
    get_teams,
)


def _sample_matches() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "match_id": 1,
                "date_id": "2025-08-10",
                "date_dt": pd.Timestamp("2025-08-10"),
                "kickoff_utc": pd.Timestamp("2025-08-10T19:00:00Z"),
                "status": "FINISHED",
                "matchday": 1,
                "home_team_id": 10,
                "home_team": "FC Alpha",
                "away_team_id": 20,
                "away_team": "FC Beta",
                "home_score": 2,
                "away_score": 1,
            },
            {
                "match_id": 2,
                "date_id": "2025-08-17",
                "date_dt": pd.Timestamp("2025-08-17"),
                "kickoff_utc": pd.Timestamp("2025-08-17T19:00:00Z"),
                "status": "FINISHED",
                "matchday": 2,
                "home_team_id": 20,
                "home_team": "FC Beta",
                "away_team_id": 10,
                "away_team": "FC Alpha",
                "home_score": 0,
                "away_score": 0,
            },
            {
                "match_id": 3,
                "date_id": "2025-08-24",
                "date_dt": pd.Timestamp("2025-08-24"),
                "kickoff_utc": pd.Timestamp("2025-08-24T19:00:00Z"),
                "status": "SCHEDULED",
                "matchday": 3,
                "home_team_id": 10,
                "home_team": "FC Alpha",
                "away_team_id": 30,
                "away_team": "FC Gamma",
                "home_score": None,
                "away_score": None,
            },
        ]
    )
    return df


def test_compute_team_kpis_ignores_unplayed_and_counts_points():
    df = _sample_matches()

    played, wins, draws, losses, gf, ga, points = compute_team_kpis(df, team_id=10)

    assert played == 2
    assert (wins, draws, losses) == (1, 1, 0)
    assert (gf, ga) == (2, 1)
    assert points == 4


def test_build_team_match_view_builds_cumulative_points():
    df = _sample_matches()

    out = build_team_match_view(df, team_id=10)

    assert list(out["Result"]) == ["W", "D"]
    assert list(out["Points"]) == [3, 1]
    assert list(out["CumulativePoints"]) == [3, 4]
    assert list(out["venue"]) == ["Domicile", "Exterieur"]


def test_build_local_league_table_ranks_by_points_then_gd():
    df = _sample_matches()

    table = build_local_league_table(df)

    assert set(table["Team"]) == {"FC Alpha", "FC Beta"}
    leader = table.iloc[0]
    assert leader["Team"] == "FC Alpha"
    assert int(leader["Pts"]) == 4
    assert int(leader["P"]) == 2
    assert int(leader["GD"]) == 1


def test_build_match_where_clause_includes_all_filters(monkeypatch):
    monkeypatch.setattr("dashboard.data.dashboard_data.fact_match_has_season_column", lambda: True)
    monkeypatch.setattr("dashboard.data.dashboard_data.fact_match_has_non_null_seasons", lambda competition_id=None: True)
    monkeypatch.setattr(
        "dashboard.data.dashboard_data.get_team_alias_ids",
        lambda team_id, competition_id=None, season=None: [int(team_id)],
    )

    clause, params = build_match_where_clause(
        DashboardFilters(
            competition_id=2014,
            season="2025-2026",
            team_id=10,
            date_start="2025-08-01",
            date_end="2025-08-31",
        )
    )

    assert "m.competition_id = :competition_id" in clause
    assert "(m.home_team_id IN (" in clause
    assert any(key.startswith("team_id_") for key in params)
    assert params["season"] == "2025-2026"
    assert params["date_start"] == "2025-08-01"


def test_build_match_where_clause_excludes_legacy_null_season_rows(monkeypatch):
    monkeypatch.setattr("dashboard.data.dashboard_data.fact_match_has_season_column", lambda: True)
    monkeypatch.setattr("dashboard.data.dashboard_data.fact_match_has_non_null_seasons", lambda competition_id=None: True)

    clause, params = build_match_where_clause(
        DashboardFilters(
            competition_id=2014,
            season=None,
            team_id=None,
            date_start=None,
            date_end=None,
        )
    )

    assert "NULLIF(TRIM(m.season), '') IS NOT NULL" in clause
    assert "season" not in params


def test_build_perspective_table_creates_home_and_away_rows():
    df = _sample_matches().head(2)

    perspective = build_perspective_table(df, team_id=None)

    assert len(perspective) == 4
    assert set(perspective["venue"]) == {"Home", "Away"}
    assert set(perspective["result"].dropna()) == {"W", "L", "D"}


def test_dashboard_overview_uses_snapshot(monkeypatch):
    captured: dict[str, object] = {}

    def fake_read_sql(query, params=None):
        captured["query"] = query
        captured["params"] = params or {}
        return pd.DataFrame(
            [
                {
                    "competition_id": 2014,
                    "season": 2020,
                    "matchday": 5,
                    "team_id": 10,
                    "team_name": "FC Alpha",
                    "short_name": "Alpha",
                    "crest_url": None,
                    "position": 1,
                    "points": 13,
                    "played_games": 5,
                    "won": 4,
                    "draw": 1,
                    "lost": 0,
                    "goals_for": 10,
                    "goals_against": 3,
                    "goal_difference": 7,
                }
            ]
        )

    monkeypatch.setattr("dashboard.data.dashboard_data._read_sql", fake_read_sql)

    standings = get_current_standings(competition_id=2014, season="2020-2021")

    assert "fact_standings_snapshot" in str(captured["query"])
    assert captured["params"]["season_start"] == 2020
    assert len(standings) == 1


def test_describe_season_source_distinguishes_current_and_historical_seasons():
    assert "football-data.org" in describe_season_source("2025-2026")
    assert "historique consolide" in describe_season_source("2024-2025")


def test_get_teams_groups_alias_variants(monkeypatch):
    monkeypatch.setattr(
        "dashboard.data.dashboard_data._read_sql",
        lambda query, params=None: pd.DataFrame(
            [
                {"team_id": 81, "team_name": "FC Barcelona", "short_name": "Barca", "crest_url": "crest"},
                {"team_id": 529, "team_name": "Barcelona", "short_name": None, "crest_url": None},
                {"team_id": 86, "team_name": "Real Madrid CF", "short_name": "Real Madrid", "crest_url": "crest"},
                {"team_id": 541, "team_name": "Real Madrid", "short_name": None, "crest_url": None},
            ]
        ),
    )

    teams = get_teams(2014, None)

    assert len(teams) == 2
    assert teams["team_name"].tolist() == ["Barcelona", "Real Madrid"]
    assert teams.iloc[0]["alias_team_ids"] == [81, 529]


def test_get_live_league_tables_groups_by_competition(monkeypatch):
    monkeypatch.setattr(
        "dashboard.data.dashboard_data._read_sql",
        lambda query, params=None: pd.DataFrame(
            [
                {
                    "competition_id": 2014,
                    "competition_name": "Primera Division",
                    "season": 2025,
                    "matchday": 26,
                    "team_id": 81,
                    "team_name": "Barcelona",
                    "short_name": "Barca",
                    "crest_url": None,
                    "position": 1,
                    "points": 57,
                    "played_games": 26,
                    "won": 18,
                    "draw": 3,
                    "lost": 5,
                    "goals_for": 71,
                    "goals_against": 25,
                    "goal_difference": 46,
                },
                {
                    "competition_id": 2021,
                    "competition_name": "Premier League",
                    "season": 2025,
                    "matchday": 27,
                    "team_id": 65,
                    "team_name": "Manchester City",
                    "short_name": "Man City",
                    "crest_url": None,
                    "position": 1,
                    "points": 61,
                    "played_games": 27,
                    "won": 19,
                    "draw": 4,
                    "lost": 4,
                    "goals_for": 60,
                    "goals_against": 24,
                    "goal_difference": 36,
                },
            ]
        ),
    )

    league_tables = get_live_league_tables()

    assert set(league_tables) == {"Premier League", "LaLiga"}
    assert int(league_tables["LaLiga"].iloc[0]["matchday"]) == 26
