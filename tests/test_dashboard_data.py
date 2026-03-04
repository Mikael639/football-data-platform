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
    get_european_competitions,
    get_historical_players_catalog,
    get_match_detail,
    get_historical_players_from_standard_csv,
    get_live_league_tables,
    get_players,
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


def test_get_european_competitions_filters_uefa_competitions(monkeypatch):
    monkeypatch.setattr(
        "dashboard.data.dashboard_data.get_competitions",
        lambda: pd.DataFrame(
            [
                {"competition_id": 2001, "competition_name": "Champions League"},
                {"competition_id": 2003, "competition_name": "Europa League"},
                {"competition_id": 2014, "competition_name": "LaLiga"},
            ]
        ),
    )

    competitions = get_european_competitions()

    assert competitions["competition_name"].tolist() == ["Champions League", "Europa League"]


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


def test_get_players_prefers_season_filtered_player_stats(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_read_sql(query, params=None):
        calls.append((query, params or {}))
        if "FROM fact_player_match_stats" in query:
            return pd.DataFrame(
                [
                    {
                        "player_id": 7,
                        "full_name": "Karim Benzema",
                        "position": "FW",
                        "nationality": None,
                        "birth_date": None,
                        "team_id": 86,
                        "team_name": "Real Madrid",
                    }
                ]
            )
        return pd.DataFrame()

    monkeypatch.setattr("dashboard.data.dashboard_data._read_sql", fake_read_sql)
    monkeypatch.setattr(
        "dashboard.data.dashboard_data.get_team_alias_ids",
        lambda team_id, competition_id=None, season=None: [int(team_id)],
    )

    players = get_players(team_id=86, competition_id=2014, season="2020-2021")

    assert not players.empty
    assert players.iloc[0]["full_name"] == "Karim Benzema"
    assert any("FROM fact_player_match_stats" in query for query, _ in calls)


def test_get_historical_players_from_standard_csv_filters_team_and_season(tmp_path, monkeypatch):
    csv_path = tmp_path / "laliga_2020_2021_standard.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Temps de jeu;;;;Performances",
                "RK;Joueur;Nation;Pos;Effectif;Âge;Né;MP",
                "1;Karim Benzema;fr FRA;FW;Real Madrid;33;1987;34",
                "2;Luka Modrić;hr CRO;MF;Real Madrid CF;35;1985;33",
                "3;Lionel Messi;ar ARG;FW;Barcelona;33;1987;35",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("dashboard.data.dashboard_data._season_standard_csv_path", lambda season: csv_path)
    monkeypatch.setattr(
        "dashboard.data.dashboard_data.get_team_alias_groups",
        lambda competition_id=None, season=None: pd.DataFrame(
            [
                {
                    "team_id": 86,
                    "team_name": "Real Madrid",
                    "alias_names": ["Real Madrid", "Real Madrid CF"],
                }
            ]
        ),
    )

    players = get_historical_players_from_standard_csv(team_id=86, season="2020-2021")

    assert players["full_name"].tolist() == ["Karim Benzema", "Luka Modrić"]
    assert players["team_name"].tolist() == ["Real Madrid", "Real Madrid CF"]


def test_get_historical_players_from_standard_csv_accepts_alias_team_id(tmp_path, monkeypatch):
    csv_path = tmp_path / "laliga_2020_2021_standard.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Temps de jeu;;;;Performances",
                "RK;Joueur;Nation;Pos;Effectif;Âge;Né;MP",
                "1;Karim Benzema;fr FRA;FW;Real Madrid;33;1987;34",
                "2;Luka Modrić;hr CRO;MF;Real Madrid CF;35;1985;33",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("dashboard.data.dashboard_data._season_standard_csv_path", lambda season: csv_path)
    monkeypatch.setattr(
        "dashboard.data.dashboard_data.get_team_alias_groups",
        lambda competition_id=None, season=None: pd.DataFrame(
            [
                {
                    "team_id": 86,
                    "team_name": "Real Madrid",
                    "alias_names": ["Real Madrid", "Real Madrid CF"],
                    "alias_team_ids": [86, 541],
                }
            ]
        ),
    )

    players = get_historical_players_from_standard_csv(team_id=541, season="2020-2021")

    assert players["full_name"].tolist() == ["Karim Benzema", "Luka Modrić"]


def test_get_historical_players_catalog_deduplicates_all_seasons(tmp_path, monkeypatch):
    season_a = tmp_path / "laliga_2020_2021_standard.csv"
    season_b = tmp_path / "laliga_2021_2022_standard.csv"
    season_a.write_text(
        "\n".join(
            [
                "Temps de jeu;;;;Performances",
                "RK;Joueur;Nation;Pos;Effectif;Ã‚ge;NÃ©;MP",
                "1;Vinicius Junior;;;Real Madrid;20;;35",
                "2;Luka ModriÄ‡;hr CRO;MF;Real Madrid CF;35;1985;33",
            ]
        ),
        encoding="utf-8",
    )
    season_b.write_text(
        "\n".join(
            [
                "Temps de jeu;;;;Performances",
                "RK;Joueur;Nation;Pos;Effectif;Ã‚ge;NÃ©;MP",
                "1;Vinicius Junior;br BRA;FW;Real Madrid;21;2000;35",
                "2;Luka ModriÄ‡;hr CRO;CM;Real Madrid;36;1985;28",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("dashboard.data.dashboard_data._season_standard_csv_paths", lambda season: [season_a, season_b])
    monkeypatch.setattr("dashboard.data.dashboard_data._birth_year_column", lambda frame: frame.columns[6])
    monkeypatch.setattr(
        "dashboard.data.dashboard_data.get_team_alias_groups",
        lambda competition_id=None, season=None: pd.DataFrame(
            [
                {
                    "team_id": 86,
                    "team_name": "Real Madrid",
                    "alias_names": ["Real Madrid", "Real Madrid CF"],
                    "alias_team_ids": [86, 541],
                }
            ]
        ),
    )

    catalog_fn = getattr(get_historical_players_catalog, "__wrapped__", get_historical_players_catalog)
    players = catalog_fn(team_id=541, season=None)

    assert players["full_name"].tolist() == ["Luka ModriÄ‡", "Vinicius Junior"]
    vinicius = players[players["full_name"] == "Vinicius Junior"].iloc[0]
    assert vinicius["position"] == "FW"
    assert vinicius["nationality"] == "BRA"
    assert vinicius["birth_date"] == "2000-01-01"
    assert vinicius["team_name"] == "Real Madrid"


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


def test_get_match_detail_normalizes_competition_and_parses_kickoff(monkeypatch):
    monkeypatch.setattr(
        "dashboard.data.dashboard_data._read_sql",
        lambda query, params=None: pd.DataFrame(
            [
                {
                    "match_id": 1001,
                    "competition_id": 2014,
                    "competition_name": "Primera Division",
                    "season": "2025-2026",
                    "match_date": "2025-08-20",
                    "date_id": "2025-08-20",
                    "kickoff_utc": "2025-08-20T19:00:00Z",
                    "status": "FINISHED",
                    "matchday": 2,
                    "home_team_id": 81,
                    "home_team": "FC Barcelona",
                    "home_short_name": "Barca",
                    "home_crest_url": None,
                    "away_team_id": 86,
                    "away_team": "Real Madrid CF",
                    "away_short_name": "Real",
                    "away_crest_url": None,
                    "home_score": 2,
                    "away_score": 1,
                }
            ]
        ),
    )

    detail = get_match_detail(1001)

    assert detail is not None
    assert detail["competition_name"] == "LaLiga"
    assert str(detail["kickoff_utc"].tz) == "UTC"
    assert detail["home_score"] == 2
