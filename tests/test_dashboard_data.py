import sys
import types

import pandas as pd

# Minimal stub so we can import dashboard data helpers without installing Streamlit in local test env.
if "streamlit" not in sys.modules:
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_data = lambda **kwargs: (lambda fn: fn)
    sys.modules["streamlit"] = streamlit_stub

from dashboard.data.dashboard_data import (
    build_local_league_table,
    build_team_match_view,
    compute_team_kpis,
)


def _sample_matches() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "match_id": 1,
                "date_id": "2025-08-10",
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
                "home_team_id": 10,
                "home_team": "FC Alpha",
                "away_team_id": 30,
                "away_team": "FC Gamma",
                "home_score": None,
                "away_score": None,
            },
        ]
    )
    df["date_dt"] = pd.to_datetime(df["date_id"], errors="coerce")
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
