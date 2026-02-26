import pandas as pd

from dashboard.data.study_players_helpers import (
    add_podium_icons_generic,
    build_study_leaders_scope,
    season_label_from_start,
    season_picker_label_from_start,
)


def test_season_label_helpers():
    assert season_label_from_start(2024) == "2024-2025"
    assert season_picker_label_from_start(2024) == "2024/25"


def test_add_podium_icons_generic_adds_top3_only():
    df = pd.DataFrame({"Joueur": ["A", "B", "C", "D"], "Score": [10, 9, 8, 7]})
    out = add_podium_icons_generic(df, "Joueur")
    assert out.loc[0, "Joueur"].startswith("🥇 ")
    assert out.loc[1, "Joueur"].startswith("🥈 ")
    assert out.loc[2, "Joueur"].startswith("🥉 ")
    assert out.loc[3, "Joueur"] == "D"


def test_build_study_leaders_scope_aggregates_all_seasons_and_multi_club():
    season_df = pd.DataFrame(
        [
            {
                "season_start": 2023,
                "player_id": 1,
                "player_name": "Player One",
                "team_name": "Club A",
                "position_group": "FWD",
                "minutes_total": 900,
                "matches_played": 10,
                "starts": 10,
                "goals_total": 5,
                "assists_total": 2,
                "ga_total": 7,
                "yellow_cards_total": 1,
            },
            {
                "season_start": 2024,
                "player_id": 1,
                "player_name": "Player One",
                "team_name": "Club B",
                "position_group": "FWD",
                "minutes_total": 450,
                "matches_played": 6,
                "starts": 4,
                "goals_total": 3,
                "assists_total": 1,
                "ga_total": 4,
                "yellow_cards_total": 2,
            },
            {
                "season_start": 2024,
                "player_id": 2,
                "player_name": "Player Two",
                "team_name": "Club C",
                "position_group": "MID",
                "minutes_total": 1200,
                "matches_played": 15,
                "starts": 14,
                "goals_total": 2,
                "assists_total": 6,
                "ga_total": 8,
                "yellow_cards_total": 3,
            },
        ]
    )

    out = build_study_leaders_scope(season_df, selected_season=None, selected_pos="Tous")

    assert set(out["player_id"]) == {1, 2}
    p1 = out[out["player_id"] == 1].iloc[0]
    assert int(p1["minutes_total"]) == 1350
    assert int(p1["goals_total"]) == 8
    assert int(p1["assists_total"]) == 3
    assert int(p1["clubs_count"]) == 2
    assert bool(p1["is_multi_club_season"]) is True
    assert "Club A" in p1["clubs_list"] and "Club B" in p1["clubs_list"]
    assert p1["team_name"] == "Club A"  # club principal (plus de minutes)
    assert round(float(p1["goals_p90"]), 2) == round(8 * 90 / 1350, 2)


def test_build_study_leaders_scope_filters_selected_season_and_position():
    season_df = pd.DataFrame(
        [
            {"season_start": 2024, "player_id": 1, "player_name": "A", "team_name": "X", "position_group": "FWD", "minutes_total": 100},
            {"season_start": 2024, "player_id": 2, "player_name": "B", "team_name": "Y", "position_group": "MID", "minutes_total": 200},
            {"season_start": 2023, "player_id": 3, "player_name": "C", "team_name": "Z", "position_group": "FWD", "minutes_total": 300},
        ]
    )
    out = build_study_leaders_scope(season_df, selected_season=2024, selected_pos="FWD")
    assert list(out["player_id"]) == [1]
    assert int(out.iloc[0]["clubs_count"]) == 1
