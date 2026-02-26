import pandas as pd

from src.study_fbref import (
    add_match_features,
    build_player_season,
    build_progression,
    build_regularity,
    build_fbref_study_outputs_from_manual_csv,
)


def _sample_match_df() -> pd.DataFrame:
    rows = []
    # Player A (FWD) across two seasons
    for season, goals_per_match in [(2023, [1, 0, 1]), (2024, [1, 1, 1])]:
        for i, gls in enumerate(goals_per_match, start=1):
            rows.append(
                {
                    "season_start": season,
                    "match_id": f"{season}-A-{i}",
                    "date_id": f"{season+1}-01-0{i}",
                    "competition": "La Liga (FBref)",
                    "team_id": 1,
                    "team_name": "FC Example",
                    "player_id": 100,
                    "player_name": "Player A",
                    "player_key": "A",
                    "position": "FW",
                    "position_group": "FWD",
                    "is_starting": True,
                    "minutes": 90,
                    "goals": gls,
                    "assists": 0,
                    "shots": 3 + gls,
                    "passes": 18 + gls,
                    "pass_accuracy": 0.80,
                }
            )

    # Player B (FWD) same two seasons, less regular and lower progression
    for season, goals_per_match in [(2023, [0, 2, 0]), (2024, [0, 1, 0])]:
        for i, gls in enumerate(goals_per_match, start=1):
            rows.append(
                {
                    "season_start": season,
                    "match_id": f"{season}-B-{i}",
                    "date_id": f"{season+1}-02-0{i}",
                    "competition": "La Liga (FBref)",
                    "team_id": 1,
                    "team_name": "FC Example",
                    "player_id": 101,
                    "player_name": "Player B",
                    "player_key": "B",
                    "position": "FW",
                    "position_group": "FWD",
                    "is_starting": True,
                    "minutes": 90,
                    "goals": gls,
                    "assists": 0,
                    "shots": 4,
                    "passes": 15,
                    "pass_accuracy": 0.76,
                }
            )
    return pd.DataFrame(rows)


def test_fbref_study_builds_regularity_and_progression():
    df_match = add_match_features(_sample_match_df())
    df_player_season = build_player_season(df_match)
    df_reg = build_regularity(df_match, min_minutes=200)
    df_prog = build_progression(df_player_season, min_minutes=200)

    assert not df_player_season.empty
    assert not df_reg.empty
    assert not df_prog.empty

    top_2024_fwd = df_reg[(df_reg["season_start"] == 2024) & (df_reg["position_group"] == "FWD")].iloc[0]
    assert top_2024_fwd["podium"] == "🥇"
    assert top_2024_fwd["player_name"] in {"Player A", "Player B"}

    prog_2024 = df_prog[df_prog["season_start"] == 2024]
    assert set(prog_2024["player_name"]) == {"Player A", "Player B"}
    player_a = prog_2024[prog_2024["player_name"] == "Player A"].iloc[0]
    assert player_a["delta_goals_p90"] > 0


def test_manual_csv_mode_builds_outputs(tmp_path):
    manual_csv = tmp_path / "player_match_manual.csv"
    pd.DataFrame(
        [
            {
                "date_id": "2024-08-15",
                "team_name": "Club A",
                "player_name": "Player X",
                "minutes": 90,
                "goals": 1,
                "assists": 0,
                "shots": 3,
                "passes": 20,
                "pass_accuracy": "80%",
                "position": "FW",
                "is_starting": 1,
            },
            {
                "date_id": "2024-08-22",
                "team_name": "Club A",
                "player_name": "Player X",
                "minutes": 90,
                "goals": 0,
                "assists": 1,
                "shots": 2,
                "passes": 22,
                "pass_accuracy": "82%",
                "position": "FW",
                "is_starting": 1,
            },
            {
                "date_id": "2023-08-22",
                "team_name": "Club A",
                "player_name": "Player X",
                "minutes": 90,
                "goals": 0,
                "assists": 0,
                "shots": 1,
                "passes": 18,
                "pass_accuracy": "78%",
                "position": "FW",
                "is_starting": 1,
            },
        ]
    ).to_csv(manual_csv, index=False)

    outputs = build_fbref_study_outputs_from_manual_csv(player_match_csv=manual_csv, min_minutes=80)
    assert set(outputs.keys()) == {"player_match", "player_season", "regularity", "progression"}
    assert not outputs["player_match"].empty
    assert not outputs["player_season"].empty
