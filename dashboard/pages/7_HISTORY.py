from __future__ import annotations

import pandas as pd
import streamlit as st

from data.dashboard_data import get_competitions, get_season_champions, get_season_history
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_section_heading
from ui.exports import render_csv_download
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="HISTORY - Football Data Platform", layout="wide")


def _season_label(value: int) -> str:
    return f"{int(value)}-{int(value) + 1}"


def _competition_selector() -> int | None:
    competitions = get_competitions()
    if competitions.empty:
        return 2014

    scoped = competitions.dropna(subset=["competition_id"]).copy()
    if scoped.empty:
        return 2014

    scoped["competition_id"] = scoped["competition_id"].astype(int)
    scoped["label"] = scoped["competition_name"].astype(str)
    scoped = scoped.sort_values("label")
    labels = scoped["label"].tolist()
    id_by_label = dict(zip(scoped["label"], scoped["competition_id"]))

    default_index = 0
    for idx, cid in enumerate(scoped["competition_id"].tolist()):
        if int(cid) == 2014:
            default_index = idx
            break

    selected_label = st.selectbox("Competition", labels, index=default_index, key="history_competition")
    return int(id_by_label[selected_label])


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "HISTORY",
        "Historique des classements de fin de saison: champions, top 4 et comparaison saison contre saison.",
        None,
    )
    render_note_card("Cette page lit les snapshots finaux en base pour comparer les saisons entre elles.")

    competition_id = _competition_selector()
    history = get_season_history(competition_id=competition_id)
    if history.empty:
        st.info("Aucun historique de classement disponible pour cette competition.")
        return

    champions = get_season_champions(competition_id=competition_id)
    render_section_heading("Champions par saison")
    champs_table = champions.rename(
        columns={
            "season_label": "Saison",
            "team_name": "Champion",
            "points": "Pts",
            "played_games": "MJ",
        }
    )[["Saison", "Champion", "Pts", "MJ"]]
    render_adaptive_table(champs_table, title="Palmares", strong_columns={"Champion"}, max_height=600)
    render_csv_download(
        df=champs_table,
        label="Export palmares (CSV)",
        filename="history_champions.csv",
        key="history_export_champions",
    )

    seasons = sorted(history["season"].dropna().astype(int).unique().tolist(), reverse=True)
    if len(seasons) < 2:
        return

    render_section_heading("Comparaison de deux saisons")
    col_a, col_b = st.columns(2)
    with col_a:
        season_a = st.selectbox("Saison A", seasons, format_func=_season_label, index=0, key="history_season_a")
    with col_b:
        default_b = 1 if len(seasons) > 1 else 0
        season_b = st.selectbox("Saison B", seasons, format_func=_season_label, index=default_b, key="history_season_b")

    season_a_label = _season_label(int(season_a))
    season_b_label = _season_label(int(season_b))
    pos_a_col = f"Pos {season_a_label}"
    pos_b_col = f"Pos {season_b_label}"
    pts_a_col = f"Pts {season_a_label}"
    pts_b_col = f"Pts {season_b_label}"

    table_a = history[history["season"].astype(int) == int(season_a)][["team_name", "position", "points"]].copy()
    table_b = history[history["season"].astype(int) == int(season_b)][["team_name", "position", "points"]].copy()
    table_a = table_a.rename(columns={"position": pos_a_col, "points": pts_a_col})
    table_b = table_b.rename(columns={"position": pos_b_col, "points": pts_b_col})
    merged = table_a.merge(table_b, on="team_name", how="outer")
    merged["Ecart position (B-A)"] = merged[pos_b_col] - merged[pos_a_col]
    merged["Ecart points (A-B)"] = merged[pts_a_col] - merged[pts_b_col]
    merged = merged.sort_values([pos_a_col, "team_name"], na_position="last").rename(columns={"team_name": "Equipe"})
    st.caption(
        "Lecture: `Ecart position (B-A) < 0` = meilleure position en saison B. "
        "`Ecart points (A-B) > 0` = plus de points en saison A."
    )
    render_adaptive_table(
        merged[
            [
                "Equipe",
                pos_a_col,
                pts_a_col,
                pos_b_col,
                pts_b_col,
                "Ecart position (B-A)",
                "Ecart points (A-B)",
            ]
        ],
        title=f"{season_a_label} vs {season_b_label}",
        badge_columns={
            "Ecart position (B-A)": "delta_position",
            "Ecart points (A-B)": "delta",
        },
        strong_columns={"Equipe"},
        max_height=840,
    )
    render_csv_download(
        df=merged,
        label="Export comparaison saisons (CSV)",
        filename=f"history_compare_{season_a}_{season_b}.csv",
        key="history_export_comparison",
    )

    render_section_heading("Trajectoire d'un club dans le temps")
    teams = sorted(history["team_name"].dropna().astype(str).unique().tolist())
    selected_team = st.selectbox("Club", teams, key="history_team_select")
    team_history = history[history["team_name"].astype(str) == str(selected_team)].copy()
    if team_history.empty:
        st.info("Aucune donnee pour ce club.")
    else:
        team_history = team_history.sort_values("season")
        team_history["Saison"] = team_history["season"].astype(int).map(_season_label)
        team_history["Position"] = team_history["position"].astype(float)
        team_history["Points"] = team_history["points"].astype(float)
        st.caption("Position: plus le chiffre est bas, meilleur est le classement.")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.line_chart(team_history.set_index("Saison")[["Position"]])
        with chart_right:
            st.line_chart(team_history.set_index("Saison")[["Points"]])
        render_csv_download(
            df=team_history[["Saison", "Position", "Points", "played_games", "goals_for", "goals_against"]],
            label="Export trajectoire club (CSV)",
            filename=f"history_team_{selected_team.lower().replace(' ', '_')}.csv",
            key="history_export_team",
        )


if __name__ == "__main__":
    main()

