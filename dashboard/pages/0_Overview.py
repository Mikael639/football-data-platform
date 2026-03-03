import pandas as pd
import streamlit as st

from data.dashboard_data import describe_season_source, get_current_standings, get_kpis, get_recent_matches, get_standings_curve
from state.filters import render_global_filters
from ui.charts import render_position_curve
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="Overview - Football Data Platform", layout="wide")


def _format_match_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.copy()
    kickoff = pd.to_datetime(table["kickoff_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    fallback = pd.to_datetime(table["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    table["kickoff"] = kickoff.fillna(fallback).fillna("Unknown")
    table["score"] = table.apply(
        lambda row: "-" if pd.isna(row["home_score"]) or pd.isna(row["away_score"]) else f"{int(row['home_score'])}-{int(row['away_score'])}",
        axis=1,
    )
    table["status"] = table["status"].fillna("UNKNOWN")
    table["matchday"] = table["matchday"].fillna("—")
    return table[["kickoff", "status", "matchday", "home_team", "score", "away_team"]]


def main() -> None:
    inject_dashboard_styles()
    st.title("Overview")
    filters = render_global_filters("overview")
    st.caption(describe_season_source(filters.season))

    kpis = get_kpis(
        competition_id=filters.competition_id,
        season=filters.season,
        team_id=filters.team_id,
        date_range=(filters.date_start, filters.date_end),
    )

    cols = st.columns(5)
    cols[0].metric("Matches", kpis["matches"])
    cols[1].metric("Goals For", kpis["goals_for"])
    cols[2].metric("Goals Against", kpis["goals_against"])
    cols[3].metric("Goal Diff", kpis["goal_diff"])
    cols[4].metric("Win Rate", "-" if kpis["win_rate"] is None else f"{kpis['win_rate']}%")

    standings = get_current_standings(filters.competition_id, filters.season)
    st.subheader("Classement courant")
    if standings.empty:
        st.info("Aucun snapshot de classement disponible. Lance le pipeline avec DATA_MODE=csv ou DATA_MODE=api.")
    else:
        standings_display = standings.rename(
            columns={
                "team_name": "Equipe",
                "position": "Pos",
                "points": "Pts",
                "played_games": "MJ",
                "won": "G",
                "draw": "N",
                "lost": "P",
                "goals_for": "BP",
                "goals_against": "BC",
                "goal_difference": "Diff",
            }
        )
        st.dataframe(
            standings_display[["Pos", "Equipe", "Pts", "MJ", "G", "N", "P", "BP", "BC", "Diff"]],
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Position au fil des journees")
    curve = get_standings_curve(filters.competition_id, filters.season, filters.team_id)
    if curve.empty:
        st.info("Pas de donnees de classement disponibles pour ce filtre.")
    else:
        render_position_curve(curve)

    st.subheader("Calendrier")
    recent_matches, upcoming_matches = get_recent_matches(
        competition_id=filters.competition_id,
        season=filters.season,
        team_id=filters.team_id,
        date_range=(filters.date_start, filters.date_end),
        recent_limit=10,
        upcoming_limit=5,
    )
    left, right = st.columns(2)
    with left:
        st.caption("Derniers 10 matchs")
        if recent_matches.empty:
            st.info("Aucun match recent disponible.")
        else:
            st.dataframe(_format_match_table(recent_matches), hide_index=True, use_container_width=True)
    with right:
        st.caption("Prochains 5 matchs")
        if upcoming_matches.empty:
            st.info("Aucun match a venir sur cette plage.")
        else:
            st.dataframe(_format_match_table(upcoming_matches), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
