import pandas as pd
import streamlit as st

from data.dashboard_data import (
    build_perspective_table,
    describe_season_source,
    get_home_away_split,
    get_kpis,
    get_matches,
    get_recent_matches,
    get_standings_curve,
    get_team_meta,
)
from state.filters import render_global_filters, require_team_selection
from ui.charts import render_form_chart, render_home_away_chart, render_position_curve
from ui.display import render_note_card, render_page_banner, render_result_strip, render_section_heading, render_team_header
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="TEAM - Football Data Platform", layout="wide")


def _render_goal_cards(kpis: dict[str, int | float | None]) -> None:
    st.markdown(
        f"""
        <div class="fdp-signal-grid">
          <div class="fdp-signal-card">
            <div class="fdp-signal-label">Goals For</div>
            <div class="fdp-signal-value" style="color:#168a5b;">{int(kpis["goals_for"])}</div>
            <div class="fdp-signal-sub">Buts marques sur le perimetre filtre</div>
          </div>
          <div class="fdp-signal-card">
            <div class="fdp-signal-label">Goals Against</div>
            <div class="fdp-signal-value" style="color:#cf3f4f;">{int(kpis["goals_against"])}</div>
            <div class="fdp-signal-sub">Buts encaisses sur le perimetre filtre</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style_goal_split(df: pd.DataFrame):
    if df.empty:
        return df.style

    return (
        df.style.format({"GoalsFor": "{:.0f}", "GoalsAgainst": "{:.0f}", "Points": "{:.0f}", "Matches": "{:.0f}"})
        .map(lambda _: "color:#168a5b;font-weight:700;", subset=["GoalsFor"])
        .map(lambda _: "color:#cf3f4f;font-weight:700;", subset=["GoalsAgainst"])
    )


def _prepare_form_df(matches: pd.DataFrame, limit: int) -> pd.DataFrame:
    if matches.empty:
        return matches
    form = (
        matches.dropna(subset=["result"])
        .sort_values(["date_dt", "match_id"], ascending=[False, False])
        .head(limit)
        .sort_values(["date_dt", "match_id"])
        .copy()
    )
    form["match_label"] = form["date_dt"].dt.strftime("%m-%d")
    return form


def _format_team_calendar(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    kickoff = (
        pd.to_datetime(out["kickoff_utc"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    fallback = pd.to_datetime(out["date_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["kickoff"] = kickoff.fillna(fallback).fillna("Unknown")
    out["score"] = out.apply(
        lambda row: "-" if pd.isna(row["goals_for"]) or pd.isna(row["goals_against"]) else f"{int(row['goals_for'])}-{int(row['goals_against'])}",
        axis=1,
    )
    out["result"] = out["result"].fillna("-")
    out["status"] = out["status"].fillna("UNKNOWN")
    out["matchday"] = out["matchday"].fillna("--")
    return out[["kickoff", "matchday", "status", "venue", "opponent_name", "score", "result", "points"]]


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "TEAM",
        "Lecture club par club: forme recente, split domicile/exterieur, calendrier et trajectoire au classement.",
        "Team.png",
    )
    filters = render_global_filters("team")
    render_note_card(describe_season_source(filters.season))
    filters = require_team_selection(filters)
    if filters is None:
        return

    team = get_team_meta(filters.team_id)
    render_team_header(team)

    kpis = get_kpis(filters.competition_id, filters.season, filters.team_id, (filters.date_start, filters.date_end))
    top = st.columns(4)
    top[0].metric("Points", kpis["points"])
    top[1].metric("Matches", kpis["matches"])
    top[2].metric("Goal Diff", kpis["goal_diff"])
    top[3].metric("Win Rate", "-" if kpis["win_rate"] is None else f"{kpis['win_rate']}%")
    _render_goal_cards(kpis)

    base_matches = get_matches(filters.competition_id, filters.season, filters.team_id, filters.date_start, filters.date_end)
    perspective = build_perspective_table(base_matches, team_id=filters.team_id)
    form5 = _prepare_form_df(perspective, 5)
    form10 = _prepare_form_df(perspective, 10)

    render_section_heading("Forme recente", "Une lecture courte sur 5 matchs et une dynamique plus longue sur 10 matchs.")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("<div class='fdp-section-title' style='margin-top:0;'>Forme 5 matchs</div>", unsafe_allow_html=True)
        render_result_strip(form5["result"].dropna().tolist())
        st.metric("Points (5)", int(form5["points"].sum()) if not form5.empty else 0)
    with right:
        st.markdown("<div class='fdp-section-title' style='margin-top:0;'>Forme 10 matchs</div>", unsafe_allow_html=True)
        render_form_chart(form10)

    render_section_heading("Domicile / Exterieur")
    split = get_home_away_split(filters.competition_id, filters.season, filters.team_id, (filters.date_start, filters.date_end))
    if split.empty:
        st.info("Aucun match joue pour calculer le split domicile/exterieur.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            render_home_away_chart(split)
        with c2:
            st.dataframe(_style_goal_split(split), hide_index=True, use_container_width=True)

    render_section_heading("Calendrier", "Derniers matchs et prochaines affiches pour le club selectionne.")
    recent, upcoming = get_recent_matches(
        competition_id=filters.competition_id,
        season=filters.season,
        team_id=filters.team_id,
        date_range=(filters.date_start, filters.date_end),
        recent_limit=10,
        upcoming_limit=5,
    )
    recent_view = build_perspective_table(recent, team_id=filters.team_id) if not recent.empty else pd.DataFrame()
    upcoming_view = build_perspective_table(upcoming, team_id=filters.team_id) if not upcoming.empty else pd.DataFrame()

    cal_left, cal_right = st.columns(2)
    with cal_left:
        st.caption("Derniers matchs")
        if recent_view.empty:
            st.info("Aucun match recent.")
        else:
            st.dataframe(_format_team_calendar(recent_view), hide_index=True, use_container_width=True)
    with cal_right:
        st.caption("Matchs a venir")
        if upcoming_view.empty:
            st.info("Aucun match a venir.")
        else:
            st.dataframe(_format_team_calendar(upcoming_view), hide_index=True, use_container_width=True)

    render_section_heading("Courbe de classement")
    curve = get_standings_curve(filters.competition_id, filters.season, filters.team_id)
    if curve.empty:
        st.info("Pas de donnees de classement pour cette equipe. Relance la synchronisation des donnees si besoin.")
    else:
        render_position_curve(curve)


if __name__ == "__main__":
    main()
