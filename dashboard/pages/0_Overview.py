import pandas as pd
import streamlit as st

from data.dashboard_data import describe_season_source, get_current_standings, get_kpis, get_recent_matches, get_standings_curve
from state.filters import render_global_filters
from ui.charts import render_position_curve
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_section_heading
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="OVERVIEW - Football Data Platform", layout="wide")


def _trend_symbol(delta: object) -> str:
    if pd.isna(delta):
        return "="
    delta_value = int(delta)
    if delta_value > 0:
        return f"↑{delta_value}"
    if delta_value < 0:
        return f"↓{abs(delta_value)}"
    return "="


def _format_match_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.copy()
    kickoff = (
        pd.to_datetime(table["kickoff_utc"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    fallback = pd.to_datetime(table["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    table["kickoff"] = kickoff.fillna(fallback).fillna("Unknown")
    table["score"] = table.apply(
        lambda row: "-" if pd.isna(row["home_score"]) or pd.isna(row["away_score"]) else f"{int(row['home_score'])}-{int(row['away_score'])}",
        axis=1,
    )
    table["status"] = table["status"].fillna("UNKNOWN")
    table["matchday"] = table["matchday"].fillna("--")
    return table[["kickoff", "status", "matchday", "home_team", "score", "away_team"]]


def _match_options(df: pd.DataFrame) -> dict[int, str]:
    if df.empty:
        return {}
    source = df.reset_index(drop=True)
    options: dict[int, str] = {}
    for index, row in source.iterrows():
        match_id = int(source.iloc[index]["match_id"])
        label = f"{row['home_team']} vs {row['away_team']}"
        options[match_id] = label
    return options


def _overview_standing_row_class(row: pd.Series) -> str:
    try:
        position = int(row["Pos"])
    except Exception:
        return ""
    if position <= 4:
        return "fdp-row-top"
    return ""


def _render_match_detail_entry(df: pd.DataFrame, key_prefix: str) -> None:
    options = _match_options(df)
    if not options:
        return
    match_ids = list(options.keys())
    with st.form(key=f"{key_prefix}_match_detail_form", border=False):
        selection = st.selectbox(
            "Open match detail",
            match_ids,
            key=f"{key_prefix}_match_detail",
            format_func=lambda match_id: options[int(match_id)],
        )
        submitted = st.form_submit_button("Go to MATCH DETAIL")
    if submitted:
        match_id = int(selection)
        st.session_state["selected_match_id"] = match_id
        st.query_params["match_id"] = str(match_id)
        st.switch_page("pages/5_MATCH_DETAIL.py")


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "OVERVIEW",
        "Vue analytique filtree: KPIs, calendrier et dernier classement disponible pour le perimetre selectionne.",
        "Overview.png",
    )
    filters = render_global_filters("overview")
    render_note_card(describe_season_source(filters.season))

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
    render_section_heading(
        "Dernier classement disponible",
        "Le classement depend surtout de la competition et de la saison choisies. La plage de dates agit surtout sur les KPI et le calendrier.",
    )
    if standings.empty:
        st.info("Aucun snapshot de classement disponible. Relance la synchronisation des donnees pour alimenter cette vue.")
    else:
        standings_display = standings.rename(
            columns={
                "team_name": "Equipe",
                "position": "Pos",
                "position_delta": "Trend",
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
        standings_display["Trend"] = standings_display["Trend"].map(_trend_symbol)
        render_adaptive_table(
            standings_display[["Pos", "Trend", "Equipe", "Pts", "MJ", "G", "N", "P", "BP", "BC", "Diff"]],
            badge_columns={"Trend": "trend"},
            row_class_renderer=_overview_standing_row_class,
            strong_columns={"Equipe"},
            max_height=980,
        )

    render_section_heading("Position au fil des journees")
    curve = get_standings_curve(filters.competition_id, filters.season, filters.team_id)
    if curve.empty:
        st.info("Pas de donnees de classement disponibles pour ce filtre.")
    else:
        render_position_curve(curve)

    render_section_heading("Calendrier", "Derniers matchs joues et prochaines affiches sur le scope filtre.")
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
        if recent_matches.empty:
            st.info("Aucun match recent disponible.")
        else:
            render_adaptive_table(
                _format_match_table(recent_matches),
                title="Derniers 10 matchs",
                badge_columns={"status": "status"},
                strong_columns={"home_team", "away_team"},
                max_height=760,
            )
            _render_match_detail_entry(recent_matches, "overview_recent")
    with right:
        if upcoming_matches.empty:
            st.info("Aucun match a venir sur cette plage.")
        else:
            render_adaptive_table(
                _format_match_table(upcoming_matches),
                title="Prochains 5 matchs",
                badge_columns={"status": "status"},
                strong_columns={"home_team", "away_team"},
                max_height=640,
            )
            _render_match_detail_entry(upcoming_matches, "overview_upcoming")


if __name__ == "__main__":
    main()
