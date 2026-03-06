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
    get_team_xg_proxy,
    get_team_meta,
)
from state.filters import render_global_filters, require_team_selection
from ui.charts import render_form_chart, render_home_away_chart, render_position_curve
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_result_strip, render_section_heading, render_team_header
from ui.exports import render_csv_download
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


def _format_goal_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Venue" in out.columns:
        out["Venue"] = out["Venue"].replace({"HOME": "Domicile", "AWAY": "Exterieur"})
    return out


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


def _prepare_venue_form(matches: pd.DataFrame, venue: str, limit: int) -> pd.DataFrame:
    if matches.empty:
        return matches
    scoped = matches[matches["venue"].astype(str).str.upper() == venue.upper()].copy()
    if scoped.empty:
        return scoped
    return _prepare_form_df(scoped, limit=limit)


def _form_points(form_df: pd.DataFrame) -> int:
    if form_df.empty:
        return 0
    return int(form_df["points"].sum())


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


def _match_options(df: pd.DataFrame) -> dict[int, str]:
    if df.empty:
        return {}
    source = df.reset_index(drop=True)
    options: dict[int, str] = {}
    for index, row in source.iterrows():
        match_id = int(source.iloc[index]["match_id"])
        label = f"{row['team_name']} vs {row['opponent_name']}"
        options[match_id] = label
    return options


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

    render_section_heading("Forme domicile vs exterieur", "Comparaison de dynamique sur 5 et 10 matchs par contexte.")
    h5 = _prepare_venue_form(perspective, venue="Home", limit=5)
    a5 = _prepare_venue_form(perspective, venue="Away", limit=5)
    h10 = _prepare_venue_form(perspective, venue="Home", limit=10)
    a10 = _prepare_venue_form(perspective, venue="Away", limit=10)
    vh, va = st.columns(2)
    with vh:
        st.markdown("<div class='fdp-section-title' style='margin-top:0;'>Domicile</div>", unsafe_allow_html=True)
        st.caption("5 derniers matchs a domicile")
        render_result_strip(h5["result"].dropna().tolist())
        m1, m2 = st.columns(2)
        m1.metric("Points (5)", _form_points(h5))
        m2.metric("Points (10)", _form_points(h10))
    with va:
        st.markdown("<div class='fdp-section-title' style='margin-top:0;'>Exterieur</div>", unsafe_allow_html=True)
        st.caption("5 derniers matchs a l'exterieur")
        render_result_strip(a5["result"].dropna().tolist())
        m1, m2 = st.columns(2)
        m1.metric("Points (5)", _form_points(a5))
        m2.metric("Points (10)", _form_points(a10))

    comparison = pd.DataFrame(
        [
            {"Contexte": "Domicile", "Points 5": _form_points(h5), "Points 10": _form_points(h10)},
            {"Contexte": "Exterieur", "Points 5": _form_points(a5), "Points 10": _form_points(a10)},
        ]
    )
    st.bar_chart(comparison.set_index("Contexte")[["Points 5", "Points 10"]])
    render_csv_download(
        df=comparison,
        label="Export forme domicile/exterieur (CSV)",
        filename="team_home_away_form_points.csv",
        key="team_export_home_away_form",
    )

    render_section_heading("Domicile / Exterieur")
    split = get_home_away_split(filters.competition_id, filters.season, filters.team_id, (filters.date_start, filters.date_end))
    if split.empty:
        st.info("Aucun match joue pour calculer le split domicile/exterieur.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            render_home_away_chart(split)
        with c2:
            render_adaptive_table(
                _format_goal_split(split),
                title="Synthese split",
                badge_columns={"Venue": "venue"},
                strong_columns={"GoalsFor", "GoalsAgainst", "Points"},
                max_height=360,
            )
        render_csv_download(
            df=_format_goal_split(split),
            label="Export split domicile/exterieur (CSV)",
            filename="team_home_away_split.csv",
            key="team_export_split",
        )

    render_section_heading("xG / xGA (proxy)", "Approximation shots-based sur les matchs termines du club.")
    xg_proxy = get_team_xg_proxy(
        competition_id=filters.competition_id,
        season=filters.season,
        team_id=filters.team_id,
        limit=12,
    )
    if xg_proxy.empty:
        st.info("xG proxy indisponible (pas assez de donnees de tirs joueurs sur ce filtre).")
    else:
        chart_df = xg_proxy[["match_date", "xg_for", "xga", "goals_for", "goals_against"]].copy()
        chart_df["match_date"] = pd.to_datetime(chart_df["match_date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["match_date"]).set_index("match_date")
        st.line_chart(chart_df[["xg_for", "xga"]])
        st.caption("Reference score reel sur la meme periode.")
        st.bar_chart(chart_df[["goals_for", "goals_against"]])
        render_csv_download(
            df=xg_proxy,
            label="Export xG proxy (CSV)",
            filename="team_xg_proxy.csv",
            key="team_export_xg_proxy",
        )

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
        if recent_view.empty:
            st.info("Aucun match recent.")
        else:
            recent_table = _format_team_calendar(recent_view)
            render_adaptive_table(
                recent_table,
                title="Derniers matchs",
                badge_columns={"status": "status", "venue": "venue", "result": "result"},
                strong_columns={"opponent_name"},
                max_height=760,
            )
            render_csv_download(
                df=recent_table,
                label="Export derniers matchs (CSV)",
                filename="team_recent_matches.csv",
                key="team_export_recent_matches",
            )
            _render_match_detail_entry(recent_view, "team_recent")
    with cal_right:
        if upcoming_view.empty:
            st.info("Aucun match a venir.")
        else:
            upcoming_table = _format_team_calendar(upcoming_view)
            render_adaptive_table(
                upcoming_table,
                title="Matchs a venir",
                badge_columns={"status": "status", "venue": "venue", "result": "result"},
                strong_columns={"opponent_name"},
                max_height=640,
            )
            render_csv_download(
                df=upcoming_table,
                label="Export matchs a venir (CSV)",
                filename="team_upcoming_matches.csv",
                key="team_export_upcoming_matches",
            )
            _render_match_detail_entry(upcoming_view, "team_upcoming")

    render_section_heading("Courbe de classement")
    curve = get_standings_curve(filters.competition_id, filters.season, filters.team_id)
    if curve.empty:
        st.info("Pas de donnees de classement pour cette equipe. Relance la synchronisation des donnees si besoin.")
    else:
        render_position_curve(curve)


if __name__ == "__main__":
    main()

