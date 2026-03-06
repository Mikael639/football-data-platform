from __future__ import annotations

import pandas as pd
import streamlit as st

from data.dashboard_data import get_competitions, get_pipeline_runs
from services.pipeline_control import get_pipeline_process_status, start_pipeline_process
from state.admin_access import is_admin_authenticated, render_admin_access_sidebar
from ui.display import asset_image_path, laliga_logo_source
from ui.styles import inject_dashboard_styles


st.set_page_config(page_title="APP - Football Data Platform", layout="wide")
inject_dashboard_styles()


def _safe_run_value(runs: pd.DataFrame, column: str, fallback: str) -> str:
    if runs.empty or column not in runs.columns:
        return fallback
    value = runs.iloc[0][column]
    if pd.isna(value):
        return fallback
    return str(value)


def _render_hero(competitions: pd.DataFrame) -> None:
    competition_names = ", ".join(competitions["competition_name"].dropna().astype(str).tolist()[:5]) or "Leagues"
    left, right = st.columns([6, 1.25], vertical_alignment="center")
    with left:
        st.markdown(
            f"""
            <div class="fdp-hero">
              <div class="fdp-home-ribbon">APP | MATCH CENTER</div>
              <div class="fdp-hero-title">Football Data Platform</div>
              <div class="fdp-hero-sub">
                Hub analytique pour suivre les ligues, les clubs et la sante de la plateforme depuis une seule interface.
              </div>
              <div class="fdp-chip-row">
                <span class="fdp-chip">Competitions: {competition_names}</span>
                <span class="fdp-chip">UI first</span>
                <span class="fdp-chip">Analytics + Live</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.image(laliga_logo_source(), width=170)


def _render_summary_cards(competitions: pd.DataFrame, runs: pd.DataFrame, *, is_admin: bool) -> None:
    latest_status = _safe_run_value(runs, "status", "No run")
    latest_started = _safe_run_value(runs, "started_at", "N/A")
    latest_duration = _safe_run_value(runs, "duration_ms", "N/A")
    pages_count = 8 if is_admin else 7
    pages_label = (
        "Overview, Team, Players, Live, Europe, Monitoring, History et Prediction."
        if is_admin
        else "Overview, Team, Players, Live, Europe, History et Prediction."
    )
    st.markdown(
        f"""
        <div class="fdp-home-summary-grid">
          <div class="fdp-home-summary-card">
            <div class="fdp-home-summary-kicker">Competitions</div>
            <div class="fdp-signal-value">{len(competitions.index)}</div>
            <div class="fdp-signal-sub">Ligues actuellement visibles dans la plateforme.</div>
          </div>
          <div class="fdp-home-summary-card">
            <div class="fdp-home-summary-kicker">Pages</div>
            <div class="fdp-signal-value">{pages_count}</div>
            <div class="fdp-signal-sub">{pages_label}</div>
          </div>
          <div class="fdp-home-summary-card">
            <div class="fdp-home-summary-kicker">Latest Status</div>
            <div class="fdp-signal-value">{latest_status}</div>
            <div class="fdp-signal-sub">Dernier run: {latest_started} | Duration: {latest_duration}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_page_tile(
    column: st.delta_generator.DeltaGenerator,
    title: str,
    description: str,
    chips: list[str],
    target: str,
    image_name: str | None,
) -> None:
    with column:
        image_path = asset_image_path(image_name) if image_name else None
        chips_html = "".join(f"<span class='fdp-chip'>{chip}</span>" for chip in chips)
        st.markdown("<div class='fdp-page-tile'>", unsafe_allow_html=True)
        if image_path:
            st.image(image_path, width=160)
        st.markdown(
            f"""
            <div class="fdp-page-card fdp-page-card-compact">
              <div class="fdp-page-tile-title">{title}</div>
              <div class="fdp-page-tile-text">{description}</div>
              <div class="fdp-chip-row">{chips_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(target, label=f"Open {title}")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_page_cards(*, is_admin: bool) -> None:
    top = st.columns(3)
    _render_page_tile(top[0], "OVERVIEW", "Vue analytique globale pour lire les KPIs, le calendrier et le dernier classement disponible.", ["KPIs", "Calendrier", "Classement"], "pages/0_OVERVIEW.py", "Overview.png")
    _render_page_tile(top[1], "TEAM", "Lecture club par club avec forme recente, splits domicile/exterieur et trajectoire au classement.", ["Forme", "Equipe", "Focus club"], "pages/1_TEAM.py", "Team.png")
    _render_page_tile(top[2], "LIVE LEAGUES", "Comparateur multi-ligues pour lire la table courante de chaque championnat disponible en base.", ["Multi-leagues", "Zones UEFA", "Live snapshot"], "pages/3_LIVE_LEAGUES.py", "Live Leagues.png")

    if is_admin:
        bottom = st.columns(3)
        _render_page_tile(bottom[0], "PLAYERS", "Effectif disponible en base avec les postes, nationalites et informations principales du club filtre.", ["Effectif", "Squad view", "Exploration"], "pages/2_PLAYERS.py", "Players.png")
        _render_page_tile(bottom[1], "EUROPE", "Suivi UEFA dedie pour la Champions League, l Europa League et la Conference League.", ["UEFA", "Stages", "Fixtures"], "pages/6_EUROPE.py", None)
        _render_page_tile(bottom[2], "MONITORING", "Sante de la plateforme: runs, qualite de donnees et indicateurs de fraicheur.", ["Runs", "DQ", "Observability"], "pages/4_MONITORING.py", "Monitoring.png")
    else:
        bottom = st.columns(2)
        _render_page_tile(bottom[0], "PLAYERS", "Effectif disponible en base avec les postes, nationalites et informations principales du club filtre.", ["Effectif", "Squad view", "Exploration"], "pages/2_PLAYERS.py", "Players.png")
        _render_page_tile(bottom[1], "EUROPE", "Suivi UEFA dedie pour la Champions League, l Europa League et la Conference League.", ["UEFA", "Stages", "Fixtures"], "pages/6_EUROPE.py", None)

    extra = st.columns(2)
    _render_page_tile(extra[0], "HISTORY", "Historique des classements par saison avec comparaison des champions.", ["Historique", "Saisons", "Top 4"], "pages/7_HISTORY.py", None)
    _render_page_tile(extra[1], "PREDICTION", "Module baseline de prediction de match (probabilites 1N2 + score).", ["Poisson", "1N2", "xG proxy"], "pages/9_PREDICTION.py", None)


def _render_pipeline_quick_action() -> None:
    if not is_admin_authenticated():
        return
    st.markdown("<div class='fdp-section-title'>Pipeline Control</div>", unsafe_allow_html=True)
    status = get_pipeline_process_status()
    cols = st.columns([1, 1, 2])
    cols[0].metric("Process", "Running" if status["running"] else "Idle")
    cols[1].metric("PID", "-" if status["pid"] is None else str(status["pid"]))
    with cols[2]:
        clicked = st.button(
            "Relancer pipeline",
            key="app_launch_pipeline",
            disabled=bool(status["running"]),
            type="primary",
            use_container_width=True,
        )
    if clicked:
        start_result = start_pipeline_process()
        if start_result.get("started"):
            st.success(f"Pipeline lancee (pid={start_result['pid']}).")
        else:
            st.warning("Une pipeline est deja en cours.")
        st.rerun()


def main() -> None:
    render_admin_access_sidebar("app")
    admin_mode = is_admin_authenticated()
    competitions = get_competitions()
    runs = get_pipeline_runs(limit=5)
    _render_hero(competitions)
    _render_summary_cards(competitions, runs, is_admin=admin_mode)
    _render_pipeline_quick_action()
    _render_page_cards(is_admin=admin_mode)


if __name__ == "__main__":
    main()


