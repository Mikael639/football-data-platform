import pandas as pd
import streamlit as st

from data.dashboard_data import (
    describe_season_source,
    get_current_standings,
    get_dq_checks,
    get_kpis,
    get_latest_quality_score,
    get_matches,
    get_pipeline_runs,
    get_recent_matches,
    get_standings_curve,
)
from state.filters import render_global_filters
from ui.charts import render_position_curve
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_section_heading
from ui.exports import render_csv_download
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


def _previous_period_range(date_start: str | None, date_end: str | None) -> tuple[str | None, str | None]:
    if not date_start or not date_end:
        return None, None
    start_dt = pd.to_datetime(date_start, errors="coerce")
    end_dt = pd.to_datetime(date_end, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None, None
    if end_dt < start_dt:
        return None, None
    days = int((end_dt - start_dt).days) + 1
    prev_end = start_dt - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=max(days - 1, 0))
    return prev_start.date().isoformat(), prev_end.date().isoformat()


def _data_reliability_snapshot(
    competition_id: int | None,
    season: str | None,
    team_id: int | None,
    date_start: str | None,
    date_end: str | None,
) -> dict[str, object]:
    runs = get_pipeline_runs(limit=1)
    run_status = "N/A"
    freshness_label = "Unknown"
    fail_count = 0
    warn_count = 0

    if not runs.empty:
        latest = runs.iloc[0]
        run_status = str(latest.get("status") or "UNKNOWN")
        started_at = pd.to_datetime(latest.get("started_at"), errors="coerce", utc=True)
        if pd.notna(started_at):
            now = pd.Timestamp.now(tz="UTC")
            hours = int((now - started_at).total_seconds() // 3600)
            if hours < 1:
                freshness_label = "< 1h"
            elif hours < 24:
                freshness_label = f"{hours}h"
            else:
                freshness_label = f"{hours // 24}d"

        run_id = str(latest.get("run_id") or "")
        if run_id:
            checks = get_dq_checks(run_id=run_id, limit=500)
            if not checks.empty:
                fail_count = int((checks["status"].astype(str) == "FAIL").sum())
                warn_count = int((checks["status"].astype(str) == "WARN").sum())

    matches = get_matches(
        competition_id=competition_id,
        season=season,
        team_id=team_id,
        date_start=date_start,
        date_end=date_end,
    )
    if matches.empty:
        return {
            "run_status": run_status,
            "freshness_label": freshness_label,
            "dq_alerts_label": f"{fail_count}F / {warn_count}W",
            "anomaly_count": 0,
            "completion_rate": None,
        }

    status = matches["status"].fillna("").astype(str).str.upper()
    finished = matches[status == "FINISHED"].copy()
    missing_scores = 0
    completion_rate: float | None = None
    if not finished.empty:
        missing_scores = int((finished["home_score"].isna() | finished["away_score"].isna()).sum())
        completion_rate = round((1 - (missing_scores / len(finished.index))) * 100, 1)
    duplicated_match_ids = int(matches["match_id"].duplicated().sum()) if "match_id" in matches.columns else 0
    anomaly_count = int(missing_scores + duplicated_match_ids)

    return {
        "run_status": run_status,
        "freshness_label": freshness_label,
        "dq_alerts_label": f"{fail_count}F / {warn_count}W",
        "anomaly_count": anomaly_count,
        "completion_rate": completion_rate,
    }


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
    prev_start, prev_end = _previous_period_range(filters.date_start, filters.date_end)
    prev_kpis = (
        get_kpis(
            competition_id=filters.competition_id,
            season=filters.season,
            team_id=filters.team_id,
            date_range=(prev_start, prev_end),
        )
        if prev_start and prev_end
        else None
    )
    if prev_start and prev_end:
        st.caption(f"Comparatif KPI vs periode precedente: {prev_start} -> {prev_end}")

    quality = get_latest_quality_score()
    cols = st.columns(6)
    cols[0].metric(
        "Matches",
        kpis["matches"],
        None if prev_kpis is None else int(kpis["matches"]) - int(prev_kpis["matches"]),
    )
    cols[1].metric(
        "Goals For",
        kpis["goals_for"],
        None if prev_kpis is None else int(kpis["goals_for"]) - int(prev_kpis["goals_for"]),
    )
    cols[2].metric(
        "Goals Against",
        kpis["goals_against"],
        None if prev_kpis is None else int(kpis["goals_against"]) - int(prev_kpis["goals_against"]),
    )
    cols[3].metric(
        "Goal Diff",
        kpis["goal_diff"],
        None if prev_kpis is None else int(kpis["goal_diff"]) - int(prev_kpis["goal_diff"]),
    )
    win_rate_delta = None
    if prev_kpis is not None and kpis["win_rate"] is not None and prev_kpis["win_rate"] is not None:
        win_rate_delta = f"{float(kpis['win_rate']) - float(prev_kpis['win_rate']):+.1f} pts"
    cols[4].metric("Win Rate", "-" if kpis["win_rate"] is None else f"{kpis['win_rate']}%", win_rate_delta)
    quality_label = (
        f"{quality['score']}/100 ({quality['grade']})"
        if quality.get("score") is not None
        else "N/A"
    )
    cols[5].metric("DQ Score", quality_label)

    reliability = _data_reliability_snapshot(
        competition_id=filters.competition_id,
        season=filters.season,
        team_id=filters.team_id,
        date_start=filters.date_start,
        date_end=filters.date_end,
    )
    render_section_heading("Fiabilite des donnees", "Fraicheur du pipeline et anomalies detectees sur le scope filtre.")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Pipeline", str(reliability["run_status"]))
    r2.metric("Fraicheur", str(reliability["freshness_label"]))
    r3.metric("DQ Alerts", str(reliability["dq_alerts_label"]))
    r4.metric("Anomalies", int(reliability["anomaly_count"]))
    completion_rate = reliability.get("completion_rate")
    r5.metric("Completeness", "-" if completion_rate is None else f"{completion_rate}%")

    standings = get_current_standings(filters.competition_id, filters.season)
    render_section_heading(
        "Dernier classement disponible",
        "Le classement depend surtout de la competition et de la saison choisies. La plage de dates agit surtout sur les KPI et le calendrier.",
    )
    if standings.empty:
        st.info(
            "Aucun snapshot de classement fiable disponible pour ce filtre "
            "(absent ou incoherent). Relance la synchronisation des donnees."
        )
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
        render_csv_download(
            df=standings_display[["Pos", "Trend", "Equipe", "Pts", "MJ", "G", "N", "P", "BP", "BC", "Diff"]],
            label="Export classement (CSV)",
            filename="overview_classement.csv",
            key="overview_export_standings",
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
            recent_table = _format_match_table(recent_matches)
            render_adaptive_table(
                recent_table,
                title="Derniers 10 matchs",
                badge_columns={"status": "status"},
                strong_columns={"home_team", "away_team"},
                max_height=760,
            )
            render_csv_download(
                df=recent_table,
                label="Export derniers matchs (CSV)",
                filename="overview_recent_matches.csv",
                key="overview_export_recent",
            )
            _render_match_detail_entry(recent_matches, "overview_recent")
    with right:
        if upcoming_matches.empty:
            st.info("Aucun match a venir sur cette plage.")
        else:
            upcoming_table = _format_match_table(upcoming_matches)
            render_adaptive_table(
                upcoming_table,
                title="Prochains 5 matchs",
                badge_columns={"status": "status"},
                strong_columns={"home_team", "away_team"},
                max_height=640,
            )
            render_csv_download(
                df=upcoming_table,
                label="Export matchs a venir (CSV)",
                filename="overview_upcoming_matches.csv",
                key="overview_export_upcoming",
            )
            _render_match_detail_entry(upcoming_matches, "overview_upcoming")


if __name__ == "__main__":
    main()

