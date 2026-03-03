import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from data.dashboard_data import get_dq_checks, get_pipeline_runs
from ui.display import render_badge, render_note_card, render_page_banner, render_section_heading, render_status_badge, style_monitoring_table
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="MONITORING - Football Data Platform", layout="wide")


def _format_json(value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "{}"
    return str(value)


def _format_timestamp(value: Any) -> str:
    if value is None or value == "":
        return "-"
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return str(value)
    return timestamp.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M")


def _format_duration_ms(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    duration_ms = int(value)
    if duration_ms < 1000:
        return f"{duration_ms} ms"
    seconds = duration_ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes} min {remaining:02d}s"


def _freshness_label(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return "Unknown"
    now = datetime.now(timezone.utc)
    delta = now - timestamp.to_pydatetime()
    hours = int(delta.total_seconds() // 3600)
    if hours < 1:
        return "Updated less than 1h ago"
    if hours < 24:
        return f"Updated {hours}h ago"
    days = hours // 24
    return f"Updated {days}d ago"


def _health_summary(runs: pd.DataFrame, dq_checks: pd.DataFrame) -> dict[str, Any]:
    latest_run = runs.iloc[0]
    latest_status = str(latest_run["status"])
    fail_count = int((dq_checks["status"] == "FAIL").sum()) if not dq_checks.empty else 0
    warn_count = int((dq_checks["status"] == "WARN").sum()) if not dq_checks.empty else 0
    if latest_status == "FAILED" or fail_count > 0:
        health = "Attention required"
        severity = "FAIL"
    elif warn_count > 0:
        health = "Watch list"
        severity = "WARN"
    else:
        health = "Healthy"
        severity = "PASS"
    return {
        "health": health,
        "severity": severity,
        "latest_run": latest_run,
        "fail_count": fail_count,
        "warn_count": warn_count,
    }


def _render_health_hero(summary: dict[str, Any]) -> None:
    latest_run = summary["latest_run"]
    st.markdown("<div class='fdp-section-title'>Platform Health</div>", unsafe_allow_html=True)
    hero_left, hero_right = st.columns([4, 1], vertical_alignment="center")
    with hero_left:
        st.markdown(
            f"""
            <div class="fdp-hero">
              <div class="fdp-hero-title">Current platform status: {summary["health"]}</div>
              <div class="fdp-hero-sub">
                Last pipeline run at {_format_timestamp(latest_run["started_at"])}.
                {_freshness_label(latest_run["started_at"])}.
                Use this page to confirm the data is up to date before reading the dashboard.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(render_badge(summary["severity"]), unsafe_allow_html=True)


def _render_summary_metrics(summary: dict[str, Any]) -> None:
    latest_run = summary["latest_run"]
    volumes = latest_run.get("volumes", {}) or {}
    cards = [
        ("Platform status", summary["health"], str(latest_run["status"])),
        ("Last update", _freshness_label(latest_run["started_at"]), _format_timestamp(latest_run["started_at"])),
        ("Pipeline duration", _format_duration_ms(latest_run["duration_ms"]), "Latest successful execution"),
        ("Matches loaded", str(int(volumes.get("rows_fact_match", 0))), "Fact table rows currently persisted"),
        ("Standings rows", str(int(volumes.get("rows_fact_standings_snapshot", 0))), "Snapshots available for dashboard standings"),
        ("DQ fails", str(summary["fail_count"]), "Blocking quality issues on latest run"),
        ("DQ warnings", str(summary["warn_count"]), "Non-blocking quality alerts"),
        ("Rows loaded", str(int(latest_run["loaded_count"]) if pd.notna(latest_run["loaded_count"]) else 0), "All rows written during latest run"),
    ]
    html = "".join(
        f"""
        <div class="fdp-signal-card">
          <div class="fdp-signal-label">{label}</div>
          <div class="fdp-signal-value">{value}</div>
          <div class="fdp-signal-sub">{subtext}</div>
        </div>
        """
        for label, value, subtext in cards
    )
    st.markdown(f"<div class='fdp-signal-grid'>{html}</div>", unsafe_allow_html=True)


def _render_simple_explanation(summary: dict[str, Any]) -> None:
    latest_run = summary["latest_run"]
    status = str(latest_run["status"])
    if status == "FAILED":
        st.error("The last pipeline run failed. The dashboard may show stale or incomplete data.")
        return
    if summary["fail_count"] > 0:
        st.error("The pipeline completed, but some quality checks failed. Review the alert section below.")
        return
    if summary["warn_count"] > 0:
        st.warning("The pipeline completed with warnings. The dashboard is usable, but some data should be reviewed.")
        return
    st.success("The latest run completed successfully and no quality check is currently failing.")


def _render_alerts(dq_checks: pd.DataFrame) -> None:
    render_section_heading("Alerts")
    if dq_checks.empty:
        st.info("No data quality checks are available for the latest run.")
        return

    alerts = dq_checks[dq_checks["status"].isin(["FAIL", "WARN"])].copy()
    if alerts.empty:
        st.success("No active warning or failure detected on the latest run.")
        return

    alert_view = alerts[["status", "check_name", "details", "created_at"]].copy()
    alert_view["created_at"] = alert_view["created_at"].map(_format_timestamp)
    st.dataframe(style_monitoring_table(alert_view), hide_index=True, use_container_width=True)


def _render_recent_runs_gallery(runs: pd.DataFrame) -> None:
    render_section_heading("Recent Runs")
    recent_runs = runs.head(6).copy()
    cards: list[str] = []
    for _, run in recent_runs.iterrows():
        cards.append(
            f"""
            <div class="fdp-run-card">
              <div class="fdp-run-top">
                <div class="fdp-run-id">{str(run["run_id"])[:8]}</div>
                {render_badge(str(run["status"]))}
              </div>
              <div class="fdp-run-meta">
                <div><strong>Started:</strong> {_format_timestamp(run["started_at"])}</div>
                <div><strong>Duration:</strong> {_format_duration_ms(run["duration_ms"])}</div>
                <div><strong>Loaded:</strong> {int(run["loaded_count"]) if pd.notna(run["loaded_count"]) else 0} rows</div>
                <div><strong>Extracted:</strong> {int(run["extracted_count"]) if pd.notna(run["extracted_count"]) else 0} rows</div>
              </div>
            </div>
            """
        )
    st.markdown(f"<div class='fdp-run-list'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _render_run_details(runs: pd.DataFrame) -> str:
    render_section_heading("Technical Detail", "Detailed execution logs for admins, devs and data maintainers.")

    runs_display = runs.copy()
    runs_display["started_at"] = runs_display["started_at"].map(_format_timestamp)
    runs_display["ended_at"] = runs_display["ended_at"].map(_format_timestamp)
    runs_display["duration"] = runs_display["duration_ms"].map(_format_duration_ms)
    runs_display["volumes_jsonb"] = runs_display["volumes"].map(_format_json)
    runs_display["metrics_jsonb"] = runs_display["metrics"].map(_format_json)

    st.dataframe(
        runs_display[
            ["run_id", "started_at", "ended_at", "status", "duration", "extracted_count", "loaded_count", "volumes_jsonb"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    return st.selectbox("Run to inspect", runs["run_id"].astype(str).tolist())


def _render_selected_run(run_id: str, runs: pd.DataFrame) -> None:
    selected = runs[runs["run_id"].astype(str) == run_id].iloc[0]
    cols = st.columns([1, 1, 1, 1])
    cols[0].metric("Run", run_id[:8])
    cols[1].metric("Started", _format_timestamp(selected["started_at"]))
    cols[2].metric("Duration", _format_duration_ms(selected["duration_ms"]))
    with cols[3]:
        render_status_badge(str(selected["status"]))

    dq_checks = get_dq_checks(run_id=run_id, limit=100)
    render_section_heading("Data Quality Checks", "Checks persisted for the selected pipeline run.")
    if dq_checks.empty:
        st.info("No quality check stored for this run.")
        return

    dq_display = dq_checks.copy()
    dq_display["created_at"] = dq_display["created_at"].map(_format_timestamp)
    dq_display["metric_value"] = dq_display["metric_value"].round(2)
    dq_display["threshold"] = dq_display["threshold"].round(2)
    st.dataframe(
        style_monitoring_table(
            dq_display[["created_at", "check_name", "status", "severity", "metric_value", "threshold", "details"]]
        ),
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "MONITORING",
        "Health page for the data platform: start with the summary, then open the technical details if needed.",
        "Monitoring.png",
    )
    render_note_card("Monitoring sert a confirmer que les donnees affichees dans le dashboard sont fraiches, chargees et controlees.")

    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    runs = get_pipeline_runs(limit=25)
    if runs.empty:
        st.info("No pipeline run is available yet.")
        return

    latest_run_id = str(runs.iloc[0]["run_id"])
    latest_checks = get_dq_checks(run_id=latest_run_id, limit=100)
    summary = _health_summary(runs, latest_checks)
    summary_tab, detail_tab = st.tabs(["Summary", "Technical Detail"])

    with summary_tab:
        _render_health_hero(summary)
        _render_summary_metrics(summary)
        _render_simple_explanation(summary)
        _render_alerts(latest_checks)
        _render_recent_runs_gallery(runs)

    with detail_tab:
        selected_run_id = _render_run_details(runs)
        _render_selected_run(selected_run_id, runs)


if __name__ == "__main__":
    main()
