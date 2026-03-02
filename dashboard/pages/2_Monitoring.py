import json

import pandas as pd
import streamlit as st

from data.dashboard_data import get_dq_checks, get_pipeline_runs
from ui.display import render_status_badge, style_monitoring_table
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="Monitoring - Football Data Platform", layout="wide")


def _format_json(value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "{}"
    return str(value)


def main() -> None:
    inject_dashboard_styles()
    st.title("Monitoring")

    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    runs = get_pipeline_runs(limit=25)
    st.subheader("Pipeline runs")
    if runs.empty:
        st.info("Aucun run pipeline disponible.")
        return

    runs_display = runs.copy()
    runs_display["metrics_jsonb"] = runs_display["metrics"].map(_format_json)
    runs_display["volumes_jsonb"] = runs_display["volumes"].map(_format_json)
    st.dataframe(
        runs_display[["run_id", "started_at", "status", "duration_ms", "extracted_count", "loaded_count", "volumes_jsonb"]],
        hide_index=True,
        use_container_width=True,
    )

    selected_run_id = st.selectbox("run_id", runs["run_id"].astype(str).tolist())
    selected_status = runs[runs["run_id"].astype(str) == selected_run_id]["status"].iloc[0]
    render_status_badge(str(selected_status))

    dq_checks = get_dq_checks(run_id=selected_run_id, limit=100)
    st.subheader("Data quality checks")
    if dq_checks.empty:
        st.info("Aucun check de qualite pour ce run.")
        return

    dq_display = dq_checks.copy()
    dq_display["metric_value"] = dq_display["metric_value"].round(2)
    dq_display["threshold"] = dq_display["threshold"].round(2)
    st.dataframe(
        style_monitoring_table(
            dq_display[["created_at", "check_name", "status", "severity", "metric_value", "threshold", "details"]]
        ),
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
