import pandas as pd
import streamlit as st

from data.dashboard_data import get_competitions, get_pipeline_runs
from ui.display import laliga_logo_source, render_status_badge
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="Football Data Platform", layout="wide")


def main() -> None:
    inject_dashboard_styles()
    st.title("Football Data Platform")
    st.caption("Tableau de bord analytics et monitoring, alimente prioritairement depuis la base locale.")

    hero_left, hero_right = st.columns([5, 1], vertical_alignment="center")
    with hero_left:
        competitions = get_competitions()
        competition_names = ", ".join(competitions["competition_name"].astype(str).tolist()) if not competitions.empty else "-"
        st.write(f"Competitions detectees: {competition_names}")
        st.write("Pages disponibles:")
        st.page_link("pages/0_Overview.py", label="Overview")
        st.page_link("pages/1_Team.py", label="Team")
        st.page_link("pages/2_Monitoring.py", label="Monitoring")
        st.page_link("pages/3_Joueurs.py", label="Joueurs")
    with hero_right:
        st.image(laliga_logo_source(), use_column_width=True)

    st.subheader("Dernier run pipeline")
    runs = get_pipeline_runs(limit=1)
    if runs.empty:
        st.info("Aucun run pipeline en base. Lance le pipeline pour initialiser le dashboard.")
        return

    run = runs.iloc[0]
    info_cols = st.columns(4)
    info_cols[0].metric("Run", str(run["run_id"])[:8])
    info_cols[1].metric("Started", str(run["started_at"]))
    info_cols[2].metric("Duration (ms)", int(run["duration_ms"]) if pd.notna(run["duration_ms"]) else 0)
    with info_cols[3]:
        render_status_badge(str(run["status"]))


if __name__ == "__main__":
    main()
