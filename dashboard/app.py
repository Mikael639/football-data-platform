import os

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import text

from components.main_tabs import render_clubs_tab, render_ligue_tab, render_team_tab
from components.study_players import render_player_study_tab as render_study_players_tab
from data.dashboard_data import (
    build_local_league_table,
    build_team_match_view,
    compute_team_kpis,
    current_season_bounds,
    current_season_label,
    current_season_start_year_dash,
    fetch_laliga_teams_live,
    fetch_live_team_squad,
    get_engine,
    upsert_players_to_db,
)
from ui.charts import render_ppm_chart, render_result_distribution_chart, render_sorted_bar_chart
from ui.labels import TAB_LABELS_MAIN
from ui.display import (
    add_leader_star,
    add_podium_icons,
    laliga_logo_source,
    render_form_timeline,
    render_quality_badges,
    render_team_hero,
    style_ligue_table,
)
from ui.styles import VISUAL_COLORS, inject_dashboard_styles

st.set_page_config(page_title="Football Data Platform", layout="wide")

engine = get_engine()
season_start_year = current_season_start_year_dash()
season_start, season_end = current_season_bounds(season_start_year)
competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")

inject_dashboard_styles()

header_left, header_right = st.columns([6.0, 1.3], vertical_alignment="center")
with header_left:
    st.title("Plateforme Data Football - LaLiga")
    st.caption(f"Saison en cours utilisee partout dans le dashboard: {current_season_label(season_start_year)}")
with header_right:
    st.image(laliga_logo_source(), use_column_width=True)

# -----------------------
# Shared selectors / data
# -----------------------
teams_df, teams_live_err = fetch_laliga_teams_live(competition_code, season_start_year)
if teams_df is None or teams_df.empty:
    teams_df = pd.read_sql(
        text(
            """
            SELECT DISTINCT t.team_id, t.team_name
            FROM dim_team t
            JOIN (
              SELECT home_team_id AS team_id
              FROM fact_match
              WHERE date_id BETWEEN :season_start AND :season_end
              UNION
              SELECT away_team_id AS team_id
              FROM fact_match
              WHERE date_id BETWEEN :season_start AND :season_end
            ) s ON s.team_id = t.team_id
            ORDER BY t.team_name;
            """
        ),
        engine,
        params={"season_start": season_start, "season_end": season_end},
    )
if teams_df.empty:
    teams_df = pd.read_sql("SELECT team_id, team_name FROM dim_team ORDER BY team_name;", engine)
team_names = ["Tous les clubs"] + teams_df["team_name"].dropna().tolist()

if "selected_team_name" not in st.session_state:
    st.session_state["selected_team_name"] = "Tous les clubs"
if st.session_state["selected_team_name"] not in team_names:
    st.session_state["selected_team_name"] = "Tous les clubs"

st.selectbox("Filtrer par club", team_names, key="selected_team_name")
selected_team_name = st.session_state["selected_team_name"]
selected_team_id = None
if selected_team_name != "Tous les clubs":
    selected_team_id = int(
        teams_df.loc[teams_df["team_name"] == selected_team_name, "team_id"].iloc[0]
    )

matches_all_query = """
SELECT
  m.match_id,
  m.date_id,
  m.home_team_id,
  ht.team_name AS home_team,
  m.away_team_id,
  at.team_name AS away_team,
  m.home_score,
  m.away_score
FROM fact_match m
JOIN dim_team ht ON ht.team_id = m.home_team_id
JOIN dim_team at ON at.team_id = m.away_team_id
WHERE m.date_id BETWEEN :season_start AND :season_end
ORDER BY m.date_id DESC NULLS LAST, m.match_id DESC;
"""
df_matches_all_season = pd.read_sql(
    text(matches_all_query),
    engine,
    params={"season_start": season_start, "season_end": season_end},
)
if not df_matches_all_season.empty:
    df_matches_all_season["date_dt"] = pd.to_datetime(df_matches_all_season["date_id"], errors="coerce")
league_local_all_season = build_local_league_table(df_matches_all_season)

match_conditions = ["m.date_id BETWEEN :season_start AND :season_end"]
match_params = {"season_start": season_start, "season_end": season_end}
if selected_team_id is not None:
    match_conditions.append("(m.home_team_id = :tid OR m.away_team_id = :tid)")
    match_params["tid"] = selected_team_id

matches_query = f"""
SELECT
  m.match_id,
  m.date_id,
  m.home_team_id,
  ht.team_name AS home_team,
  m.away_team_id,
  at.team_name AS away_team,
  m.home_score,
  m.away_score
FROM fact_match m
JOIN dim_team ht ON ht.team_id = m.home_team_id
JOIN dim_team at ON at.team_id = m.away_team_id
WHERE {' AND '.join(match_conditions)}
ORDER BY m.date_id DESC NULLS LAST, m.match_id DESC;
"""
df_matches = pd.read_sql(text(matches_query), engine, params=match_params)
if not df_matches.empty:
    df_matches["date_dt"] = pd.to_datetime(df_matches["date_id"], errors="coerce")

players_query = """
SELECT
  p.player_id, p.full_name, p.position, p.nationality, p.birth_date,
  p.team_id, t.team_name
FROM dim_player p
LEFT JOIN dim_team t ON t.team_id = p.team_id
"""
df_players_all = pd.read_sql(text(players_query), engine)

if not df_players_all.empty:
    df_players_all = (
        df_players_all.sort_values(["team_name", "full_name"], ascending=[True, True])
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

if selected_team_id is not None:
    df_players = df_players_all[df_players_all["team_id"] == selected_team_id].copy()
else:
    df_players = df_players_all.copy()

club_summary = (
    teams_df.copy()
    .merge(
        df_players_all.groupby("team_id", dropna=True)["player_id"].nunique().rename("players_count"),
        how="left",
        on="team_id",
    )
)
club_summary["players_count"] = club_summary["players_count"].fillna(0).astype(int)

if not df_matches_all_season.empty:
    home_cnt = df_matches_all_season.groupby("home_team_id")["match_id"].nunique()
    away_cnt = df_matches_all_season.groupby("away_team_id")["match_id"].nunique()
    club_summary["matches_in_scope"] = (
        club_summary["team_id"].map(home_cnt).fillna(0) + club_summary["team_id"].map(away_cnt).fillna(0)
    ).astype(int)
else:
    club_summary["matches_in_scope"] = 0

club_summary["data_quality_status"] = "OK"
club_summary.loc[
    (club_summary["matches_in_scope"] <= 0),
    "data_quality_status",
] = "A_VERIFIER"
club_summary.loc[
    (club_summary["matches_in_scope"] > 0) & (club_summary["players_count"] <= 0),
    "data_quality_status",
] = "INCOMPLET"
club_summary = club_summary.sort_values("team_name").reset_index(drop=True)

tab_team, tab_study, tab_standings, tab_clubs = st.tabs(TAB_LABELS_MAIN)

with tab_team:
    render_team_tab(
        selected_team_name=selected_team_name,
        season_start_year=season_start_year,
        df_matches=df_matches,
        selected_team_id=selected_team_id,
        teams_df=teams_df,
        render_team_hero=render_team_hero,
    )

with tab_study:
    render_study_players_tab()

with tab_standings:
    render_ligue_tab(
        season_start_year=season_start_year,
        league_local_all_season=league_local_all_season,
        selected_team_name=selected_team_name,
    )

with tab_clubs:
    render_clubs_tab(
        league_local_all_season=league_local_all_season,
        club_summary=club_summary,
        engine=engine,
    )
