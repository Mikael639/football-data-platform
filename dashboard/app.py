import os
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from components.main_tabs import (
    render_clubs_tab,
    render_ligue_tab,
    render_player_details_tab,
    render_team_tab,
)
from data.dashboard_data import (
    build_local_league_table,
    current_season_bounds,
    current_season_label,
    current_season_start_year_dash,
    fetch_laliga_teams_live,
    get_engine,
)
from ui.display import laliga_logo_source, render_team_hero
from ui.labels import TAB_LABELS_MAIN
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="Football Data Platform", layout="wide")


def render_header(season_start_year: int) -> None:
    header_left, header_right = st.columns([6.0, 1.3], vertical_alignment="center")
    with header_left:
        st.title("Plateforme Data Football - LaLiga")
        st.caption(f"Saison en cours utilisee partout dans le dashboard: {current_season_label(season_start_year)}")
    with header_right:
        st.image(laliga_logo_source(), use_column_width=True)


def load_teams(engine: Any, competition_code: str, season_start_year: int, season_start: str, season_end: str) -> pd.DataFrame:
    teams_df, _teams_live_err = fetch_laliga_teams_live(competition_code, season_start_year)
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
    return teams_df


def resolve_selected_team(teams_df: pd.DataFrame) -> tuple[str, int | None]:
    team_names = ["Tous les clubs"] + teams_df["team_name"].dropna().tolist()

    if "selected_team_name" not in st.session_state:
        st.session_state["selected_team_name"] = "Tous les clubs"
    if st.session_state["selected_team_name"] not in team_names:
        st.session_state["selected_team_name"] = "Tous les clubs"

    st.selectbox("Filtrer par club", team_names, key="selected_team_name")
    selected_team_name = st.session_state["selected_team_name"]
    if selected_team_name == "Tous les clubs":
        return selected_team_name, None

    selected_team_id = int(teams_df.loc[teams_df["team_name"] == selected_team_name, "team_id"].iloc[0])
    return selected_team_name, selected_team_id


def load_matches_all_season(engine: Any, season_start: str, season_end: str) -> pd.DataFrame:
    query = """
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
    matches_df = pd.read_sql(text(query), engine, params={"season_start": season_start, "season_end": season_end})
    if not matches_df.empty:
        matches_df["date_dt"] = pd.to_datetime(matches_df["date_id"], errors="coerce")
    return matches_df


def load_matches_for_scope(
    engine: Any,
    season_start: str,
    season_end: str,
    selected_team_id: int | None,
) -> pd.DataFrame:
    conditions = ["m.date_id BETWEEN :season_start AND :season_end"]
    params: dict[str, Any] = {"season_start": season_start, "season_end": season_end}
    if selected_team_id is not None:
        conditions.append("(m.home_team_id = :tid OR m.away_team_id = :tid)")
        params["tid"] = selected_team_id

    query = f"""
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
    WHERE {' AND '.join(conditions)}
    ORDER BY m.date_id DESC NULLS LAST, m.match_id DESC;
    """
    matches_df = pd.read_sql(text(query), engine, params=params)
    if not matches_df.empty:
        matches_df["date_dt"] = pd.to_datetime(matches_df["date_id"], errors="coerce")
    return matches_df


def load_players(engine: Any) -> pd.DataFrame:
    query = """
    SELECT
      p.player_id, p.full_name, p.position, p.nationality, p.birth_date,
      p.team_id, t.team_name
    FROM dim_player p
    LEFT JOIN dim_team t ON t.team_id = p.team_id
    """
    players_df = pd.read_sql(text(query), engine)
    if players_df.empty:
        return players_df

    return (
        players_df.sort_values(["team_name", "full_name"], ascending=[True, True])
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )


def build_club_summary(
    teams_df: pd.DataFrame,
    players_df: pd.DataFrame,
    matches_all_df: pd.DataFrame,
) -> pd.DataFrame:
    club_summary = teams_df.copy().merge(
        players_df.groupby("team_id", dropna=True)["player_id"].nunique().rename("players_count"),
        how="left",
        on="team_id",
    )
    club_summary["players_count"] = club_summary["players_count"].fillna(0).astype(int)

    if not matches_all_df.empty:
        home_count = matches_all_df.groupby("home_team_id")["match_id"].nunique()
        away_count = matches_all_df.groupby("away_team_id")["match_id"].nunique()
        club_summary["matches_in_scope"] = (
            club_summary["team_id"].map(home_count).fillna(0) + club_summary["team_id"].map(away_count).fillna(0)
        ).astype(int)
    else:
        club_summary["matches_in_scope"] = 0

    club_summary["data_quality_status"] = "OK"
    club_summary.loc[club_summary["matches_in_scope"] <= 0, "data_quality_status"] = "A_VERIFIER"
    club_summary.loc[
        (club_summary["matches_in_scope"] > 0) & (club_summary["players_count"] <= 0),
        "data_quality_status",
    ] = "INCOMPLET"
    return club_summary.sort_values("team_name").reset_index(drop=True)


def load_dashboard_state() -> dict[str, Any]:
    engine = get_engine()
    season_start_year = current_season_start_year_dash()
    season_start, season_end = current_season_bounds(season_start_year)
    competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")

    teams_df = load_teams(engine, competition_code, season_start_year, season_start, season_end)
    selected_team_name, selected_team_id = resolve_selected_team(teams_df)
    matches_all_df = load_matches_all_season(engine, season_start, season_end)
    matches_df = load_matches_for_scope(engine, season_start, season_end, selected_team_id)
    players_all_df = load_players(engine)
    players_df = (
        players_all_df[players_all_df["team_id"] == selected_team_id].copy()
        if selected_team_id is not None
        else players_all_df.copy()
    )

    return {
        "engine": engine,
        "season_start_year": season_start_year,
        "teams_df": teams_df,
        "selected_team_name": selected_team_name,
        "selected_team_id": selected_team_id,
        "df_matches_all_season": matches_all_df,
        "df_matches": matches_df,
        "df_players": players_df,
        "league_local_all_season": build_local_league_table(matches_all_df),
        "club_summary": build_club_summary(teams_df, players_all_df, matches_all_df),
    }


def render_tabs(state: dict[str, Any]) -> None:
    tab_team, tab_standings, tab_clubs, tab_player_details = st.tabs(TAB_LABELS_MAIN)

    with tab_team:
        render_team_tab(
            selected_team_name=state["selected_team_name"],
            season_start_year=state["season_start_year"],
            df_matches=state["df_matches"],
            selected_team_id=state["selected_team_id"],
            teams_df=state["teams_df"],
            render_team_hero=render_team_hero,
        )

    with tab_standings:
        render_ligue_tab(
            season_start_year=state["season_start_year"],
            league_local_all_season=state["league_local_all_season"],
            selected_team_name=state["selected_team_name"],
        )

    with tab_clubs:
        render_clubs_tab(
            league_local_all_season=state["league_local_all_season"],
            club_summary=state["club_summary"],
            engine=state["engine"],
        )

    with tab_player_details:
        render_player_details_tab()


def main() -> None:
    inject_dashboard_styles()
    state = load_dashboard_state()
    render_header(state["season_start_year"])
    render_tabs(state)


if __name__ == "__main__":
    main()
