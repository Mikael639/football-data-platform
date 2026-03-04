from __future__ import annotations

import streamlit as st

from data.dashboard_data import (
    DashboardFilters,
    get_competitions,
    get_seasons,
    get_teams,
)


def _competition_options() -> tuple[list[str], dict[str, int | None]]:
    competitions = get_competitions()
    labels: list[str] = ["Toutes les competitions"]
    mapping: dict[str, int | None] = {"Toutes les competitions": None}
    for _, row in competitions.iterrows():
        label = str(row["competition_name"])
        mapping[label] = None if row["competition_id"] is None else int(row["competition_id"])
        labels.append(label)
    return labels, mapping


def _season_options(competition_id: int | None) -> tuple[list[str], dict[str, str | None]]:
    seasons = get_seasons(competition_id)
    labels = ["Toutes les saisons"]
    mapping: dict[str, str | None] = {"Toutes les saisons": None}
    for _, row in seasons.iterrows():
        label = str(row["season"])
        labels.append(label)
        mapping[label] = label
    return labels, mapping


def _team_options(competition_id: int | None, season: str | None) -> tuple[list[str], dict[str, int | None]]:
    teams = get_teams(competition_id, season)
    labels = ["Tous les clubs"]
    mapping: dict[str, int | None] = {"Tous les clubs": None}
    for _, row in teams.iterrows():
        label = str(row["team_name"])
        mapping[label] = int(row["team_id"])
        labels.append(label)
    return labels, mapping


def render_global_filters(page_key: str) -> DashboardFilters:
    st.sidebar.header("Filtres globaux")

    competition_labels, competition_map = _competition_options()
    competition_key = f"{page_key}_competition"
    if st.session_state.get(competition_key) not in competition_labels and competition_labels:
        st.session_state[competition_key] = competition_labels[0]
    selected_competition = st.sidebar.selectbox(
        "Competition",
        competition_labels,
        key=competition_key,
        disabled=len(competition_labels) <= 1,
    )
    competition_id = competition_map.get(selected_competition)

    season_labels, season_map = _season_options(competition_id)
    season_key = f"{page_key}_season"
    if st.session_state.get(season_key) not in season_labels and season_labels:
        st.session_state[season_key] = season_labels[0]
    selected_season = st.sidebar.selectbox("Saison", season_labels, key=season_key)
    season = season_map.get(selected_season)

    team_labels, team_map = _team_options(competition_id, season)
    team_key = f"{page_key}_team"
    if st.session_state.get(team_key) not in team_labels:
        st.session_state[team_key] = team_labels[0]
    selected_team = st.sidebar.selectbox("Equipe", team_labels, key=team_key)
    team_id = team_map.get(selected_team)

    date_key = f"{page_key}_date_range"
    selected_dates = st.sidebar.date_input(
        "Plage de dates",
        value=st.session_state.get(date_key, ()),
    )
    st.session_state[date_key] = selected_dates

    date_start = None
    date_end = None
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        date_start = selected_dates[0].isoformat()
        date_end = selected_dates[1].isoformat()
    elif hasattr(selected_dates, "isoformat"):
        date_start = selected_dates.isoformat()
        date_end = selected_dates.isoformat()

    filters = DashboardFilters(
        competition_id=competition_id,
        season=season,
        team_id=team_id,
        date_start=date_start,
        date_end=date_end,
    )
    st.session_state["global_filters"] = filters
    return filters


def require_team_selection(filters: DashboardFilters) -> DashboardFilters | None:
    if filters.team_id is not None:
        return filters
    st.info("Selectionne un club dans la barre laterale pour afficher cette page.")
    return None
