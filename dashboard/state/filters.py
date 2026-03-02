from __future__ import annotations

from datetime import date as dt_date

import streamlit as st

from data.dashboard_data import (
    DashboardFilters,
    current_season_label,
    get_competitions,
    get_date_bounds,
    get_default_date_range,
    get_seasons,
    get_teams,
)


def _competition_options() -> tuple[list[str], dict[str, int | None]]:
    competitions = get_competitions()
    labels: list[str] = []
    mapping: dict[str, int | None] = {}
    for _, row in competitions.iterrows():
        label = str(row["competition_name"])
        mapping[label] = None if row["competition_id"] is None else int(row["competition_id"])
        labels.append(label)
    return labels, mapping


def _season_options(competition_id: int | None) -> tuple[list[str], dict[str, int | None]]:
    seasons = get_seasons(competition_id)
    labels: list[str] = []
    mapping: dict[str, int | None] = {}
    for _, row in seasons.iterrows():
        season_start = int(row["season_start"])
        label = current_season_label(season_start)
        labels.append(label)
        mapping[label] = season_start
    return labels, mapping


def _team_options(competition_id: int | None, season_start: int | None) -> tuple[list[str], dict[str, int | None]]:
    teams = get_teams(competition_id, season_start)
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
    default_competition = st.session_state.get(competition_key, competition_labels[0] if competition_labels else None)
    if default_competition not in competition_labels and competition_labels:
        default_competition = competition_labels[0]
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
    season_start = season_map.get(selected_season)

    team_labels, team_map = _team_options(competition_id, season_start)
    team_key = f"{page_key}_team"
    if st.session_state.get(team_key) not in team_labels:
        st.session_state[team_key] = team_labels[0]
    selected_team = st.sidebar.selectbox("Equipe", team_labels, key=team_key)
    team_id = team_map.get(selected_team)

    bounds = get_date_bounds(competition_id, season_start, team_id)
    default_start, default_end = get_default_date_range(bounds)
    default_date_value: tuple[dt_date, dt_date] | tuple[()] = ()
    if default_start and default_end:
        default_date_value = (dt_date.fromisoformat(default_start), dt_date.fromisoformat(default_end))
    date_key = f"{page_key}_date_range"
    selected_dates = st.sidebar.date_input(
        "Plage de dates",
        value=st.session_state.get(date_key, default_date_value),
        key=date_key,
    )

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
        season_start=season_start,
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
