from __future__ import annotations

import streamlit as st

from data.dashboard_data import (
    DashboardFilters,
    get_competitions,
    get_seasons,
    get_teams,
)

FAV_COMPETITION_ID_KEY = "fdp_favorite_competition_id"
FAV_TEAM_ID_KEY = "fdp_favorite_team_id"
USE_FAVORITES_KEY = "fdp_use_favorites"
FAVORITES_APPLY_NONCE_KEY = "fdp_favorites_apply_nonce"
FAVORITES_FORCE_ENABLE_KEY = "fdp_favorites_force_enable"


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


def _label_for_value(mapping: dict[str, int | None], target: int | None) -> str | None:
    for label, value in mapping.items():
        if value == target:
            return label
    return None


def render_global_filters(page_key: str, forced_season: str | None = None) -> DashboardFilters:
    if bool(st.session_state.get(FAVORITES_FORCE_ENABLE_KEY, False)):
        st.session_state[USE_FAVORITES_KEY] = True
        st.session_state[FAVORITES_FORCE_ENABLE_KEY] = False

    st.sidebar.header("Filtres globaux")
    use_favorites = st.sidebar.checkbox(
        "Utiliser mes favoris",
        value=bool(st.session_state.get(USE_FAVORITES_KEY, True)),
        key=USE_FAVORITES_KEY,
    )
    apply_nonce = int(st.session_state.get(FAVORITES_APPLY_NONCE_KEY, 0))
    apply_seen_key = f"{page_key}_favorites_apply_seen"
    apply_favorites_for_page = use_favorites and int(st.session_state.get(apply_seen_key, 0)) < apply_nonce

    competition_labels, competition_map = _competition_options()
    competition_key = f"{page_key}_competition"
    if (st.session_state.get(competition_key) not in competition_labels and competition_labels) or apply_favorites_for_page:
        default_label = competition_labels[0]
        if use_favorites:
            favorite_competition_id = st.session_state.get(FAV_COMPETITION_ID_KEY)
            favorite_competition_label = _label_for_value(competition_map, favorite_competition_id)
            if favorite_competition_label in competition_labels:
                default_label = str(favorite_competition_label)
        st.session_state[competition_key] = default_label
    selected_competition = st.sidebar.selectbox(
        "Competition",
        competition_labels,
        key=competition_key,
        disabled=len(competition_labels) <= 1,
    )
    competition_id = competition_map.get(selected_competition)

    season_key = f"{page_key}_season"
    if forced_season is not None:
        season_labels = [str(forced_season)]
        season_map: dict[str, str | None] = {str(forced_season): str(forced_season)}
        st.session_state[season_key] = str(forced_season)
        selected_season = st.sidebar.selectbox("Saison", season_labels, key=season_key, disabled=True)
    else:
        season_labels, season_map = _season_options(competition_id)
        if st.session_state.get(season_key) not in season_labels and season_labels:
            st.session_state[season_key] = season_labels[0]
        selected_season = st.sidebar.selectbox("Saison", season_labels, key=season_key)
    season = season_map.get(selected_season)

    team_labels, team_map = _team_options(competition_id, season)
    team_key = f"{page_key}_team"
    if st.session_state.get(team_key) not in team_labels or apply_favorites_for_page:
        default_team_label = team_labels[0]
        if use_favorites:
            favorite_team_id = st.session_state.get(FAV_TEAM_ID_KEY)
            favorite_team_label = _label_for_value(team_map, favorite_team_id)
            if favorite_team_label in team_labels:
                default_team_label = str(favorite_team_label)
        st.session_state[team_key] = default_team_label
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

    with st.sidebar.expander("Favoris", expanded=False):
        fav_competition_id = st.session_state.get(FAV_COMPETITION_ID_KEY)
        fav_team_id = st.session_state.get(FAV_TEAM_ID_KEY)
        fav_competition_label = _label_for_value(competition_map, fav_competition_id) or "Aucune"
        fav_team_label = _label_for_value(team_map, fav_team_id) or "Aucun"
        st.caption(f"Competition favorite: {fav_competition_label}")
        st.caption(f"Club favori: {fav_team_label}")
        save_key = f"{page_key}_save_favorites"
        clear_key = f"{page_key}_clear_favorites"
        if st.button("Enregistrer le filtre courant", key=save_key, use_container_width=True):
            if competition_id is None:
                st.session_state.pop(FAV_COMPETITION_ID_KEY, None)
            else:
                st.session_state[FAV_COMPETITION_ID_KEY] = int(competition_id)
            if team_id is None:
                st.session_state.pop(FAV_TEAM_ID_KEY, None)
            else:
                st.session_state[FAV_TEAM_ID_KEY] = int(team_id)
            st.success("Favoris enregistres.")
        if st.button("Effacer les favoris", key=clear_key, use_container_width=True):
            st.session_state.pop(FAV_COMPETITION_ID_KEY, None)
            st.session_state.pop(FAV_TEAM_ID_KEY, None)
            st.success("Favoris supprimes.")
        apply_all_key = f"{page_key}_apply_favorites_all"
        if st.button("Appliquer favoris partout maintenant", key=apply_all_key, use_container_width=True):
            st.session_state[FAVORITES_APPLY_NONCE_KEY] = int(st.session_state.get(FAVORITES_APPLY_NONCE_KEY, 0)) + 1
            st.session_state[FAVORITES_FORCE_ENABLE_KEY] = True
            st.success("Favoris appliques. Recharge en cours...")
            st.rerun()

    st.session_state[apply_seen_key] = int(st.session_state.get(FAVORITES_APPLY_NONCE_KEY, 0))

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
