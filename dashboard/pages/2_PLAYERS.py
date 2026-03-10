import pandas as pd
import streamlit as st

from data.dashboard_data import get_player_impact_stats, get_players, get_team_meta
from state.filters import render_global_filters
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_section_heading, render_team_header
from ui.exports import render_csv_download
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="JOUEURS - Football Data Platform", layout="wide")


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "JOUEURS",
        "Vue effectif pour analyser les joueurs disponibles selon le club et la competition selectionnes.",
        "Players.png",
    )
    filters = render_global_filters("players")
    render_note_card("Utilisez les filtres pour affiner l effectif et analyser les joueurs disponibles pour un club.")

    if filters.team_id is not None:
        render_team_header(get_team_meta(filters.team_id))

    players = get_players(filters.team_id, filters.competition_id, filters.season)
    if players.empty:
        st.info("Aucun joueur disponible pour ce filtre.")
        return

    impact = get_player_impact_stats(filters.team_id, filters.competition_id, filters.season)
    if impact.empty:
        enriched = players.copy()
        enriched["matches_played"] = 0
        enriched["minutes"] = 0
        enriched["goals"] = 0
        enriched["assists"] = 0
        enriched["goal_contrib"] = 0
        enriched["goal_contrib_p90"] = 0.0
    else:
        enriched = players.merge(
            impact[
                [
                    "player_id",
                    "matches_played",
                    "minutes",
                    "goals",
                    "assists",
                    "goal_contrib",
                    "goal_contrib_p90",
                ]
            ],
            on="player_id",
            how="left",
        )
        for column in ["matches_played", "minutes", "goals", "assists", "goal_contrib", "goal_contrib_p90"]:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0)

    enriched["position_clean"] = enriched["position"].fillna("-").astype(str).replace("", "-")
    enriched["nationality_clean"] = enriched["nationality"].fillna("-").astype(str).replace("", "-")
    position_options = sorted(enriched["position_clean"].dropna().astype(str).unique().tolist())
    nationality_options = sorted(enriched["nationality_clean"].dropna().astype(str).unique().tolist())
    max_minutes = int(enriched["minutes"].max()) if not enriched.empty else 0
    max_minutes = max(max_minutes, 0)

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_positions = st.multiselect("Poste", position_options, key="players_position_filter")
    with f2:
        selected_nationalities = st.multiselect("Nationalite", nationality_options, key="players_nationality_filter")
    with f3:
        slider_key = "players_minutes_min_filter"
        current_slider_value = st.session_state.get(slider_key, 0)
        try:
            current_slider_value = int(current_slider_value)
        except (TypeError, ValueError):
            current_slider_value = 0
        current_slider_value = min(max(current_slider_value, 0), max_minutes)
        st.session_state[slider_key] = current_slider_value

        if max_minutes <= 0:
            st.number_input(
                "Minutes min",
                min_value=0,
                max_value=0,
                value=0,
                step=1,
                key="players_minutes_min_filter_disabled",
                disabled=True,
            )
            minutes_min = 0
        else:
            minutes_min = st.slider(
                "Minutes min",
                min_value=0,
                max_value=max_minutes,
                step=10 if max_minutes >= 10 else 1,
                key=slider_key,
            )

    scoped = enriched.copy()
    if selected_positions:
        scoped = scoped[scoped["position_clean"].isin(selected_positions)]
    if selected_nationalities:
        scoped = scoped[scoped["nationality_clean"].isin(selected_nationalities)]
    scoped = scoped[scoped["minutes"] >= int(minutes_min)]

    if scoped.empty:
        st.info("Aucun joueur disponible avec ces filtres.")
        return

    meta = scoped.copy()
    top = st.columns(3)
    top[0].metric("Joueurs", len(meta))
    top[1].metric("Postes", meta["position"].fillna("-").replace("", "-").nunique())
    top[2].metric("Nationalites", meta["nationality"].fillna("-").replace("", "-").nunique())

    render_section_heading("Top impact offensif")
    impact_table = (
        scoped.sort_values(["goal_contrib_p90", "goal_contrib", "minutes"], ascending=[False, False, False])
        .rename(
            columns={
                "full_name": "Joueur",
                "team_name": "Club",
                "position_clean": "Poste",
                "matches_played": "Matchs",
                "minutes": "Minutes",
                "goals": "Buts",
                "assists": "Passes dec.",
                "goal_contrib": "G+A",
                "goal_contrib_p90": "G+A p90",
            }
        )
        .head(12)
    )
    impact_table["G+A p90"] = impact_table["G+A p90"].round(2)
    render_adaptive_table(
        impact_table[["Joueur", "Club", "Poste", "Matchs", "Minutes", "Buts", "Passes dec.", "G+A", "G+A p90"]],
        title="Top joueurs impact",
        strong_columns={"Joueur"},
        max_height=620,
    )
    render_csv_download(
        df=impact_table,
        label="Exporter top impact (CSV)",
        filename="players_top_impact.csv",
        key="players_export_top_impact",
    )

    render_section_heading("Effectif disponible")
    players_table = scoped.rename(
            columns={
                "full_name": "Joueur",
                "position": "Poste",
                "nationality": "Nationalite",
                "birth_date": "Date de naissance",
                "team_name": "Club",
                "matches_played": "Matchs",
                "minutes": "Minutes",
                "goals": "Buts",
                "assists": "Passes dec.",
            }
        )[["Joueur", "Poste", "Nationalite", "Date de naissance", "Club", "Matchs", "Minutes", "Buts", "Passes dec."]]
    render_adaptive_table(
        players_table,
        title="Liste de l effectif",
        strong_columns={"Joueur"},
        max_height=980,
    )
    render_csv_download(
        df=players_table,
        label="Exporter effectif (CSV)",
        filename="players_squad.csv",
        key="players_export_squad",
    )


if __name__ == "__main__":
    main()

