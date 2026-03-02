import streamlit as st

from data.dashboard_data import get_players, get_team_meta
from state.filters import render_global_filters
from ui.display import render_team_header
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="Joueurs - Football Data Platform", layout="wide")


def main() -> None:
    inject_dashboard_styles()
    st.title("Joueurs")
    filters = render_global_filters("players")

    if filters.team_id is not None:
        render_team_header(get_team_meta(filters.team_id))

    players = get_players(filters.team_id)
    if players.empty:
        st.info("Aucun joueur disponible pour ce filtre.")
        return

    st.dataframe(
        players.rename(
            columns={
                "full_name": "Joueur",
                "position": "Poste",
                "nationality": "Nationalite",
                "birth_date": "Date de naissance",
                "team_name": "Equipe",
            }
        )[["Joueur", "Poste", "Nationalite", "Date de naissance", "Equipe"]],
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
