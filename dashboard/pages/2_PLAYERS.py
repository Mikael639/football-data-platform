import streamlit as st

from data.dashboard_data import get_players, get_team_meta
from state.filters import render_global_filters
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_section_heading, render_team_header
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="PLAYERS - Football Data Platform", layout="wide")


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "PLAYERS",
        "Squad view for exploring the players currently available for the selected club and competition.",
        "Players.png",
    )
    filters = render_global_filters("players")
    render_note_card("Use the filters to narrow the squad view and inspect the players currently available for a club.")

    if filters.team_id is not None:
        render_team_header(get_team_meta(filters.team_id))

    players = get_players(filters.team_id, filters.competition_id, filters.season)
    if players.empty:
        st.info("No players available for this filter.")
        return

    meta = players.copy()
    top = st.columns(3)
    top[0].metric("Players", len(meta))
    top[1].metric("Positions", meta["position"].fillna("-").replace("", "-").nunique())
    top[2].metric("Nationalities", meta["nationality"].fillna("-").replace("", "-").nunique())

    render_section_heading("Available squad")
    render_adaptive_table(
        players.rename(
            columns={
                "full_name": "Player",
                "position": "Position",
                "nationality": "Nationality",
                "birth_date": "Birth date",
                "team_name": "Team",
            }
        )[["Player", "Position", "Nationality", "Birth date", "Team"]],
        title="Squad list",
        strong_columns={"Player"},
        max_height=980,
    )


if __name__ == "__main__":
    main()
