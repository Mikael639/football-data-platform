from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from .styles import VISUAL_COLORS


def chart_base(df: pd.DataFrame) -> alt.Chart:
    return alt.Chart(df).configure_axis(labelColor="#29415f", titleColor="#29415f").configure_view(strokeOpacity=0)


def render_position_curve(df: pd.DataFrame, height: int = 320) -> None:
    if df is None or df.empty:
        st.info("Aucune donnee disponible pour la courbe de classement.")
        return

    chart = (
        chart_base(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("matchday:Q", title="Journee"),
            y=alt.Y("position:Q", title="Position", sort="descending", scale=alt.Scale(reverse=True, zero=False)),
            color=alt.Color("team_name:N", title="Equipe"),
            tooltip=["team_name", "matchday", "position", "points"],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_form_chart(df: pd.DataFrame, height: int = 220) -> None:
    if df is None or df.empty:
        st.info("Aucune donnee disponible pour la forme.")
        return

    chart = (
        chart_base(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("match_label:N", title="Match"),
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(
                    domain=["W", "D", "L"],
                    range=[VISUAL_COLORS["attack"], VISUAL_COLORS["neutral"], VISUAL_COLORS["defense"]],
                ),
                legend=None,
            ),
            tooltip=["match_label", "opponent_name", "result", "points", "goals_for", "goals_against"],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_home_away_chart(df: pd.DataFrame, height: int = 220) -> None:
    if df is None or df.empty:
        st.info("Aucune donnee disponible pour domicile/exterieur.")
        return

    melted = df.melt(id_vars=["venue"], value_vars=["GoalsFor", "GoalsAgainst"], var_name="metric", value_name="value")
    chart = (
        chart_base(melted)
        .mark_bar()
        .encode(
            x=alt.X("venue:N", title=""),
            y=alt.Y("value:Q", title="Buts"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["GoalsFor", "GoalsAgainst"], range=[VISUAL_COLORS["attack"], VISUAL_COLORS["defense"]]),
                title="",
            ),
            xOffset="metric:N",
            tooltip=["venue", "metric", "value"],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)
