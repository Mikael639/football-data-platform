import altair as alt
import pandas as pd
import streamlit as st

from .styles import VISUAL_COLORS


def render_sorted_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    descending: bool = True,
    height: int = 320,
    bar_color: str | None = None,
    signed_colors: bool = False,
) -> None:
    if df is None or df.empty:
        st.info("Aucune donnee pour ce graphique.")
        return

    chart_df = df[[category_col, value_col]].copy()
    chart_df = chart_df.dropna(subset=[category_col, value_col])
    if chart_df.empty:
        st.info("Aucune donnee pour ce graphique.")
        return

    chart_df = chart_df.sort_values(value_col, ascending=not descending).reset_index(drop=True)
    chart_df["rank_order"] = range(1, len(chart_df) + 1)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title=value_col),
            y=alt.Y(
                f"{category_col}:N",
                sort=alt.SortField(field="rank_order", order="ascending"),
                title="",
            ),
            tooltip=[category_col, value_col],
            color=(
                alt.condition(
                    alt.datum[value_col] < 0,
                    alt.value(VISUAL_COLORS["defense"]),
                    alt.value(VISUAL_COLORS["attack"]),
                )
                if signed_colors
                else alt.value(bar_color or VISUAL_COLORS["points"])
            ),
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_result_distribution_chart(df: pd.DataFrame, height: int = 220) -> None:
    if df is None or df.empty:
        st.info("Aucune donnee pour ce graphique.")
        return
    order = ["W", "D", "L"]
    color_map = {"W": VISUAL_COLORS["attack"], "D": VISUAL_COLORS["neutral"], "L": VISUAL_COLORS["defense"]}
    chart_df = df.copy()
    chart_df["rank_order"] = chart_df["Result"].map({k: i for i, k in enumerate(order, start=1)}).fillna(99)
    chart_df["color"] = chart_df["Result"].map(color_map).fillna(VISUAL_COLORS["points"])
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Count:Q", title="Nombre"),
            y=alt.Y("Result:N", sort=alt.SortField("rank_order", order="ascending"), title=""),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=["Result", "Count"],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_ppm_chart(df: pd.DataFrame, height: int = 220) -> None:
    if df is None or df.empty:
        st.info("Aucune donnee pour ce graphique.")
        return
    chart_df = df.copy()
    colors = {"Domicile": VISUAL_COLORS["points"], "Exterieur": VISUAL_COLORS["teal"]}
    chart_df["color"] = chart_df["venue"].map(colors).fillna(VISUAL_COLORS["points"])
    chart_df = chart_df.sort_values("PPM", ascending=False)
    chart_df["rank_order"] = range(1, len(chart_df) + 1)
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("PPM:Q", title="PPM"),
            y=alt.Y("venue:N", sort=alt.SortField("rank_order", order="ascending"), title=""),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=["venue", "PPM"],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)
