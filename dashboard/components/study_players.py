import json
import os
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from ui.charts import render_sorted_bar_chart
from ui.labels import (
    LABEL_ALL_PLAYERS,
    LABEL_ALL_SEASONS,
    LABEL_NO_PLAYERS,
    STUDY_INFO_MANUAL_MODE,
    STUDY_INFO_MISSING_DATA,
    STUDY_INFO_PLAYER_NO_REGULARITY,
    STUDY_INFO_REGULARITY_UNAVAILABLE,
    STUDY_INFO_SELECT_PLAYER_TEMPLATE,
    STUDY_SECTION_SELECTION,
    STUDY_SECTION_VIEWS,
    STUDY_SUBTAB_LEADERS,
    STUDY_TITLE,
    STUDY_WARNING_EMPTY_FILES,
)
from ui.styles import VISUAL_COLORS
from data.study_players_helpers import (
    add_podium_icons_generic,
    build_study_leaders_scope,
    season_label_from_start,
    season_picker_label_from_start,
    study_expected_complete_seasons,
)

STUDY_FBREF_DIR = Path(__file__).resolve().parents[2] / "data" / "study" / "fbref"

@st.cache_data(show_spinner=False, ttl=60)
def load_fbref_study_datasets(_backend_hint: str | None = None):
    backend = (os.getenv("FBREF_STUDY_BACKEND") or "local").strip().lower()
    supabase_db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("STUDY_SUPABASE_DB_URL")
    if backend in {"supabase", "postgres"} and supabase_db_url:
        try:
            engine = create_engine(supabase_db_url, pool_pre_ping=True)
            data = {}
            queries = {
                "player_match": "SELECT * FROM public.study_fbref_player_match",
                "player_season": "SELECT * FROM public.study_fbref_player_season",
                "regularity": "SELECT * FROM public.study_fbref_regularity",
                "progression": "SELECT * FROM public.study_fbref_progression",
            }
            for key, sql_query in queries.items():
                try:
                    data[key] = pd.read_sql(text(sql_query), engine)
                except Exception:
                    data[key] = pd.DataFrame()
            try:
                meta_df = pd.read_sql(
                    text("SELECT * FROM public.study_fbref_meta WHERE dataset_name = 'fbref_study' LIMIT 1"),
                    engine,
                )
                if not meta_df.empty:
                    row = meta_df.iloc[0].to_dict()
                    for json_col in ["seasons_start_years", "season_labels", "files"]:
                        if json_col in row and isinstance(row[json_col], str):
                            try:
                                row[json_col] = json.loads(row[json_col])
                            except Exception:
                                pass
                    data["meta"] = row
                else:
                    data["meta"] = None
            except Exception:
                data["meta"] = None

            if any(not data[k].empty for k in ["player_match", "player_season", "regularity", "progression"]):
                return data
        except Exception:
            pass

    files = {
        "player_match": STUDY_FBREF_DIR / "player_match.csv",
        "player_season": STUDY_FBREF_DIR / "player_season.csv",
        "regularity": STUDY_FBREF_DIR / "regularity.csv",
        "progression": STUDY_FBREF_DIR / "progression.csv",
        "meta": STUDY_FBREF_DIR / "meta.json",
    }
    if not any(files[k].exists() for k in ["player_match", "player_season", "regularity", "progression"]):
        return None

    data = {}
    for key, path in files.items():
        if key == "meta":
            if path.exists():
                try:
                    data["meta"] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data["meta"] = None
            else:
                data["meta"] = None
            continue
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
    return data


def _render_study_scatter(df: pd.DataFrame, x_col: str, y_col: str, label_col: str, color_col: str) -> None:
    if df is None or df.empty:
        st.info("Aucune donnee pour ce graphique.")
        return
    chart = (
        alt.Chart(df)
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            color=alt.Color(f"{color_col}:N", title="Poste"),
            tooltip=[label_col, color_col, x_col, y_col],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)


def render_study_leaders_section(
    season_df: pd.DataFrame,
    *,
    selected_season: int | None,
    selected_season_label: str | None,
    selected_pos: str = "Tous",
) -> None:
    st.subheader("Leaders (Buteurs, Passeurs, Penaltys, etc.)")
    if season_df is None or season_df.empty:
        st.info("Donnees joueurs-saison indisponibles pour les leaders.")
        return

    leaders_scope = build_study_leaders_scope(season_df, selected_season, selected_pos)
    if leaders_scope.empty:
        st.info("Aucune donnee leaders pour ce filtre.")
        return

    metric_options = {
        "Buteurs": ("goals_total", "Buts"),
        "Passeurs": ("assists_total", "Passes D"),
        "G+A": ("ga_total", "G+A"),
        "Buts sur penalty": ("pk_goals_total", "Buts (PK)"),
        "Buts hors penalty": ("goals_non_pk_total", "Buts (hors PK)"),
        "Cartons jaunes": ("yellow_cards_total", "Jaunes"),
        "Cartons rouges": ("red_cards_total", "Rouges"),
    }

    l1, l2 = st.columns([0.45, 0.55])
    with l1:
        metric_label = st.selectbox("Metrique leaders", list(metric_options.keys()), key="study_leaders_metric")
    with l2:
        scope_text = selected_season_label if (selected_season_label and selected_season is not None) else LABEL_ALL_SEASONS
        st.caption(f"Scope leaders: {scope_text} | Poste: {selected_pos}")

    metric_col, metric_display = metric_options[metric_label]
    if metric_col not in leaders_scope.columns:
        st.info(f"Metrique indisponible dans la source actuelle: {metric_label}")
        return

    rank_df = leaders_scope.copy()
    rank_df[metric_col] = pd.to_numeric(rank_df[metric_col], errors="coerce").fillna(0)
    rank_df = rank_df[rank_df[metric_col] > 0].copy()
    if rank_df.empty:
        st.info("Aucune valeur non nulle pour cette metrique sur ce filtre.")
        return

    rank_df = rank_df.sort_values([metric_col, "minutes_total"], ascending=[False, False]).head(10).copy()
    rank_df["Joueur"] = rank_df["player_name"].astype(str)
    rank_df = add_podium_icons_generic(rank_df, "Joueur")
    if "ga_p90" in rank_df.columns:
        rank_df["G+A/90"] = pd.to_numeric(rank_df["ga_p90"], errors="coerce").fillna(0).round(2)
    if "clubs_count" in rank_df.columns:
        rank_df["Nb clubs"] = pd.to_numeric(rank_df["clubs_count"], errors="coerce").fillna(1).astype(int)
    if "clubs_list" in rank_df.columns:
        rank_df["Clubs"] = rank_df["clubs_list"].fillna(rank_df.get("team_name", "")).astype(str)

    cols = ["Joueur", "team_name", "position_group", metric_col, "minutes_total", "G+A/90", "Nb clubs", "Clubs"]
    cols = [c for c in cols if c in rank_df.columns]
    st.dataframe(
        rank_df[cols].rename(
            columns={
                "team_name": "Club affiche",
                "position_group": "Poste",
                metric_col: metric_display,
                "minutes_total": "Min",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )



def render_player_study_tab() -> None:
    st.header(STUDY_TITLE)
    refresh_col, _ = st.columns([0.24, 0.76])
    with refresh_col:
        if st.button("Rafraichir les donnees", key="study_refresh_btn", use_container_width=True):
            load_fbref_study_datasets.clear()
            st.rerun()

    study = load_fbref_study_datasets(os.getenv("FBREF_STUDY_BACKEND", "local"))
    expected = study_expected_complete_seasons(3)
    expected_labels = ", ".join(season_label_from_start(s) for s in expected)

    if not study:
        st.info(STUDY_INFO_MISSING_DATA)
        st.caption(STUDY_INFO_MANUAL_MODE)
        st.caption(f"Saisons ciblees (3 dernieres saisons completes): {expected_labels}")
        return

    meta = study.get("meta") or {}
    reg_df = study.get("regularity", pd.DataFrame()).copy()
    prog_df = study.get("progression", pd.DataFrame()).copy()
    season_df = study.get("player_season", pd.DataFrame()).copy()
    match_df = study.get("player_match", pd.DataFrame()).copy()

    if reg_df.empty and prog_df.empty and season_df.empty and match_df.empty:
        st.warning(STUDY_WARNING_EMPTY_FILES)
        return

    generated_seasons = meta.get("season_labels") if isinstance(meta, dict) else None
    if generated_seasons:
        st.caption(
            "Saisons etudiees (FBref, saisons completes hors saison en cours): "
            + ", ".join(generated_seasons)
        )
    else:
        st.caption(f"Saisons cibles (attendues): {expected_labels}")

    seasons_metric_values = set()
    for df in (season_df, reg_df, prog_df):
        if isinstance(df, pd.DataFrame) and (not df.empty) and ("season_start" in df.columns):
            seasons_metric_values.update(df["season_start"].dropna().astype(int).tolist())
    player_count_metric = 0
    if not season_df.empty and "player_id" in season_df.columns:
        player_count_metric = int(season_df["player_id"].nunique())
    elif not reg_df.empty and "player_id" in reg_df.columns:
        player_count_metric = int(reg_df["player_id"].nunique())
    elif not prog_df.empty and "player_id" in prog_df.columns:
        player_count_metric = int(prog_df["player_id"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saisons", int(len(seasons_metric_values)))
    c2.metric("Joueurs (etude)", player_count_metric)
    c3.metric("Lignes progression", int(len(prog_df)))
    c4.metric("Matchs joueurs", int(len(match_df)) if not match_df.empty else 0)

    if reg_df.empty:
        st.info(STUDY_INFO_REGULARITY_UNAVAILABLE)

    season_candidates = set()
    for df in (season_df, reg_df, prog_df):
        if isinstance(df, pd.DataFrame) and (not df.empty) and ("season_start" in df.columns):
            season_candidates.update(df["season_start"].dropna().astype(int).tolist())
    available_seasons = sorted(season_candidates, reverse=True)
    season_label_map = {LABEL_ALL_SEASONS: None}
    season_label_map.update({season_picker_label_from_start(s): s for s in available_seasons})

    st.markdown(STUDY_SECTION_SELECTION)
    sel1, sel2, sel3 = st.columns([0.32, 0.32, 0.36])
    with sel1:
        selected_global_season_label = (
            st.selectbox("Saison", list(season_label_map.keys()), key="study_global_season")
            if season_label_map
            else None
        )
    selected_global_season = season_label_map.get(selected_global_season_label) if selected_global_season_label else None

    players_for_selector = pd.DataFrame()
    if not season_df.empty and "season_start" in season_df.columns:
        if selected_global_season is None:
            players_for_selector = season_df.copy()
        else:
            players_for_selector = season_df[season_df["season_start"].astype(int) == int(selected_global_season)].copy()
    elif not reg_df.empty and "season_start" in reg_df.columns:
        if selected_global_season is None:
            players_for_selector = reg_df.copy()
        else:
            players_for_selector = reg_df[reg_df["season_start"].astype(int) == int(selected_global_season)].copy()

    if not players_for_selector.empty:
        if "minutes_total" in players_for_selector.columns:
            players_for_selector = players_for_selector.sort_values(
                ["minutes_total", "player_name"], ascending=[False, True]
            )
        players_for_selector = players_for_selector.drop_duplicates(subset=["player_id"], keep="first")
        players_for_selector["player_label"] = (
            players_for_selector.get("player_name", pd.Series("", index=players_for_selector.index)).astype(str)
            + " - "
            + players_for_selector.get("team_name", pd.Series("", index=players_for_selector.index)).fillna("").astype(str)
            + " ("
            + players_for_selector.get("position_group", pd.Series("", index=players_for_selector.index)).fillna("").astype(str)
            + ")"
        ).str.replace("  ", " ", regex=False)

        player_labels = [LABEL_ALL_PLAYERS] + players_for_selector["player_label"].tolist()
        with sel2:
            selected_player_label = st.selectbox(
                "Joueur (clique / recherche)",
                player_labels,
                key="study_player_selector",
            )
        player_lookup = {LABEL_ALL_PLAYERS: None}
        player_lookup.update(dict(zip(players_for_selector["player_label"], players_for_selector["player_id"])))
        selected_player_id = player_lookup.get(selected_player_label)
    else:
        with sel2:
            st.selectbox("Joueur (clique / recherche)", [LABEL_NO_PLAYERS], key="study_player_selector_empty")
        selected_player_label = None
        selected_player_id = None

    position_options = ["Tous"] + sorted(
        [p for p in reg_df.get("position_group", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if p]
    )
    with sel3:
        selected_pos = st.selectbox("Poste (classement regularite)", position_options, key="study_regularite_position")

    selected_season = selected_global_season

    st.markdown(STUDY_SECTION_VIEWS)
    (leaders_tab,) = st.tabs([STUDY_SUBTAB_LEADERS])
    with leaders_tab:
        render_study_leaders_section(
            season_df,
            selected_season=selected_season,
            selected_season_label=selected_global_season_label,
            selected_pos=selected_pos,
                    )

    st.divider()

    reg_scope = reg_df.copy()
    if selected_season is not None:
        reg_scope = reg_scope[reg_scope["season_start"].astype(int) == selected_season]
    if selected_pos != "Tous":
        reg_scope = reg_scope[reg_scope["position_group"] == selected_pos]

    r1, r2 = st.columns([1.2, 1.8])
    with r1:
        st.subheader("Top regularite")
        if reg_scope.empty:
            st.info("Aucune donnee pour ce filtre.")
        else:
            top_reg = reg_scope.sort_values(
                ["regularity_score", "ga_p90_mean", "minutes_total"], ascending=[False, False, False]
            ).head(10).copy()
            top_reg["Joueur"] = top_reg["player_name"].astype(str)
            top_reg = add_podium_icons_generic(top_reg, "Joueur")
            top_reg["Score"] = top_reg["regularity_score"].round(3)
            top_reg["GA/90"] = top_reg["ga_p90_mean"].round(2)
            top_reg["Variabilite"] = top_reg["stability_proxy"].round(2)
            st.dataframe(
                top_reg[["Joueur", "team_name", "position_group", "minutes_total", "GA/90", "Variabilite", "Score"]]
                .rename(columns={"team_name": "Club", "position_group": "Poste", "minutes_total": "Min"}),
                use_container_width=True,
                hide_index=True,
            )

            bar_df = top_reg.copy()
            bar_df["JoueurScore"] = bar_df["Joueur"]
            render_sorted_bar_chart(bar_df, "JoueurScore", "regularity_score", descending=True, bar_color=VISUAL_COLORS["teal"])

    with r2:
        st.subheader("Performance vs variabilite")
        if reg_scope.empty:
            st.info("Aucune donnee pour ce filtre.")
        else:
            scatter_cols = ["player_name", "position_group", "ga_p90_mean", "stability_proxy"]
            scatter_df = reg_scope[scatter_cols].copy().rename(
                columns={
                    "player_name": "Joueur",
                    "position_group": "Poste",
                    "ga_p90_mean": "Perf_GA90",
                    "stability_proxy": "Variabilite",
                }
            )
            _render_study_scatter(scatter_df, "Perf_GA90", "Variabilite", "Joueur", "Poste")
            st.caption("Interpretation: plus a droite = meilleure production (GA/90), plus bas = plus regulier.")

    st.divider()
    st.subheader("Fiche joueur (selection par nom)")
    if selected_player_id is None:
        st.info(STUDY_INFO_SELECT_PLAYER_TEMPLATE.format(all_players=LABEL_ALL_PLAYERS))
    else:
        player_season_rows = season_df[season_df["player_id"] == selected_player_id].copy() if not season_df.empty else pd.DataFrame()
        if not player_season_rows.empty:
            player_season_rows["season_start"] = player_season_rows["season_start"].astype(int)
            player_season_rows = player_season_rows.sort_values("season_start", ascending=False)
        player_reg_rows = reg_df[reg_df["player_id"] == selected_player_id].copy() if not reg_df.empty else pd.DataFrame()
        player_prog_rows = prog_df[prog_df["player_id"] == selected_player_id].copy() if not prog_df.empty else pd.DataFrame()
        player_match_rows = match_df[match_df["player_id"] == selected_player_id].copy() if not match_df.empty else pd.DataFrame()

        player_season_row = pd.DataFrame()
        if (not player_season_rows.empty) and (selected_season is not None):
            player_season_row = player_season_rows[player_season_rows["season_start"] == int(selected_season)].head(1)
        if player_season_row.empty and not player_season_rows.empty:
            player_season_row = player_season_rows.head(1)

        display_name = selected_player_label or "Joueur"
        if not player_season_row.empty:
            row0 = player_season_row.iloc[0]
            display_name = f"{row0.get('player_name', display_name)} - {row0.get('team_name', '')} ({row0.get('position_group', '')})"

        st.markdown(
            f"<div class='fdp-hero'><div class='fdp-hero-title'>{display_name}</div>"
            "<div class='fdp-hero-sub'>Performance, regularite et progression sur les 3 dernieres saisons completes.</div></div>",
            unsafe_allow_html=True,
        )

        if not player_season_row.empty:
            row0 = player_season_row.iloc[0]
            clubs_count = int(pd.to_numeric(pd.Series([row0.get("clubs_count", 1)]), errors="coerce").fillna(1).iloc[0])
            clubs_list = str(row0.get("clubs_list", row0.get("team_name", "")) or "")
            is_multi = bool(row0.get("is_multi_club_season", False)) or clubs_count > 1
            if is_multi:
                st.caption(f"Qualite data: saison multi-clubs detectee (transfert/pret) | Clubs: {clubs_list}")

        if not player_season_row.empty:
            row0 = player_season_row.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Minutes", int(row0.get("minutes_total", 0)))
            m2.metric("Matchs", int(row0.get("matches_played", 0)))
            m3.metric("Titularisations", int(row0.get("starts", 0)))
            m4.metric("G+A / 90", round(float(row0.get("ga_p90", 0.0)), 2))
            m5.metric("Passes / 90", round(float(row0.get("passes_p90", 0.0)), 1))

        tab_perf, tab_reg, tab_prog = st.tabs(["Performance", "Regularite", "Progression"])

        with tab_perf:
            if player_season_rows.empty:
                st.info("Aucune donnee joueur-saison disponible.")
            else:
                season_perf = player_season_rows.copy()
                season_perf["Saison"] = season_perf["season_start"].astype(int).map(season_picker_label_from_start)
                season_perf = season_perf.sort_values("season_start", ascending=True)
                if "is_multi_club_season" in season_perf.columns:
                    season_perf["Contexte club"] = np.where(
                        season_perf["is_multi_club_season"].fillna(False),
                        "Transfert/Pret",
                        "Club unique",
                    )
                if "clubs_count" in season_perf.columns:
                    season_perf["Nb clubs"] = pd.to_numeric(season_perf["clubs_count"], errors="coerce").fillna(1).astype(int)
                if "clubs_list" in season_perf.columns:
                    season_perf["Clubs saison"] = season_perf["clubs_list"].fillna(season_perf.get("team_name", ""))

                pcols = [
                    "Saison", "team_name", "position_group", "Contexte club", "Nb clubs", "Clubs saison",
                    "minutes_total", "matches_played", "starts", "goals_total", "assists_total", "ga_total", "ga_p90", "passes_p90"
                ]
                available_pcols = [c for c in pcols if c in season_perf.columns]
                perf_table = season_perf[available_pcols].rename(
                    columns={
                        "team_name": "Club",
                        "position_group": "Poste",
                        "Contexte club": "Contexte",
                        "Nb clubs": "Nb clubs",
                        "Clubs saison": "Clubs (saison)",
                        "minutes_total": "Min",
                        "matches_played": "Matchs",
                        "starts": "Tit.",
                        "goals_total": "Buts",
                        "assists_total": "Passes D",
                        "ga_total": "G+A",
                        "ga_p90": "G+A/90",
                        "passes_p90": "Passes/90",
                    }
                )
                st.dataframe(perf_table, use_container_width=True, hide_index=True)

                if {"Saison", "G+A/90"}.issubset(perf_table.columns):
                    season_line = season_perf[["Saison"]].copy()
                    season_line["G+A/90"] = season_perf["ga_p90"].round(2)
                    season_line["Passes/90"] = season_perf.get("passes_p90", pd.Series(0, index=season_perf.index)).round(1)
                    line_long = season_line.melt("Saison", var_name="Metrique", value_name="Valeur")
                    chart = (
                        alt.Chart(line_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Saison:N", title="Saison"),
                            y=alt.Y("Valeur:Q", title="Valeur"),
                            color=alt.Color("Metrique:N", scale=alt.Scale(range=[VISUAL_COLORS["points"], VISUAL_COLORS["teal"]])),
                            tooltip=["Saison", "Metrique", "Valeur"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart, use_container_width=True)

            if not player_match_rows.empty and selected_season is not None:
                match_scope = player_match_rows[player_match_rows["season_start"].astype(int) == int(selected_season)].copy()
                if not match_scope.empty:
                    st.caption(
                        f"Matchs (saison {selected_global_season_label})"
                        if selected_global_season_label and selected_global_season is not None
                        else "Matchs (toutes saisons)"
                    )
                    if "date_id" in match_scope.columns:
                        match_scope["date_id"] = pd.to_datetime(match_scope["date_id"], errors="coerce")
                        match_scope = match_scope.sort_values("date_id")
                        match_scope["Date"] = match_scope["date_id"].dt.strftime("%Y-%m-%d")
                    else:
                        match_scope["Date"] = range(1, len(match_scope) + 1)
                    for col in ["ga_p90_match", "shots_p90_match", "passes_p90_match"]:
                        if col not in match_scope.columns:
                            match_scope[col] = 0.0
                    match_long = match_scope[["Date", "ga_p90_match", "shots_p90_match"]].rename(
                        columns={"ga_p90_match": "G+A/90", "shots_p90_match": "Tirs/90"}
                    ).melt("Date", var_name="Metrique", value_name="Valeur")
                    match_chart = (
                        alt.Chart(match_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Date:N", title="Match"),
                            y=alt.Y("Valeur:Q", title="Par 90"),
                            color=alt.Color("Metrique:N", scale=alt.Scale(range=[VISUAL_COLORS["attack"], VISUAL_COLORS["violet"]])),
                            tooltip=["Date", "Metrique", "Valeur"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(match_chart, use_container_width=True)

        with tab_reg:
            if player_reg_rows.empty:
                st.info(STUDY_INFO_PLAYER_NO_REGULARITY)
            else:
                player_reg_rows["season_start"] = player_reg_rows["season_start"].astype(int)
                player_reg_rows = player_reg_rows.sort_values("season_start", ascending=False)
                reg_current = (
                    player_reg_rows[player_reg_rows["season_start"] == int(selected_season)].head(1)
                    if selected_season is not None
                    else pd.DataFrame()
                )
                if not reg_current.empty:
                    rr = reg_current.iloc[0]
                    rr1, rr2, rr3, rr4 = st.columns(4)
                    rr1.metric("Rang regularite (poste)", int(rr.get("regularity_rank_pos", 0)))
                    rr2.metric("Score regularite", round(float(rr.get("regularity_score", 0.0)), 3))
                    rr3.metric("GA/90 moyen", round(float(rr.get("ga_p90_mean", 0.0)), 2))
                    rr4.metric("Variabilite", round(float(rr.get("stability_proxy", 0.0)), 3))

                reg_table = player_reg_rows.copy()
                reg_table["Saison"] = reg_table["season_start"].map(season_picker_label_from_start)
                if "podium" in reg_table.columns:
                    reg_table["Joueur"] = (reg_table["podium"].fillna("") + " " + reg_table["player_name"].astype(str)).str.strip()
                reg_cols = ["Saison", "team_name", "position_group", "minutes_total", "regularity_rank_pos", "ga_p90_mean", "stability_proxy", "regularity_score"]
                reg_cols = [c for c in reg_cols if c in reg_table.columns]
                st.dataframe(
                    reg_table[reg_cols].rename(
                        columns={
                            "team_name": "Club",
                            "position_group": "Poste",
                            "minutes_total": "Min",
                            "regularity_rank_pos": "Rang poste",
                            "ga_p90_mean": "GA/90 moyen",
                            "stability_proxy": "Variabilite",
                            "regularity_score": "Score",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        with tab_prog:
            if player_prog_rows.empty:
                st.info("Aucune ligne de progression pour ce joueur (il faut 2 saisons eligibles).")
            else:
                player_prog_rows["season_start"] = player_prog_rows["season_start"].astype(int)
                player_prog_rows = player_prog_rows.sort_values("season_start", ascending=False)
                prog_current = (
                    player_prog_rows[player_prog_rows["season_start"] == int(selected_season)].head(1)
                    if selected_season is not None
                    else pd.DataFrame()
                )
                if not prog_current.empty:
                    pr = prog_current.iloc[0]
                    pr1, pr2, pr3, pr4 = st.columns(4)
                    pr1.metric("Score progression", round(float(pr.get("progress_score", 0.0)), 3))
                    pr2.metric("Delta G+A/90", round(float(pr.get("delta_ga_p90", 0.0)), 2))
                    pr3.metric("Delta Passes/90", round(float(pr.get("delta_passes_p90", 0.0)), 2))
                    pr4.metric("Delta minutes", int(pr.get("delta_minutes_total", 0)))
                else:
                    st.caption("Pas de progression pour cette saison selectionnee (ex: premiere saison de la serie).")

                prog_table = player_prog_rows.copy()
                prog_table["Saison N"] = prog_table["season_start"].map(season_picker_label_from_start)
                if "podium" in prog_table.columns:
                    prog_table["Joueur"] = (prog_table["podium"].fillna("") + " " + prog_table["player_name"].astype(str)).str.strip()
                prog_cols = ["Saison N", "team_name", "position_group", "delta_ga_p90", "delta_passes_p90", "delta_pass_acc_mean", "delta_minutes_total", "progress_score", "progress_rank_pos"]
                prog_cols = [c for c in prog_cols if c in prog_table.columns]
                st.dataframe(
                    prog_table[prog_cols].rename(
                        columns={
                            "team_name": "Club",
                            "position_group": "Poste",
                            "delta_ga_p90": "Delta GA/90",
                            "delta_passes_p90": "Delta Passes/90",
                            "delta_pass_acc_mean": "Delta Precision passes",
                            "delta_minutes_total": "Delta Min",
                            "progress_score": "Score",
                            "progress_rank_pos": "Rang poste",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    st.divider()
    st.subheader("Progression / Regression (saison vs saison precedente)")
    if prog_df.empty:
        st.info("Donnees progression indisponibles.")
        return

    prog_seasons = sorted(prog_df["season_start"].dropna().astype(int).unique().tolist(), reverse=True)
    prog_label_map = {LABEL_ALL_SEASONS: None}
    prog_label_map.update({season_picker_label_from_start(s): s for s in prog_seasons})
    p1, p2 = st.columns(2)
    with p1:
        selected_prog_label = st.selectbox(
            "Saison progression (saison N)",
            list(prog_label_map.keys()),
            key="study_progression_season",
        )
    with p2:
        prog_pos_options = ["Tous"] + sorted(
            [p for p in prog_df.get("position_group", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if p]
        )
        selected_prog_pos = st.selectbox("Poste (progression)", prog_pos_options, key="study_progression_position")

    selected_prog_season = prog_label_map.get(selected_prog_label) if selected_prog_label else None
    prog_scope = prog_df.copy()
    if selected_prog_season is not None:
        prog_scope = prog_scope[prog_scope["season_start"].astype(int) == selected_prog_season]
    if selected_prog_pos != "Tous":
        prog_scope = prog_scope[prog_scope["position_group"] == selected_prog_pos]

    ptop, pdrop = st.columns(2)
    with ptop:
        st.caption("Top progressions")
        if prog_scope.empty:
            st.info("Aucune donnee pour ce filtre.")
        else:
            top_prog = prog_scope.sort_values(["progress_score", "delta_ga_p90"], ascending=[False, False]).head(10).copy()
            top_prog["Joueur"] = top_prog["player_name"].astype(str)
            top_prog = add_podium_icons_generic(top_prog, "Joueur")
            top_prog["Score"] = top_prog["progress_score"].round(3)
            top_prog["Î” GA/90"] = top_prog["delta_ga_p90"].round(2)
            top_prog["Î” Passes/90"] = top_prog["delta_passes_p90"].round(2)
            st.dataframe(
                top_prog[["Joueur", "team_name", "position_group", "minutes_total_prev", "minutes_total", "Î” GA/90", "Î” Passes/90", "Score"]]
                .rename(columns={
                    "team_name": "Club",
                    "position_group": "Poste",
                    "minutes_total_prev": "Min N-1",
                    "minutes_total": "Min N",
                }),
                use_container_width=True,
                hide_index=True,
            )
            bar_df = top_prog.copy()
            bar_df["JoueurScore"] = bar_df["Joueur"]
            render_sorted_bar_chart(bar_df, "JoueurScore", "progress_score", descending=True, bar_color=VISUAL_COLORS["attack"])

    with pdrop:
        st.caption("Top regressions")
        if prog_scope.empty:
            st.info("Aucune donnee pour ce filtre.")
        else:
            worst_prog = prog_scope.sort_values(["progress_score", "delta_ga_p90"], ascending=[True, True]).head(10).copy()
            worst_prog["Joueur"] = worst_prog["player_name"].astype(str)
            worst_prog["Score"] = worst_prog["progress_score"].round(3)
            worst_prog["Î” GA/90"] = worst_prog["delta_ga_p90"].round(2)
            worst_prog["Î” Passes/90"] = worst_prog["delta_passes_p90"].round(2)
            st.dataframe(
                worst_prog[["Joueur", "team_name", "position_group", "minutes_total_prev", "minutes_total", "Î” GA/90", "Î” Passes/90", "Score"]]
                .rename(columns={
                    "team_name": "Club",
                    "position_group": "Poste",
                    "minutes_total_prev": "Min N-1",
                    "minutes_total": "Min N",
                }),
                use_container_width=True,
                hide_index=True,
            )
            render_sorted_bar_chart(
                worst_prog.assign(JoueurScore=worst_prog["Joueur"]),
                "JoueurScore",
                "progress_score",
                descending=False,
                signed_colors=True,
            )



