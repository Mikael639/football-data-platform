import os
import unicodedata

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

from data.dashboard_data import (
    build_local_league_table,
    build_team_match_view,
    compute_team_kpis,
    current_season_label,
    fetch_laliga_teams_live,
    fetch_live_team_squad,
    upsert_players_to_db,
)
from ui.charts import render_ppm_chart, render_result_distribution_chart, render_sorted_bar_chart
from ui.display import add_podium_icons, render_form_timeline, render_quality_badges, style_ligue_table
from ui.styles import VISUAL_COLORS


def render_team_tab(
    *,
    selected_team_name,
    season_start_year,
    df_matches,
    selected_team_id,
    teams_df,
    render_team_hero,
):

    st.header("Indicateurs equipe")
    render_team_hero(selected_team_name, current_season_label(season_start_year))
    st.caption(f"KPI filtres sur la saison {current_season_label(season_start_year)} (base locale)")

    if df_matches.empty:
        st.warning(
            "Aucun match trouve dans la base locale pour la saison en cours. "
            "Relance le pipeline API pour charger toute la Liga."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)

        if selected_team_id is None:
            total_matches = df_matches["match_id"].nunique()
            played_matches = df_matches.dropna(subset=["home_score", "away_score"])["match_id"].nunique()
            upcoming_matches = total_matches - played_matches

            col1.metric("Matchs (saison)", int(total_matches))
            col2.metric("Joues", int(played_matches))
            col3.metric("A venir", int(upcoming_matches))
            col4.metric("Clubs en base", int(teams_df["team_id"].nunique()))
        else:
            played, wins, draws, losses, gf, ga, points = compute_team_kpis(df_matches, selected_team_id)
            col1.metric("Joues", played)
            col2.metric("Points", points)
            col3.metric("GF / GA", f"{gf} / {ga}")
            col4.metric("V - N - D", f"{wins}-{draws}-{losses}")

        st.subheader("Derniers matchs")
        recent_display = df_matches.dropna(subset=["home_score", "away_score"]).head(15).copy()
        st.dataframe(
            recent_display.rename(
                columns={
                    "date_id": "date",
                    "home_team": "domicile",
                    "away_team": "exterieur",
                    "home_score": "score_dom",
                    "away_score": "score_ext",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Matchs a venir")
        today = pd.Timestamp.now().normalize()
        df_upcoming = df_matches[
            (df_matches["date_dt"] >= today)
            | (df_matches["home_score"].isna())
            | (df_matches["away_score"].isna())
        ].sort_values("date_dt", ascending=True)
        upcoming_display = df_upcoming.head(15).copy()
        st.dataframe(
            upcoming_display.rename(
                columns={
                    "date_id": "date",
                    "home_team": "domicile",
                    "away_team": "exterieur",
                    "home_score": "score_dom",
                    "away_score": "score_ext",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Visualisations (KPI)")
        if selected_team_id is None:
            league_local = build_local_league_table(df_matches)
            if league_local.empty:
                st.info("Aucun match joue pour generer les visuels de ligue.")
            else:
                v1, v2 = st.columns(2)
                with v1:
                    st.caption("Top 10 clubs par points (base locale, saison courante)")
                    top_points = league_local[["Team", "Pts"]].head(10).sort_values("Pts", ascending=False)
                    render_sorted_bar_chart(
                        top_points, "Team", "Pts", descending=True, bar_color=VISUAL_COLORS["points"]
                    )
                with v2:
                    goals_per_match = (
                        df_matches.dropna(subset=["home_score", "away_score"])
                        .assign(total_goals=lambda d: d["home_score"] + d["away_score"])
                        ["total_goals"]
                        .value_counts()
                        .sort_values(ascending=False)
                    )
                    st.caption("Totaux de buts les plus frequents (ordre decroissant)")
                    gpm_df = goals_per_match.reset_index()
                    gpm_df.columns = ["ButsTotal", "NbMatchs"]
                    gpm_df["ButsTotal"] = gpm_df["ButsTotal"].astype(str)
                    render_sorted_bar_chart(
                        gpm_df, "ButsTotal", "NbMatchs", descending=True, bar_color=VISUAL_COLORS["violet"]
                    )
        else:
            team_match_view = build_team_match_view(df_matches, selected_team_id)
            if team_match_view.empty:
                st.info("Aucun match joue pour ce club.")
            else:
                v1, v2 = st.columns(2)
                with v1:
                    st.caption("Repartition des resultats (V/N/D)")
                    result_counts = (
                        team_match_view["Result"]
                        .value_counts()
                        .reindex(["W", "D", "L"], fill_value=0)
                    )
                    result_df = result_counts.sort_values(ascending=False).reset_index()
                    result_df.columns = ["Result", "Count"]
                    render_result_distribution_chart(result_df, height=220)
                with v2:
                    st.caption("Points cumules sur la saison")
                    st.line_chart(team_match_view.set_index("date_dt")["CumulativePoints"])

                st.caption("Buts marques vs encaisses par match")
                chart_df = team_match_view[["date_dt", "GF", "GA"]].copy()
                chart_df["match_date"] = chart_df["date_dt"].dt.strftime("%Y-%m-%d")
                st.bar_chart(chart_df.set_index("match_date")[["GF", "GA"]].tail(15))

                v3, v4 = st.columns(2)
                with v3:
                    st.caption("Domicile vs exterieur - GF/GA + points par match")
                    home_away = (
                        team_match_view.groupby("venue")
                        .agg(
                            Matches=("match_id", "count"),
                            Points=("Points", "sum"),
                            GF=("GF", "sum"),
                            GA=("GA", "sum"),
                        )
                        .reindex(["Domicile", "Exterieur"])
                        .fillna(0)
                    )
                    home_away["PPM"] = (home_away["Points"] / home_away["Matches"].replace(0, pd.NA)).fillna(0)
                    home_away["GF_per_match"] = (home_away["GF"] / home_away["Matches"].replace(0, pd.NA)).fillna(0)
                    home_away["GA_per_match"] = (home_away["GA"] / home_away["Matches"].replace(0, pd.NA)).fillna(0)
                    home_away = home_away.sort_values(["PPM", "Points"], ascending=False)

                    st.caption("GF vs GA (par lieu)")
                    st.bar_chart(home_away[["GF", "GA"]])
                    st.caption("Points par match (PPM)")
                    ppm_df = home_away.reset_index()[["venue", "PPM"]]
                    render_ppm_chart(ppm_df, height=220)
                    st.dataframe(
                        home_away[["Matches", "Points", "PPM", "GF", "GA", "GF_per_match", "GA_per_match"]]
                        .round(2),
                        use_container_width=True,
                    )
                with v4:
                    st.caption("Forme sur les 10 derniers matchs (timeline)")
                    last10 = team_match_view.sort_values(["date_dt", "match_id"]).tail(10).copy()
                    last10["date"] = last10["date_dt"].dt.strftime("%Y-%m-%d")
                    st.metric("Points (10 derniers)", int(last10["Points"].sum()))
                    render_form_timeline(last10)
                    form_symbols = {"W": "[W]", "D": "[D]", "L": "[L]"}
                    st.write(" | ".join(form_symbols.get(r, r) for r in last10["Result"].tolist()))
                    st.dataframe(
                        last10[["date", "venue", "opponent", "GF", "GA", "Result", "Points"]],
                        use_container_width=True,
                        hide_index=True,
                    )



def render_ligue_tab(
    *,
    season_start_year,
    league_local_all_season,
    selected_team_name,
):

    st.header("Ligue - Classement et analyse (Saison en cours)")

    token = os.getenv("FOOTBALL_DATA_TOKEN")
    base_url = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
    competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")

    st.caption(f"Saison utilisee: {current_season_label(season_start_year)} (auto)")

    local_cols = ["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]
    if not token:
        st.info("FOOTBALL_DATA_TOKEN est absent dans le conteneur dashboard.")
        if not league_local_all_season.empty:
            st.caption("Fallback: classement calcule depuis la base locale (saison courante)")
            st.dataframe(
                style_ligue_table(league_local_all_season[local_cols].rename(columns={"Team": "Equipe"})),
                use_container_width=True,
                hide_index=True,
            )
    else:
        api_error_message = None
        df_table = None
        try:
            response = requests.get(
                f"{base_url}/competitions/{competition_code}/standings",
                headers={"X-Auth-Token": token},
                params={"season": season_start_year},
                timeout=30,
            )
        except requests.RequestException as exc:
            api_error_message = f"Echec de la requete classement: {exc}"
        else:
            if response.status_code != 200:
                api_error_message = f"Classement indisponible (status={response.status_code})."
            else:
                data = response.json()
                total_table = None
                for standing in data.get("standings", []):
                    if standing.get("type") == "TOTAL":
                        total_table = standing.get("table", [])
                        break

                if not total_table:
                    api_error_message = "Aucun tableau TOTAL trouve."
                else:
                    df_table = pd.DataFrame(
                        [
                            {
                                "Pos": row.get("position"),
                                "Team": row.get("team", {}).get("name"),
                                "Pts": row.get("points"),
                                "P": row.get("playedGames"),
                                "W": row.get("won"),
                                "D": row.get("draw"),
                                "L": row.get("lost"),
                                "GF": row.get("goalsFor"),
                                "GA": row.get("goalsAgainst"),
                                "GD": row.get("goalDifference"),
                            }
                            for row in total_table
                        ]
                    )

        standings_source = None
        standings_df = None
        if df_table is not None and not df_table.empty:
            standings_source = "API live football-data.org"
            standings_df = df_table.copy()
        elif not league_local_all_season.empty:
            standings_source = "Base locale (fallback)"
            standings_df = league_local_all_season.copy()
            if "Pos" not in standings_df.columns:
                standings_df.insert(0, "Pos", range(1, len(standings_df) + 1))
            if api_error_message:
                st.caption(f"API indisponible: {api_error_message}")
        else:
            standings_df = None
            if api_error_message:
                st.warning(api_error_message)

        if standings_df is None or standings_df.empty:
            st.info("Aucun classement disponible (API et fallback local indisponibles).")
        else:
            st.caption(f"Source des donnees: {standings_source}")

            leader = standings_df.iloc[0]
            best_attack = standings_df.sort_values(["GF", "Team"], ascending=[False, True]).iloc[0]
            best_defense = standings_df.sort_values(["GA", "Team"], ascending=[True, True]).iloc[0]
            best_gd = standings_df.sort_values(["GD", "Team"], ascending=[False, True]).iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Leader", f"★ {leader['Team']} ({int(leader['Pts'])} pts)")
            c2.metric("Meilleure attaque", f"{best_attack['Team']} ({int(best_attack['GF'])} GF)")
            c3.metric("Meilleure defense", f"{best_defense['Team']} ({int(best_defense['GA'])} GA)")
            c4.metric("Meilleur diff.", f"{best_gd['Team']} ({int(best_gd['GD'])})")

            if selected_team_name != "Tous les clubs":
                selected_row = standings_df[standings_df["Team"].str.contains(selected_team_name, case=False, na=False)]
                if not selected_row.empty:
                    sr = selected_row.iloc[0]
                    gap = int(leader["Pts"]) - int(sr["Pts"])
                    st.info(
                        f"{selected_team_name} | Pos {int(sr['Pos'])} | {int(sr['Pts'])} pts | "
                        f"Ecart avec le leader: {gap} pts"
                    )

            v1, v2 = st.columns(2)
            with v1:
                st.caption("Top 10 par points")
                top10_pts = standings_df.sort_values(["Pts", "GD", "GF"], ascending=False).head(10)
                render_sorted_bar_chart(
                    top10_pts, "Team", "Pts", descending=True, bar_color=VISUAL_COLORS["points"]
                )
            with v2:
                st.caption("Difference de buts (Top 10)")
                top10_gd = standings_df.sort_values(["GD", "Pts", "GF"], ascending=False).head(10)
                render_sorted_bar_chart(top10_gd, "Team", "GD", descending=True, signed_colors=True)

            z1, z2 = st.columns(2)
            with z1:
                st.subheader("Top 4")
                st.dataframe(
                    style_ligue_table(
                        add_podium_icons(
                            standings_df.head(4)[["Pos", "Team", "Pts", "P", "W", "D", "L", "GF", "GA", "GD"]]
                            .rename(columns={"Team": "Equipe"})
                        )
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with z2:
                st.subheader("Zone de relegation (3 derniers)")
                st.dataframe(
                    style_ligue_table(
                        standings_df.tail(3)[["Pos", "Team", "Pts", "P", "W", "D", "L", "GF", "GA", "GD"]]
                        .rename(columns={"Team": "Equipe"})
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.expander("Classement detaille", expanded=False).dataframe(
                style_ligue_table(
                    add_podium_icons(
                        standings_df[["Pos", "Team", "Pts", "P", "W", "D", "L", "GF", "GA", "GD"]]
                        .rename(columns={"Team": "Equipe"})
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )



def render_clubs_tab(
    *,
    league_local_all_season,
    club_summary,
    engine,
):

    st.header("Clubs (LaLiga)")
    st.caption(
        "Clique sur un club pour ouvrir la page Joueurs (Streamlit multipage) "
        "avec l'effectif du club."
    )
    st.page_link("pages/1_Joueurs.py", label="Ouvrir la page Joueurs")

    if not league_local_all_season.empty:
        st.subheader("Top clubs (KPI locaux - saison courante)")
        top_cols = ["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]
        st.dataframe(
            style_ligue_table(
                add_podium_icons(league_local_all_season[top_cols].rename(columns={"Team": "Equipe"}))
            ),
            use_container_width=True,
            hide_index=True,
        )
    quality_counts = club_summary["data_quality_status"].value_counts().to_dict()
    quality_counts["TOTAL"] = int(len(club_summary))
    st.caption("Qualite de la couverture locale (clubs / joueurs / calendrier)")
    render_quality_badges(quality_counts)

    with st.expander("Maintenance data - completer les effectifs manquants", expanded=False):
        missing_local = club_summary[
            (club_summary["data_quality_status"] == "INCOMPLET") & (club_summary["matches_in_scope"] > 0)
        ][["team_id", "team_name"]]
        st.write(f"Clubs incomplets detectes: {len(missing_local)}")
        if not missing_local.empty:
            st.dataframe(missing_local, use_container_width=True, hide_index=True)

        if st.button("Completer les effectifs manquants (API live vers DB locale)", key="backfill_missing_squads_btn"):
            updated_clubs = 0
            inserted_players_total = 0
            errors = []

            with st.spinner("Mise a jour des effectifs en cours..."):
                for _, club in missing_local.iterrows():
                    live_df, live_err = fetch_live_team_squad(int(club["team_id"]))
                    if live_err or live_df is None or live_df.empty:
                        errors.append(f"{club['team_name']}: {live_err or 'aucun effectif renvoye'}")
                        continue
                    try:
                        inserted_players_total += upsert_players_to_db(engine, live_df)
                        updated_clubs += 1
                    except Exception as exc:
                        errors.append(f"{club['team_name']}: erreur base ({exc})")

            if updated_clubs > 0:
                st.success(
                    f"Mise a jour terminee: {updated_clubs} club(s) mis a jour, {inserted_players_total} joueur(s) ajoutes/mis a jour."
                )
                fetch_live_team_squad.clear()
                fetch_laliga_teams_live.clear()
                st.rerun()
            elif errors:
                st.warning("Mise a jour terminee sans modification.")

            if errors:
                st.expander("Details des erreurs de mise a jour", expanded=False).write("\n".join(errors))

    st.expander("Controle des donnees (couverture locale)", expanded=False).dataframe(
        club_summary.rename(
            columns={
                "team_name": "club",
                "data_quality_status": "qualite",
                "players_count": "joueurs_locaux",
                "matches_in_scope": "matchs_saison",
            }
        )[["club", "qualite", "joueurs_locaux", "matchs_saison"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Ouvrir un club")
    cols_per_row = 4
    for start in range(0, len(club_summary), cols_per_row):
        row_slice = club_summary.iloc[start : start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (_, club) in zip(cols, row_slice.iterrows()):
            with col:
                if st.button(
                    f"{club['team_name']}",
                    key=f"club_btn_{int(club['team_id'])}",
                    use_container_width=True,
                ):
                    st.session_state["players_selected_team_name"] = club["team_name"]
                    st.session_state["players_selected_team_id"] = int(club["team_id"])
                    st.switch_page("pages/1_Joueurs.py")


def render_player_details_tab() -> None:
    st.header("Details Joueurs")
    st.caption(
        "Exploitation des donnees joueurs: filtres dynamiques, comparaison inter-saisons et profils objectifs."
    )
    st.page_link("pages/1_Joueurs.py", label="Ouvrir la page Joueurs detaillee")

    supabase_db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("STUDY_SUPABASE_DB_URL")
    if not supabase_db_url:
        st.info("SUPABASE_DB_URL non configuree dans le conteneur dashboard.")
        return

    def _norm(value: object) -> str:
        if value is None:
            return ""
        s = str(value).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return s

    def _season_label(start_year: int) -> str:
        return f"{start_year}/{str(start_year + 1)[-2:]}"

    try:
        engine = create_engine(supabase_db_url, pool_pre_ping=True)
        season_df = pd.read_sql(text("SELECT * FROM public.study_fbref_player_season"), engine)
        match_df = pd.read_sql(text("SELECT * FROM public.study_fbref_player_match"), engine)
        reg_df = pd.read_sql(text("SELECT * FROM public.study_fbref_regularity"), engine)
        prog_df = pd.read_sql(text("SELECT * FROM public.study_fbref_progression"), engine)
    except Exception:
        st.info("Impossible de lire les donnees detail depuis Supabase.")
        return

    if season_df.empty and match_df.empty and reg_df.empty and prog_df.empty:
        st.info("Aucune donnee joueurs disponible dans Supabase.")
        return

    global_players = int(season_df["player_id"].nunique()) if (not season_df.empty and "player_id" in season_df.columns) else 0
    match_players = int(match_df["player_id"].nunique()) if (not match_df.empty and "player_id" in match_df.columns) else 0
    progression_rows = int(len(prog_df))
    regularity_rows = int(len(reg_df))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joueurs globaux (LaLiga)", global_players)
    c2.metric("Joueurs suivis (match-by-match)", match_players)
    c3.metric("Lignes progression", progression_rows)
    c4.metric("Lignes regularite", regularity_rows)

    st.info(
        "Filtrage dynamique par poste, club ou joueur. "
        "Comparaison inter-saisons rapide. Detection de profils reguliers, emergents ou sous-cotes."
    )

    global_tab, tracked_tab = st.tabs(["Vue globale LaLiga", "Joueur suivi (match-by-match)"])

    with global_tab:
        if season_df.empty:
            st.info("Aucune donnee globale joueurs-saison.")
        else:
            base = season_df.copy()
            for col in [
                "season_start",
                "minutes_total",
                "goals_total",
                "assists_total",
                "ga_total",
                "ga_p90",
                "passes_p90",
                "matches_played",
            ]:
                if col in base.columns:
                    base[col] = pd.to_numeric(base[col], errors="coerce")

            seasons = sorted(base["season_start"].dropna().astype(int).unique().tolist(), reverse=True)
            season_options = ["? Tous ?"] + [_season_label(s) for s in seasons]
            season_map = {opt: None for opt in season_options}
            for s in seasons:
                season_map[_season_label(s)] = s

            clubs = sorted(base.get("team_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
            positions = sorted(base.get("position_group", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                selected_season_label = st.selectbox("Saison", season_options, key="details_global_season")
            with f2:
                selected_club = st.selectbox("Club", ["? Tous ?"] + clubs, key="details_global_club")
            with f3:
                selected_pos = st.selectbox("Poste", ["? Tous ?"] + positions, key="details_global_pos")
            with f4:
                min_minutes = st.slider("Minutes min", min_value=0, max_value=2500, value=600, step=50, key="details_global_min")

            scope = base.copy()
            selected_season = season_map[selected_season_label]
            if selected_season is not None:
                scope = scope[scope["season_start"].astype(int) == int(selected_season)]
            if selected_club != "? Tous ?":
                scope = scope[scope["team_name"].astype(str) == selected_club]
            if selected_pos != "? Tous ?":
                scope = scope[scope["position_group"].astype(str) == selected_pos]
            scope = scope[pd.to_numeric(scope["minutes_total"], errors="coerce").fillna(0) >= min_minutes].copy()

            if scope.empty:
                st.info("Aucune ligne apres filtres.")
            else:
                current_season_for_profiles = int(selected_season) if selected_season is not None else int(scope["season_start"].dropna().astype(int).max())

                reg_scope = reg_df.copy()
                if not reg_scope.empty:
                    reg_scope = reg_scope[reg_scope["season_start"].astype(int) == current_season_for_profiles]
                    if selected_pos != "? Tous ?":
                        reg_scope = reg_scope[reg_scope["position_group"].astype(str) == selected_pos]
                    if selected_club != "? Tous ?":
                        reg_scope = reg_scope[reg_scope["team_name"].astype(str) == selected_club]

                prog_scope = prog_df.copy()
                if not prog_scope.empty:
                    if selected_pos != "? Tous ?":
                        prog_scope = prog_scope[prog_scope["position_group"].astype(str) == selected_pos]
                    if selected_club != "? Tous ?":
                        prog_scope = prog_scope[prog_scope["team_name"].astype(str) == selected_club]

                regular_name, regular_note = "N/A", "Aucun candidat"
                if not reg_scope.empty:
                    r0 = reg_scope.sort_values(["regularity_score", "minutes_total"], ascending=[False, False]).head(1).iloc[0]
                    regular_name = str(r0.get("player_name", "N/A"))
                    regular_note = f"Score {round(float(r0.get('regularity_score', 0.0)), 3)}"

                emergent_name, emergent_note = "N/A", "Aucun candidat"
                if not prog_scope.empty:
                    p0 = prog_scope.sort_values(["progress_score", "delta_ga_p90"], ascending=[False, False]).head(1).iloc[0]
                    emergent_name = str(p0.get("player_name", "N/A"))
                    emergent_note = f"Delta GA/90 {round(float(p0.get('delta_ga_p90', 0.0)), 2)}"

                under_name, under_note = "N/A", "Aucun candidat"
                under_pool = scope.copy()
                under_pool["minutes_total"] = pd.to_numeric(under_pool["minutes_total"], errors="coerce").fillna(0)
                under_pool["ga_p90"] = pd.to_numeric(under_pool.get("ga_p90"), errors="coerce").fillna(0)
                under_pool = under_pool[(under_pool["minutes_total"] >= min_minutes) & (under_pool["minutes_total"] <= (min_minutes + 500))]
                if not under_pool.empty:
                    u0 = under_pool.sort_values(["ga_p90", "minutes_total"], ascending=[False, True]).head(1).iloc[0]
                    under_name = str(u0.get("player_name", "N/A"))
                    under_note = f"GA/90 {round(float(u0.get('ga_p90', 0.0)), 2)} avec {int(u0.get('minutes_total', 0))} min"

                p1, p2, p3 = st.columns(3)
                p1.metric("Profil regulier", regular_name, regular_note)
                p2.metric("Profil emergent", emergent_name, emergent_note)
                p3.metric("Profil sous-cote", under_name, under_note)

                agg = scope.groupby("player_name", as_index=False).agg(
                    goals_total=("goals_total", "sum"),
                    assists_total=("assists_total", "sum"),
                    ga_total=("ga_total", "sum"),
                    minutes_total=("minutes_total", "sum"),
                )
                meta = (
                    scope.sort_values("minutes_total", ascending=False)
                    .drop_duplicates(subset=["player_name"], keep="first")[["player_name", "team_name", "position_group"]]
                )
                leaders = agg.merge(meta, on="player_name", how="left")

                t1, t2 = st.columns(2)
                with t1:
                    st.caption("Top buteurs")
                    st.dataframe(
                        leaders.sort_values(["goals_total", "minutes_total"], ascending=[False, False]).head(12)[
                            ["player_name", "team_name", "position_group", "goals_total", "minutes_total"]
                        ].rename(
                            columns={
                                "player_name": "Joueur",
                                "team_name": "Club",
                                "position_group": "Poste",
                                "goals_total": "Buts",
                                "minutes_total": "Min",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                with t2:
                    st.caption("Top passeurs")
                    st.dataframe(
                        leaders.sort_values(["assists_total", "minutes_total"], ascending=[False, False]).head(12)[
                            ["player_name", "team_name", "position_group", "assists_total", "minutes_total"]
                        ].rename(
                            columns={
                                "player_name": "Joueur",
                                "team_name": "Club",
                                "position_group": "Poste",
                                "assists_total": "Passes D",
                                "minutes_total": "Min",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                player_choices = ["? Tous ?"] + sorted(scope["player_name"].dropna().astype(str).unique().tolist())
                selected_player = st.selectbox("Joueur (comparaison inter-saisons)", player_choices, key="details_global_player")

                if selected_player != "? Tous ?":
                    player_hist = base[base["player_name"].astype(str) == selected_player].copy()
                    if selected_club != "? Tous ?":
                        player_hist = player_hist[player_hist["team_name"].astype(str) == selected_club]
                    if selected_pos != "? Tous ?":
                        player_hist = player_hist[player_hist["position_group"].astype(str) == selected_pos]
                    player_hist = player_hist.sort_values("season_start")

                    if not player_hist.empty:
                        player_hist["Saison"] = player_hist["season_start"].astype(int).map(_season_label)

                        chart_base = player_hist[["Saison", "goals_total", "assists_total", "ga_total", "minutes_total"]].copy()
                        for metric in ["goals_total", "assists_total", "ga_total", "minutes_total"]:
                            max_val = float(chart_base[metric].max()) if chart_base[metric].notna().any() else 0.0
                            chart_base[metric] = chart_base[metric].fillna(0) / max_val if max_val > 0 else 0.0
                        long_comp = chart_base.melt("Saison", var_name="Metrique", value_name="Indice")

                        st.caption("Comparaison inter-saisons (indice normalise)")
                        comp_chart = (
                            alt.Chart(long_comp)
                            .mark_bar()
                            .encode(
                                x=alt.X("Metrique:N", sort=["goals_total", "assists_total", "ga_total", "minutes_total"]),
                                y=alt.Y("Indice:Q", stack="zero", title="Indice"),
                                color=alt.Color("Saison:N"),
                                tooltip=["Saison", "Metrique", alt.Tooltip("Indice:Q", format=".2f")],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(comp_chart, use_container_width=True)

                        profile_row = player_hist[player_hist["season_start"].astype(int) == int(selected_season)].head(1) if selected_season is not None else pd.DataFrame()
                        if profile_row.empty:
                            profile_row = player_hist.sort_values("season_start", ascending=False).head(1)

                        if not profile_row.empty:
                            pr = profile_row.iloc[0]
                            season_ref = int(pr.get("season_start"))
                            pos_ref = str(pr.get("position_group", ""))
                            ref_scope = base[base["season_start"].astype(int) == season_ref].copy()
                            if pos_ref:
                                ref_scope = ref_scope[ref_scope["position_group"].astype(str) == pos_ref]

                            metrics = {
                                "Matchs": "matches_played",
                                "Minutes": "minutes_total",
                                "Buts": "goals_total",
                                "Passes D": "assists_total",
                                "G+A/90": "ga_p90",
                            }
                            rows = []
                            for label, col in metrics.items():
                                player_val = float(pd.to_numeric(pd.Series([pr.get(col)]), errors="coerce").fillna(0).iloc[0])
                                avg_val = float(pd.to_numeric(ref_scope.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not ref_scope.empty else 0.0
                                denom = max(player_val, avg_val, 1e-6)
                                rows.append({"Metrique": label, "Serie": "Joueur", "Indice": player_val / denom})
                                rows.append({"Metrique": label, "Serie": "Moyenne poste", "Indice": avg_val / denom})
                            prof_df = pd.DataFrame(rows)

                            st.caption("Profil visuel (joueur vs moyenne poste)")
                            stacked_chart = (
                                alt.Chart(prof_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Metrique:N"),
                                    y=alt.Y("Indice:Q", stack="zero"),
                                    color=alt.Color("Serie:N", scale=alt.Scale(range=["#2A9D8F", "#457B9D"])),
                                    tooltip=["Serie", "Metrique", alt.Tooltip("Indice:Q", format=".2f")],
                                )
                                .properties(height=280)
                            )
                            st.altair_chart(stacked_chart, use_container_width=True)

                            radar_metrics = [r["Metrique"] for r in rows if r["Serie"] == "Joueur"]
                            radar_data = []
                            n = len(radar_metrics)
                            for series in ["Joueur", "Moyenne poste"]:
                                sub = prof_df[prof_df["Serie"] == series].set_index("Metrique")["Indice"].to_dict()
                                first_point = None
                                for i, m in enumerate(radar_metrics):
                                    ang = (2 * np.pi * i) / max(n, 1)
                                    val = float(sub.get(m, 0.0))
                                    point = {"Serie": series, "Metrique": m, "x": val * np.cos(ang), "y": val * np.sin(ang), "Indice": val}
                                    radar_data.append(point)
                                    if i == 0:
                                        first_point = point
                                if first_point is not None:
                                    radar_data.append(first_point.copy())

                            labels_data = []
                            for i, m in enumerate(radar_metrics):
                                ang = (2 * np.pi * i) / max(n, 1)
                                labels_data.append({"Metrique": m, "x": 1.2 * np.cos(ang), "y": 1.2 * np.sin(ang)})

                            radar_df = pd.DataFrame(radar_data)
                            labels_df = pd.DataFrame(labels_data)
                            radar_chart = alt.layer(
                                alt.Chart(radar_df)
                                .mark_area(opacity=0.12)
                                .encode(
                                    x=alt.X("x:Q", axis=None),
                                    y=alt.Y("y:Q", axis=None),
                                    color=alt.Color("Serie:N", scale=alt.Scale(range=["#2A9D8F", "#457B9D"])),
                                    detail="Serie:N",
                                ),
                                alt.Chart(radar_df)
                                .mark_line(point=True)
                                .encode(
                                    x="x:Q",
                                    y="y:Q",
                                    color=alt.Color("Serie:N", scale=alt.Scale(range=["#2A9D8F", "#457B9D"])),
                                    detail="Serie:N",
                                    tooltip=["Serie", "Metrique", alt.Tooltip("Indice:Q", format=".2f")],
                                ),
                                alt.Chart(labels_df).mark_text(dy=-2, fontSize=11).encode(x="x:Q", y="y:Q", text="Metrique:N"),
                            ).properties(height=360)
                            st.altair_chart(radar_chart, use_container_width=True)

    with tracked_tab:
        if match_df.empty:
            st.info("Aucune donnee match-by-match disponible.")
        else:
            tracked = match_df.copy()
            tracked["season_start"] = pd.to_numeric(tracked.get("season_start"), errors="coerce")
            tracked["date_id"] = pd.to_datetime(tracked.get("date_id"), errors="coerce")
            tracked["minutes"] = pd.to_numeric(tracked.get("minutes"), errors="coerce").fillna(0)
            tracked["goals"] = pd.to_numeric(tracked.get("goals"), errors="coerce").fillna(0)
            tracked["assists"] = pd.to_numeric(tracked.get("assists"), errors="coerce").fillna(0)
            tracked["shots"] = pd.to_numeric(tracked.get("shots"), errors="coerce").fillna(0)
            tracked["passes"] = pd.to_numeric(tracked.get("passes"), errors="coerce").fillna(0)

            players_df = (
                tracked.sort_values("date_id", ascending=False)
                .drop_duplicates(subset=["player_id"], keep="first")[["player_id", "player_name", "team_name", "position_group"]]
                .sort_values("player_name")
            )
            players_df["label"] = (
                players_df["player_name"].astype(str)
                + " - "
                + players_df["team_name"].fillna("").astype(str)
                + " ("
                + players_df["position_group"].fillna("").astype(str)
                + ")"
            )
            label_to_pid = dict(zip(players_df["label"], players_df["player_id"]))

            seasons = sorted(tracked["season_start"].dropna().astype(int).unique().tolist(), reverse=True)
            s_options = ["? Tous ?"] + [_season_label(int(s)) for s in seasons]
            s_map = {opt: None for opt in s_options}
            for s in seasons:
                s_map[_season_label(int(s))] = int(s)

            d1, d2 = st.columns([0.35, 0.65])
            with d1:
                selected_tracked_season_label = st.selectbox("Saison (detail)", s_options, key="details_tracked_season")
            with d2:
                selected_player_label = st.selectbox("Joueur suivi", players_df["label"].tolist(), key="details_tracked_player")

            selected_pid = label_to_pid[selected_player_label]
            selected_season = s_map[selected_tracked_season_label]

            p_scope = tracked[tracked["player_id"] == selected_pid].copy()
            if selected_season is not None:
                p_scope = p_scope[p_scope["season_start"].astype(int) == int(selected_season)].copy()
            p_scope = p_scope.sort_values("date_id")

            if p_scope.empty:
                st.info("Aucune ligne pour ce joueur et ce filtre saison.")
            else:
                total_min = int(p_scope["minutes"].sum())
                total_match = int(len(p_scope))
                total_goals = int(p_scope["goals"].sum())
                total_assists = int(p_scope["assists"].sum())
                ga = total_goals + total_assists
                ga_p90 = round((ga * 90 / total_min), 2) if total_min > 0 else 0.0

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Matchs", total_match)
                k2.metric("Minutes", total_min)
                k3.metric("Buts", total_goals)
                k4.metric("Passes D", total_assists)
                k5.metric("G+A/90", ga_p90)

                line_df = p_scope[["date_id", "goals", "assists", "shots"]].copy()
                line_df["Date"] = line_df["date_id"].dt.strftime("%Y-%m-%d")
                st.caption("Evolution match par match")
                st.line_chart(line_df.set_index("Date")[["goals", "assists", "shots"]])

                st.caption("Derniers matchs")
                p_scope["date"] = p_scope["date_id"].dt.strftime("%Y-%m-%d")
                st.dataframe(
                    p_scope[[c for c in ["date", "team_name", "position_group", "minutes", "goals", "assists", "shots", "passes"] if c in p_scope.columns]].rename(
                        columns={
                            "date": "Date",
                            "team_name": "Club",
                            "position_group": "Poste",
                            "minutes": "Min",
                            "goals": "Buts",
                            "assists": "Passes D",
                            "shots": "Tirs",
                            "passes": "Passes",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

