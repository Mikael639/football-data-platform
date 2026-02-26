import os

import pandas as pd
import requests
import streamlit as st

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

