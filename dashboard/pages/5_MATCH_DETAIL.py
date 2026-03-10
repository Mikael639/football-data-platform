import pandas as pd
import streamlit as st

from data.dashboard_data import (
    build_perspective_table,
    get_home_away_split,
    get_match_detail,
    get_match_head_to_head,
    get_match_player_stats,
    get_match_picker,
    get_match_ranking_context,
    get_matches,
    get_team_xg_proxy,
)
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_result_strip, render_section_heading
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="MATCH DETAIL - Football Data Platform", layout="wide")


def _resolve_selected_match_id() -> int | None:
    selected = st.session_state.get("selected_match_id")
    if selected is not None:
        try:
            return int(selected)
        except (TypeError, ValueError):
            pass

    query_match_id = st.query_params.get("match_id")
    if query_match_id:
        try:
            match_id = int(str(query_match_id))
            st.session_state["selected_match_id"] = match_id
            return match_id
        except ValueError:
            pass
    return None


def _format_kickoff(value: object, fallback: object) -> str:
    kickoff = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.notna(kickoff):
        return kickoff.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M")
    fallback_ts = pd.to_datetime(fallback, errors="coerce")
    if pd.notna(fallback_ts):
        return fallback_ts.strftime("%Y-%m-%d")
    return "Unknown"


def _format_match_label(row: pd.Series) -> str:
    kickoff = _format_kickoff(row.get("match_ts"), None)
    status = str(row.get("status") or "UNKNOWN")
    competition = str(row.get("competition_name") or "Championnat")
    season = str(row.get("season") or "-")
    return f"[{competition} | {season}] {row['home_team']} vs {row['away_team']} ({kickoff}, {status})"


def _build_detail_label(detail: dict[str, object]) -> str:
    kickoff = _format_kickoff(detail.get("kickoff_utc"), detail.get("match_date"))
    competition = str(detail.get("competition_name") or "Championnat")
    season = str(detail.get("season") or "-")
    status = str(detail.get("status") or "UNKNOWN")
    return f"[{competition} | {season}] {detail['home_team']} vs {detail['away_team']} ({kickoff}, {status})"


def _render_picker(selected_match_id: int | None) -> int | None:
    picker = get_match_picker(limit=3000)
    if picker.empty:
        st.info("Aucun match disponible en base.")
        return None

    options = picker.copy()
    if selected_match_id is not None and int(selected_match_id) not in options["match_id"].astype(int).tolist():
        selected_detail = get_match_detail(int(selected_match_id))
        if selected_detail:
            options = pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "match_id": int(selected_detail["match_id"]),
                                "competition_id": selected_detail.get("competition_id"),
                                "competition_name": selected_detail.get("competition_name"),
                                "season": selected_detail.get("season"),
                                "match_ts": selected_detail.get("kickoff_utc") or selected_detail.get("match_date"),
                                "status": selected_detail.get("status") or "UNKNOWN",
                                "home_team": selected_detail.get("home_team"),
                                "away_team": selected_detail.get("away_team"),
                            }
                        ]
                    ),
                    options,
                ],
                ignore_index=True,
            ).drop_duplicates(subset=["match_id"], keep="first")

    options["match_ts"] = pd.to_datetime(options.get("match_ts"), errors="coerce", utc=True)
    options["status"] = options["status"].fillna("UNKNOWN").astype(str)
    options["competition_name"] = options["competition_name"].fillna("Championnat").astype(str)
    options["season"] = options["season"].fillna("-").astype(str)

    with st.expander("Changer de match", expanded=False):
        st.caption("Utilise ce selecteur seulement si tu veux quitter le match ouvert depuis TEAM ou OVERVIEW.")
        filters = st.columns(3)
        competition_labels = ["Toutes"] + sorted(options["competition_name"].dropna().astype(str).unique().tolist())
        selected_competition = filters[0].selectbox("Championnat", competition_labels, key="match_detail_filter_competition")

        scoped = options.copy()
        if selected_competition != "Toutes":
            scoped = scoped[scoped["competition_name"] == selected_competition]

        season_labels = ["Toutes"] + sorted(scoped["season"].dropna().astype(str).unique().tolist(), reverse=True)
        selected_season = filters[1].selectbox("Saison", season_labels, key="match_detail_filter_season")
        if selected_season != "Toutes":
            scoped = scoped[scoped["season"] == selected_season]

        scope_label = filters[2].selectbox(
            "Perimetre",
            ["Tous", "Matchs joues", "Matchs a venir"],
            key="match_detail_filter_scope",
        )
        if scope_label == "Matchs joues":
            scoped = scoped[scoped["status"].str.upper() == "FINISHED"]
        elif scope_label == "Matchs a venir":
            scoped = scoped[scoped["status"].str.upper() != "FINISHED"]

        if scoped.empty:
            st.info("Aucun match pour ce filtre. Reviens sur 'Tous' pour voir toute la base.")
            scoped = options.copy()

        scoped = scoped.sort_values(["match_ts", "match_id"], ascending=[False, False]).reset_index(drop=True)
        scoped["label"] = scoped.apply(_format_match_label, axis=1)

        match_ids = scoped["match_id"].astype(int).tolist()
        active_match_id = int(selected_match_id) if selected_match_id in match_ids else match_ids[0]
        default_index = match_ids.index(active_match_id)
        label_by_match_id = {
            int(row["match_id"]): str(row["label"])
            for _, row in scoped.iterrows()
        }

        picked_match_id = st.selectbox(
            "Choisir un match",
            match_ids,
            index=default_index,
            key="match_detail_manual_picker",
            format_func=lambda match_id: label_by_match_id.get(int(match_id), f"Match {match_id}"),
        )
        picked_match_id = int(picked_match_id)

        if st.button("Charger le match selectionne", key="match_detail_load_selected"):
            st.session_state["selected_match_id"] = picked_match_id
            st.query_params["match_id"] = str(picked_match_id)
            st.rerun()

    match_ids = options["match_id"].astype(int).tolist()
    active_match_id = int(selected_match_id) if selected_match_id in match_ids else match_ids[0]

    st.session_state["selected_match_id"] = active_match_id
    st.query_params["match_id"] = str(active_match_id)
    return active_match_id


def _format_scoreline(home_score: object, away_score: object) -> str:
    if pd.isna(home_score) or pd.isna(away_score):
        return "-"
    return f"{int(home_score)}-{int(away_score)}"


def _delta_badge(delta: int | float | None) -> tuple[str, str]:
    if delta is None or pd.isna(delta) or int(delta) == 0:
        return "=", ""
    if int(delta) > 0:
        return f"↑{int(delta)}", "fdp-rank-delta-up"
    return f"↓{abs(int(delta))}", "fdp-rank-delta-down"


def _render_ranking_cards(df: pd.DataFrame) -> None:
    cards: list[str] = []
    for _, row in df.iterrows():
        delta_text, delta_class = _delta_badge(row.get("delta"))
        cards.append(
            (
                f'<div class="fdp-rank-card">'
                f'<div class="fdp-rank-phase">{row["phase"]}</div>'
                f'<div class="fdp-rank-team">{row["team_name"]}</div>'
                f'<div class="fdp-rank-main">'
                f'<div class="fdp-rank-pos">#{int(row["position"]) if pd.notna(row["position"]) else "--"}</div>'
                f'<div class="fdp-rank-delta {delta_class}">{delta_text}</div>'
                f"</div>"
                f'<div class="fdp-rank-meta">'
                f'<div class="fdp-rank-meta-item"><div class="fdp-rank-meta-label">Pts</div><div class="fdp-rank-meta-value">{int(row["points"]) if pd.notna(row["points"]) else "--"}</div></div>'
                f'<div class="fdp-rank-meta-item"><div class="fdp-rank-meta-label">MP</div><div class="fdp-rank-meta-value">{int(row["played_games"]) if pd.notna(row["played_games"]) else "--"}</div></div>'
                f'<div class="fdp-rank-meta-item"><div class="fdp-rank-meta-label">GD</div><div class="fdp-rank-meta-value">{int(row["goal_difference"]) if pd.notna(row["goal_difference"]) else "--"}</div></div>'
                f"</div>"
                f"</div>"
            )
        )
    st.markdown(f"<div class='fdp-match-visual-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _render_h2h_timeline(df: pd.DataFrame) -> None:
    cards: list[str] = []
    for _, row in df.iterrows():
        cards.append(
            (
                f'<div class="fdp-h2h-card">'
                f'<div class="fdp-h2h-date">{_format_kickoff(row.get("match_ts"), None)}</div>'
                f'<div class="fdp-h2h-match">{row["home_team"]} vs {row["away_team"]}</div>'
                f'<div class="fdp-h2h-score">{_format_scoreline(row.get("home_score"), row.get("away_score"))}</div>'
                f'<div class="fdp-h2h-meta">{row["competition_name"]} • {row["season"]} • MD {row["matchday"] if pd.notna(row["matchday"]) else "--"}</div>'
                f"</div>"
            )
        )
    st.markdown(f"<div class='fdp-h2h-strip'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _recent_form(matches: pd.DataFrame, team_id: int, limit: int = 5) -> list[str]:
    if matches.empty:
        return []
    perspective = build_perspective_table(matches, team_id=team_id)
    if perspective.empty:
        return []
    played = (
        perspective.dropna(subset=["result"])
        .sort_values(["date_dt", "match_id"], ascending=[False, False])
        .head(limit)
        .sort_values(["date_dt", "match_id"])
    )
    return played["result"].astype(str).tolist()


def _split_row(split_df: pd.DataFrame, venue: str) -> dict[str, int | float] | None:
    if split_df.empty:
        return None
    matched = split_df[split_df["venue"] == venue]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _build_event_timeline(
    player_stats: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    home_team_name: str,
    away_team_name: str,
) -> pd.DataFrame:
    if player_stats.empty:
        return pd.DataFrame()

    allowed_team_ids = {int(home_team_id), int(away_team_id)}
    stats = player_stats.copy()
    stats = stats[stats["team_id"].fillna(-1).astype(int).isin(allowed_team_ids)]
    if stats.empty:
        return pd.DataFrame()

    events: list[dict[str, object]] = []
    for _, row in stats.iterrows():
        team_id = int(row["team_id"])
        team_name = home_team_name if team_id == int(home_team_id) else away_team_name
        player_name = str(row.get("full_name") or "Unknown")
        minutes = pd.to_numeric(row.get("minutes"), errors="coerce")
        minute_sort = int(minutes) if pd.notna(minutes) else 90
        minute_label = f"{int(minutes)}'" if pd.notna(minutes) else "--"

        goals = int(pd.to_numeric(row.get("goals"), errors="coerce") or 0)
        assists = int(pd.to_numeric(row.get("assists"), errors="coerce") or 0)
        if goals > 0:
            events.append(
                {
                    "minute_sort": minute_sort,
                    "Minute": minute_label,
                    "Equipe": team_name,
                    "Event": "But",
                    "Joueur": player_name,
                    "Detail": f"x{goals}" if goals > 1 else "1 but",
                }
            )
        if assists > 0:
            events.append(
                {
                    "minute_sort": minute_sort,
                    "Minute": minute_label,
                    "Equipe": team_name,
                    "Event": "Passe dec.",
                    "Joueur": player_name,
                    "Detail": f"x{assists}" if assists > 1 else "1 assist",
                }
            )

    # Proxy substitution events: low minutes usually indicate bench appearances.
    sub_candidates = stats[
        (pd.to_numeric(stats["minutes"], errors="coerce") > 0)
        & (pd.to_numeric(stats["minutes"], errors="coerce") <= 30)
        & (pd.to_numeric(stats["goals"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(stats["assists"], errors="coerce").fillna(0) == 0)
    ].copy()
    if not sub_candidates.empty:
        sub_candidates["minutes_num"] = pd.to_numeric(sub_candidates["minutes"], errors="coerce")
        for team_id, group in sub_candidates.groupby("team_id"):
            for _, row in group.sort_values(["minutes_num", "full_name"]).head(2).iterrows():
                team_id_int = int(team_id)
                team_name = home_team_name if team_id_int == int(home_team_id) else away_team_name
                minute_val = int(row["minutes_num"]) if pd.notna(row["minutes_num"]) else 0
                events.append(
                    {
                        "minute_sort": minute_val,
                        "Minute": f"{minute_val}'",
                        "Equipe": team_name,
                        "Event": "Entree (proxy)",
                        "Joueur": str(row.get("full_name") or "Unknown"),
                        "Detail": "Minutes faibles detectees",
                    }
                )

    if not events:
        return pd.DataFrame()

    timeline = pd.DataFrame(events).sort_values(["minute_sort", "Equipe", "Event", "Joueur"]).reset_index(drop=True)
    return timeline[["Minute", "Equipe", "Event", "Joueur", "Detail"]]


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "MATCH DETAIL",
        "Vue detail d'un match selectionne depuis OVERVIEW ou TEAM.",
        None,
    )

    selected_match_id = _resolve_selected_match_id()
    render_note_card(
        "Le match ouvert depuis TEAM ou OVERVIEW reste prioritaire. Le changement manuel est disponible dans un panneau secondaire."
    )
    selected_match_id = _render_picker(selected_match_id)
    if selected_match_id is None:
        return

    detail = get_match_detail(int(selected_match_id))
    if not detail:
        st.info("Le match selectionne est introuvable en base.")
        return

    home_score = "-" if pd.isna(detail.get("home_score")) else int(detail["home_score"])
    away_score = "-" if pd.isna(detail.get("away_score")) else int(detail["away_score"])

    left, center, right = st.columns([3, 2, 3], vertical_alignment="center")
    with left:
        if detail.get("home_crest_url"):
            st.image(str(detail["home_crest_url"]), width=78)
        st.markdown(f"## {detail['home_team']}")
    with center:
        st.markdown(f"## {home_score} - {away_score}")
        st.caption(_format_kickoff(detail.get("kickoff_utc"), detail.get("match_date")))
        st.caption(str(detail.get("status") or "UNKNOWN"))
    with right:
        if detail.get("away_crest_url"):
            st.image(str(detail["away_crest_url"]), width=78)
        st.markdown(f"## {detail['away_team']}")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Championnat": detail["competition_name"],
                    "Saison": detail["season"],
                    "Journee": "--" if pd.isna(detail.get("matchday")) else int(detail["matchday"]),
                    "Statut": detail.get("status") or "UNKNOWN",
                }
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    all_matches = get_matches(
        competition_id=int(detail["competition_id"]),
        season=str(detail["season"]),
        team_id=None,
        date_start=None,
        date_end=None,
    )

    render_section_heading("Comparaison de forme", "Lecture rapide de la dynamique recente des deux clubs avant ce match.")
    form_left, form_right = st.columns(2)
    home_form = _recent_form(all_matches, int(detail["home_team_id"]))
    away_form = _recent_form(all_matches, int(detail["away_team_id"]))
    with form_left:
        st.caption(detail["home_team"])
        render_result_strip(home_form)
        st.metric("Points sur les 5 derniers", sum(3 if result == "W" else 1 if result == "D" else 0 for result in home_form))
    with form_right:
        st.caption(detail["away_team"])
        render_result_strip(away_form)
        st.metric("Points sur les 5 derniers", sum(3 if result == "W" else 1 if result == "D" else 0 for result in away_form))

    render_section_heading("Domicile vs exterieur", "Performance du club a domicile et de l adversaire a l exterieur sur la saison.")
    split_left, split_right = st.columns(2)
    home_split = get_home_away_split(int(detail["competition_id"]), str(detail["season"]), int(detail["home_team_id"]), None)
    away_split = get_home_away_split(int(detail["competition_id"]), str(detail["season"]), int(detail["away_team_id"]), None)
    home_row = _split_row(home_split, "Home")
    away_row = _split_row(away_split, "Away")
    with split_left:
        st.caption(f"{detail['home_team']} at home")
        if home_row is None:
            st.info("No home split available.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Pts", int(home_row["Points"]))
            c2.metric("GF", int(home_row["GoalsFor"]))
            c3.metric("GA", int(home_row["GoalsAgainst"]))
    with split_right:
        st.caption(f"{detail['away_team']} away")
        if away_row is None:
            st.info("No away split available.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Pts", int(away_row["Points"]))
            c2.metric("GF", int(away_row["GoalsFor"]))
            c3.metric("GA", int(away_row["GoalsAgainst"]))

    render_section_heading("xG proxy comparison", "Tendance recente des deux equipes sur la saison de ce match (proxy via tirs).")
    xg_left, xg_right = st.columns(2)
    home_xg = get_team_xg_proxy(
        competition_id=int(detail["competition_id"]),
        season=str(detail["season"]),
        team_id=int(detail["home_team_id"]),
        limit=10,
    )
    away_xg = get_team_xg_proxy(
        competition_id=int(detail["competition_id"]),
        season=str(detail["season"]),
        team_id=int(detail["away_team_id"]),
        limit=10,
    )
    with xg_left:
        st.caption(str(detail["home_team"]))
        if home_xg.empty:
            st.info("xG proxy indisponible.")
        else:
            st.metric("xG for moyen", f"{home_xg['xg_for'].mean():.2f}")
            st.metric("xGA moyen", f"{home_xg['xga'].mean():.2f}")
            st.line_chart(home_xg.set_index("match_label")[["xg_for", "xga"]])
    with xg_right:
        st.caption(str(detail["away_team"]))
        if away_xg.empty:
            st.info("xG proxy indisponible.")
        else:
            st.metric("xG for moyen", f"{away_xg['xg_for'].mean():.2f}")
            st.metric("xGA moyen", f"{away_xg['xga'].mean():.2f}")
            st.line_chart(away_xg.set_index("match_label")[["xg_for", "xga"]])

    render_section_heading("Event timeline (available data)")
    st.caption(
        "Timeline construite depuis les stats joueurs en base (buts, assists, et entrees proxy). "
        "Cartons et remplacements exacts ne sont pas encore disponibles."
    )
    player_stats = get_match_player_stats(int(selected_match_id))
    timeline = _build_event_timeline(
        player_stats=player_stats,
        home_team_id=int(detail["home_team_id"]),
        away_team_id=int(detail["away_team_id"]),
        home_team_name=str(detail["home_team"]),
        away_team_name=str(detail["away_team"]),
    )
    if timeline.empty:
        st.info("Aucun evenement detaille disponible pour ce match.")
    else:
        render_adaptive_table(
            timeline,
            title="Timeline",
            strong_columns={"Equipe", "Joueur"},
            max_height=620,
        )

    render_section_heading("Head-to-head", "Dernieres confrontations deja jouees entre ces deux equipes.")
    h2h = get_match_head_to_head(int(selected_match_id), limit=5)
    if h2h.empty:
        st.info("Aucune confrontation precedente disponible en base.")
    else:
        _render_h2h_timeline(h2h)

    render_section_heading("Before / After ranking", "Position et points des deux clubs autour de la journee de ce match.")
    ranking_context = get_match_ranking_context(
        competition_id=int(detail["competition_id"]),
        season=detail.get("season"),
        matchday=detail.get("matchday"),
        home_team_id=int(detail["home_team_id"]),
        away_team_id=int(detail["away_team_id"]),
    )
    if ranking_context.empty:
        st.info("Pas de snapshot de classement disponible pour comparer avant/apres ce match.")
    else:
        before_df = ranking_context[ranking_context["phase"] == "Before"].copy()
        after_df = ranking_context[ranking_context["phase"] == "After"].copy()
        if before_df.empty and after_df.empty:
            st.info("Aucun snapshot exploitable pour ce match.")
        else:
            merged = after_df.merge(
                before_df[["team_id", "position"]].rename(columns={"position": "before_position"}),
                on="team_id",
                how="left",
            )
            merged["delta"] = merged["before_position"] - merged["position"]
            _render_ranking_cards(merged)


if __name__ == "__main__":
    main()

