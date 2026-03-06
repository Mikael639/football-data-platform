from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from data.dashboard_data import (
    DashboardFilters,
    current_season_label,
    current_season_start_year_dash,
    get_matches,
    get_team_xg_proxy,
    get_teams,
    split_recent_and_upcoming_matches,
)
from state.filters import render_global_filters
from ui.adaptive_tables import render_adaptive_table
from ui.display import render_note_card, render_page_banner, render_section_heading
from ui.exports import render_csv_download
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="PREDICTION - Football Data Platform", layout="wide")


def _poisson_pmf(lmbda: float, k: int) -> float:
    if lmbda <= 0:
        return 0.0 if k > 0 else 1.0
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)


def _build_probabilities(home_lambda: float, away_lambda: float, max_goals: int = 6) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = _poisson_pmf(home_lambda, home_goals) * _poisson_pmf(away_lambda, away_goals)
            rows.append(
                {"home_goals": home_goals, "away_goals": away_goals, "probability": float(probability)}
            )
    table = pd.DataFrame(rows)
    total = float(table["probability"].sum()) or 1.0
    table["probability"] = table["probability"] / total
    return table


def _expected_goals(matches: pd.DataFrame, home_team_id: int, away_team_id: int) -> tuple[float, float]:
    played = matches.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        return 1.2, 1.0

    league_home_avg = float(played["home_score"].mean() or 1.2)
    league_away_avg = float(played["away_score"].mean() or 1.0)

    home_rows = played[played["home_team_id"] == int(home_team_id)]
    away_rows = played[played["away_team_id"] == int(away_team_id)]
    home_def_rows = played[played["home_team_id"] == int(home_team_id)]
    away_def_rows = played[played["away_team_id"] == int(away_team_id)]

    home_attack = float(home_rows["home_score"].mean() or league_home_avg) / max(league_home_avg, 0.01)
    away_attack = float(away_rows["away_score"].mean() or league_away_avg) / max(league_away_avg, 0.01)
    home_defence = float(home_def_rows["away_score"].mean() or league_away_avg) / max(league_away_avg, 0.01)
    away_defence = float(away_def_rows["home_score"].mean() or league_home_avg) / max(league_home_avg, 0.01)

    expected_home = league_home_avg * home_attack * away_defence
    expected_away = league_away_avg * away_attack * home_defence

    expected_home = min(max(expected_home, 0.2), 3.8)
    expected_away = min(max(expected_away, 0.2), 3.8)
    return float(expected_home), float(expected_away)


def _resolve_training_scope(matches: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, str]:
    if matches.empty:
        return matches, "global (fallback)"

    competition_id = row.get("competition_id")
    season = str(row.get("season") or "")
    same_comp_season = matches[
        (matches["competition_id"] == competition_id) & (matches["season"].astype(str) == season)
    ]
    same_comp = matches[matches["competition_id"] == competition_id]
    global_played = matches.dropna(subset=["home_score", "away_score"])

    if len(same_comp_season.dropna(subset=["home_score", "away_score"])) >= 20:
        return same_comp_season, "competition+saison"
    if len(same_comp.dropna(subset=["home_score", "away_score"])) >= 20:
        return same_comp, "competition"
    return global_played, "global (fallback)"


def _kickoff_label(row: pd.Series) -> str:
    kickoff = pd.to_datetime(row.get("kickoff_utc"), errors="coerce", utc=True)
    if pd.notna(kickoff):
        return kickoff.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M")
    match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
    if pd.notna(match_date):
        return match_date.strftime("%Y-%m-%d")
    return "Unknown"


def _compute_upcoming_predictions(matches: pd.DataFrame, limit: int) -> pd.DataFrame:
    _, upcoming = split_recent_and_upcoming_matches(
        matches,
        recent_limit=2000,
        upcoming_limit=max(int(limit), 1),
    )
    if upcoming.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    training_cache: dict[tuple[Any, str], tuple[pd.DataFrame, str]] = {}
    for _, fixture in upcoming.iterrows():
        home_team_id = int(fixture["home_team_id"])
        away_team_id = int(fixture["away_team_id"])
        cache_key = (fixture.get("competition_id"), str(fixture.get("season") or ""))
        if cache_key not in training_cache:
            training_cache[cache_key] = _resolve_training_scope(matches, fixture)
        training_set, training_scope = training_cache[cache_key]

        home_lambda, away_lambda = _expected_goals(training_set, home_team_id, away_team_id)
        probs = _build_probabilities(home_lambda, away_lambda, max_goals=6)
        most_likely = probs.sort_values("probability", ascending=False).head(1).iloc[0]
        home_win = float(probs[probs["home_goals"] > probs["away_goals"]]["probability"].sum())
        draw = float(probs[probs["home_goals"] == probs["away_goals"]]["probability"].sum())
        away_win = float(probs[probs["home_goals"] < probs["away_goals"]]["probability"].sum())

        best_outcome = max(
            [("1", home_win), ("N", draw), ("2", away_win)],
            key=lambda item: item[1],
        )
        rows.append(
            {
                "Kickoff (Paris)": _kickoff_label(fixture),
                "Competition": str(fixture.get("competition_name") or ""),
                "Saison": str(fixture.get("season") or ""),
                "Match": f"{fixture['home_team']} vs {fixture['away_team']}",
                "1 (%)": round(home_win * 100, 1),
                "N (%)": round(draw * 100, 1),
                "2 (%)": round(away_win * 100, 1),
                "Score probable": f"{int(most_likely['home_goals'])}-{int(most_likely['away_goals'])}",
                "xG home": round(home_lambda, 2),
                "xG away": round(away_lambda, 2),
                "Signal": best_outcome[0],
                "Confiance (%)": round(best_outcome[1] * 100, 1),
                "Scope modele": training_scope,
            }
        )
    return pd.DataFrame(rows)


def _current_prediction_season() -> str:
    return current_season_label(current_season_start_year_dash())


def _match_outcome(home_score: object, away_score: object) -> str:
    home = int(home_score)
    away = int(away_score)
    if home > away:
        return "1"
    if home == away:
        return "N"
    return "2"


def _calibrate_baseline_model(matches: pd.DataFrame, min_train_matches: int = 30) -> tuple[dict[str, Any], pd.DataFrame]:
    if matches.empty:
        return {"count": 0, "accuracy_pct": None, "brier": None, "logloss": None, "level": "N/A"}, pd.DataFrame()

    played = matches.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty or len(played.index) < (min_train_matches + 5):
        return {"count": 0, "accuracy_pct": None, "brier": None, "logloss": None, "level": "Insufficient data"}, pd.DataFrame()

    played["kickoff_sort"] = pd.to_datetime(played["kickoff_utc"], errors="coerce", utc=True).dt.tz_convert(None)
    played["kickoff_sort"] = played["kickoff_sort"].fillna(pd.to_datetime(played["date_dt"], errors="coerce"))
    played = played.sort_values(["kickoff_sort", "match_id"], ascending=[True, True]).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    first_eval_index = min_train_matches
    for idx in range(first_eval_index, len(played.index)):
        train = played.iloc[:idx].copy()
        row = played.iloc[idx]

        home_lambda, away_lambda = _expected_goals(train, int(row["home_team_id"]), int(row["away_team_id"]))
        probs = _build_probabilities(home_lambda, away_lambda, max_goals=6)
        p1 = float(probs[probs["home_goals"] > probs["away_goals"]]["probability"].sum())
        pn = float(probs[probs["home_goals"] == probs["away_goals"]]["probability"].sum())
        p2 = float(probs[probs["home_goals"] < probs["away_goals"]]["probability"].sum())

        actual = _match_outcome(row["home_score"], row["away_score"])
        prob_by_outcome = {"1": p1, "N": pn, "2": p2}
        predicted = max(prob_by_outcome.items(), key=lambda item: item[1])[0]
        y1 = 1.0 if actual == "1" else 0.0
        yn = 1.0 if actual == "N" else 0.0
        y2 = 1.0 if actual == "2" else 0.0
        brier = ((p1 - y1) ** 2 + (pn - yn) ** 2 + (p2 - y2) ** 2) / 3.0
        p_actual = max(prob_by_outcome[actual], 1e-9)
        logloss = -math.log(p_actual)

        records.append(
            {
                "Kickoff (Paris)": _kickoff_label(row),
                "Competition": str(row.get("competition_name") or ""),
                "Match": f"{row['home_team']} vs {row['away_team']}",
                "Pred": predicted,
                "Actual": actual,
                "P(actual) %": round(p_actual * 100, 1),
                "Brier": round(float(brier), 4),
                "LogLoss": round(float(logloss), 4),
                "Correct": 1 if predicted == actual else 0,
            }
        )

    if not records:
        return {"count": 0, "accuracy_pct": None, "brier": None, "logloss": None, "level": "Insufficient data"}, pd.DataFrame()

    calibration = pd.DataFrame(records)
    accuracy_pct = float(calibration["Correct"].mean() * 100.0)
    brier_score = float(calibration["Brier"].mean())
    logloss_score = float(calibration["LogLoss"].mean())
    level = "Good" if brier_score <= 0.20 else "Medium" if brier_score <= 0.26 else "Weak"
    summary = {
        "count": int(len(calibration.index)),
        "accuracy_pct": round(accuracy_pct, 1),
        "brier": round(brier_score, 4),
        "logloss": round(logloss_score, 4),
        "level": level,
    }
    calibration = calibration.sort_values("Kickoff (Paris)", ascending=False).reset_index(drop=True)
    return summary, calibration


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "PREDICTION",
        "Modele baseline Poisson: probabilites 1N2 et score probable sur la saison en cours.",
        None,
    )
    render_note_card("MVP prediction: utile pour un ordre de grandeur, pas pour du betting.")

    forced_season = _current_prediction_season()
    filters = render_global_filters("prediction", forced_season=forced_season)
    effective_filters = DashboardFilters(
        competition_id=filters.competition_id,
        season=forced_season,
        team_id=filters.team_id,
        date_start=filters.date_start,
        date_end=filters.date_end,
    )
    st.caption(f"Saison forcee sur cet onglet: {forced_season}")

    scope_matches = get_matches(
        competition_id=effective_filters.competition_id,
        season=effective_filters.season,
        team_id=effective_filters.team_id,
        date_start=effective_filters.date_start,
        date_end=effective_filters.date_end,
    )
    model_matches = get_matches(
        competition_id=effective_filters.competition_id,
        season=effective_filters.season,
        team_id=None,
        date_start=effective_filters.date_start,
        date_end=effective_filters.date_end,
    )
    if model_matches.empty:
        st.info("Aucun match disponible pour entrainer le baseline.")
        return

    render_section_heading("Predictions automatiques des prochains matchs")
    c_pred1, c_pred2 = st.columns([2, 1])
    with c_pred1:
        st.caption("Predictions 1/N/2 pour les matchs a venir dans les competitions visibles du filtre.")
    with c_pred2:
        upcoming_limit = st.slider(
            "Nombre de matchs a predire",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
            key="prediction_upcoming_limit",
        )

    upcoming_predictions = _compute_upcoming_predictions(scope_matches, limit=upcoming_limit)
    if upcoming_predictions.empty:
        st.info("Aucun match a venir detecte sur ce filtre.")
    else:
        render_adaptive_table(
            upcoming_predictions,
            title=f"Prochains {len(upcoming_predictions)} matchs",
            strong_columns={"Match", "Score probable"},
            max_height=840,
        )
        render_csv_download(
            df=upcoming_predictions,
            label="Export predictions matchs a venir (CSV)",
            filename="prediction_upcoming_matches.csv",
            key="prediction_export_upcoming",
        )

    render_section_heading("Calibration du modele (retrospective)")
    st.caption(
        "Evaluation out-of-time: chaque match est predit avec les matchs precedents uniquement "
        f"(minimum {30} matchs d'entrainement)."
    )
    calibration_summary, calibration_table = _calibrate_baseline_model(model_matches, min_train_matches=30)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches eval", int(calibration_summary["count"]))
    c2.metric(
        "Accuracy",
        "-" if calibration_summary["accuracy_pct"] is None else f"{calibration_summary['accuracy_pct']}%",
    )
    c3.metric("Brier", "-" if calibration_summary["brier"] is None else f"{calibration_summary['brier']}")
    c4.metric("LogLoss", "-" if calibration_summary["logloss"] is None else f"{calibration_summary['logloss']}")
    c5.metric("Level", str(calibration_summary["level"]))
    if calibration_table.empty:
        st.info("Pas assez de matchs termines pour calibrer le modele sur ce scope.")
    else:
        recent_calibration = calibration_table.head(20).copy()
        render_adaptive_table(
            recent_calibration[
                ["Kickoff (Paris)", "Competition", "Match", "Pred", "Actual", "P(actual) %", "Brier", "LogLoss"]
            ],
            title="Dernieres evaluations",
            strong_columns={"Match"},
            max_height=700,
        )
        render_csv_download(
            df=calibration_table,
            label="Export calibration modele (CSV)",
            filename="prediction_model_calibration.csv",
            key="prediction_export_calibration",
        )

    with st.expander("Prediction manuelle d'une affiche (deplier/replier)", expanded=False):
        teams = get_teams(effective_filters.competition_id, effective_filters.season)
        if teams.empty:
            st.info("Aucune equipe disponible pour ce filtre.")
            return

        team_options = teams[["team_id", "team_name"]].drop_duplicates().sort_values("team_name")
        names = team_options["team_name"].astype(str).tolist()
        ids_by_name = dict(zip(team_options["team_name"].astype(str), team_options["team_id"].astype(int)))

        c1, c2 = st.columns(2)
        with c1:
            home_team_name = st.selectbox("Home team", names, key="prediction_home_team")
        with c2:
            away_team_name = st.selectbox(
                "Away team",
                names,
                index=1 if len(names) > 1 else 0,
                key="prediction_away_team",
            )

        home_team_id = int(ids_by_name[home_team_name])
        away_team_id = int(ids_by_name[away_team_name])
        if home_team_id == away_team_id:
            st.warning("Selectionne deux clubs differents.")
            return

        home_lambda, away_lambda = _expected_goals(model_matches, home_team_id, away_team_id)
        probs = _build_probabilities(home_lambda, away_lambda, max_goals=6)
        most_likely = probs.sort_values("probability", ascending=False).head(1).iloc[0]
        home_win = float(probs[probs["home_goals"] > probs["away_goals"]]["probability"].sum())
        draw = float(probs[probs["home_goals"] == probs["away_goals"]]["probability"].sum())
        away_win = float(probs[probs["home_goals"] < probs["away_goals"]]["probability"].sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Home win", f"{home_win * 100:.1f}%")
        m2.metric("Draw", f"{draw * 100:.1f}%")
        m3.metric("Away win", f"{away_win * 100:.1f}%")
        m4.metric("Score probable", f"{int(most_likely['home_goals'])}-{int(most_likely['away_goals'])}")

        render_section_heading("Expected goals (baseline)")
        eg1, eg2 = st.columns(2)
        eg1.metric(f"xG {home_team_name}", f"{home_lambda:.2f}")
        eg2.metric(f"xG {away_team_name}", f"{away_lambda:.2f}")

        render_section_heading("Top scorelines")
        top_scores = probs.sort_values("probability", ascending=False).head(10).copy()
        top_scores["Score"] = top_scores.apply(
            lambda row: f"{int(row['home_goals'])}-{int(row['away_goals'])}", axis=1
        )
        top_scores["Probability"] = (top_scores["probability"] * 100).round(2)
        top_display = top_scores[["Score", "Probability"]]
        render_adaptive_table(top_display, title="Top 10", strong_columns={"Score"}, max_height=500)
        render_csv_download(
            df=top_display,
            label="Export probabilites de score (CSV)",
            filename="prediction_score_probabilities.csv",
            key="prediction_export_scores",
        )

        render_section_heading("xG proxy recent (context)")
        st.caption(
            "Ce xG proxy concerne les 2 equipes choisies ci-dessus (prediction manuelle), "
            "sur la competition filtree et la saison en cours. Il est estime via les tirs (proxy)."
        )
        ctx_left, ctx_right = st.columns(2)
        with ctx_left:
            home_xg = get_team_xg_proxy(
                effective_filters.competition_id,
                effective_filters.season,
                home_team_id,
                limit=8,
            )
            if home_xg.empty:
                st.info("xG proxy indisponible pour le home team.")
            else:
                st.line_chart(home_xg.set_index("match_label")[["xg_for", "xga"]])
        with ctx_right:
            away_xg = get_team_xg_proxy(
                effective_filters.competition_id,
                effective_filters.season,
                away_team_id,
                limit=8,
            )
            if away_xg.empty:
                st.info("xG proxy indisponible pour le away team.")
            else:
                st.line_chart(away_xg.set_index("match_label")[["xg_for", "xga"]])


if __name__ == "__main__":
    main()

