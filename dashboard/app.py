import os
import json
from datetime import date, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Football Data Platform", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_PLAYER_PLACEHOLDER = APP_DIR / "assets" / "player-placeholder.svg"
DEFAULT_LALIGA_BADGE = APP_DIR / "assets" / "laliga-badge.svg"
STUDY_FBREF_DIR = APP_DIR.parent / "data" / "study" / "fbref"
VISUAL_COLORS = {
    "points": "#0E6FFF",
    "attack": "#1FA774",
    "defense": "#D64B4B",
    "neutral": "#C58B1A",
    "teal": "#1697A6",
    "violet": "#5B6CF0",
}


def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --fdp-bg-soft: #f3f6fb;
          --fdp-ink: #132238;
          --fdp-accent: #0e6fff;
          --fdp-accent-2: #16b39a;
          --fdp-border: #d9e2ef;
          --fdp-card: #ffffff;
        }

        .stApp {
          background:
            radial-gradient(circle at 0% 0%, rgba(14,111,255,0.08), transparent 38%),
            radial-gradient(circle at 100% 20%, rgba(22,179,154,0.08), transparent 40%),
            linear-gradient(180deg, #f7f9fd 0%, #eef3fa 100%);
        }

        div[data-testid="stMetric"] {
          background: var(--fdp-card);
          border: 1px solid var(--fdp-border);
          border-radius: 14px;
          padding: 10px 12px;
          box-shadow: 0 3px 12px rgba(19,34,56,0.05);
        }

        div[data-testid="stTabs"] button[role="tab"] {
          border-radius: 12px;
          padding: 8px 14px;
          border: 1px solid transparent;
          color: #29415f;
          font-weight: 600;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
          background: rgba(14,111,255,0.10);
          border-color: rgba(14,111,255,0.25);
          color: #0b4fb4;
        }

        div.stButton > button {
          border-radius: 12px;
          border: 1px solid #c9d7ea;
          background: #ffffff;
          color: #163252;
          font-weight: 600;
        }

        div.stButton > button:hover {
          border-color: #0e6fff;
          color: #0e6fff;
        }

        .fdp-hero {
          background:
            linear-gradient(135deg, rgba(14,111,255,0.12), rgba(22,179,154,0.10)),
            #ffffff;
          border: 1px solid var(--fdp-border);
          border-radius: 18px;
          padding: 16px 18px;
          box-shadow: 0 8px 24px rgba(19,34,56,0.06);
          margin: 4px 0 12px 0;
        }

        .fdp-hero-title {
          font-size: 1.1rem;
          font-weight: 700;
          color: var(--fdp-ink);
          margin-bottom: 4px;
        }

        .fdp-hero-sub {
          color: #405a78;
          font-size: 0.92rem;
          line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "football_dw")
    user = os.getenv("DB_USER", "football")
    pwd = os.getenv("DB_PASSWORD", "football")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, pool_pre_ping=True)


def current_season_start_year_dash() -> int:
    today = datetime.utcnow().date()
    return today.year if today.month >= 7 else today.year - 1


def current_season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def current_season_bounds(start_year: int) -> tuple[str, str]:
    season_start = date(start_year, 7, 1)
    season_end = date(start_year + 1, 6, 30)
    return season_start.isoformat(), season_end.isoformat()


@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_laliga_teams_live(competition_code: str, season_start_year: int):
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    base_url = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
    if not token:
        return None, "Missing FOOTBALL_DATA_TOKEN"

    try:
        response = requests.get(
            f"{base_url}/competitions/{competition_code}/teams",
            headers={"X-Auth-Token": token},
            params={"season": season_start_year},
            timeout=20,
        )
    except requests.RequestException as exc:
        return None, f"API request failed: {exc}"

    if response.status_code != 200:
        return None, f"API returned status={response.status_code}"

    try:
        payload = response.json()
    except ValueError:
        return None, "Invalid JSON from API"

    teams = payload.get("teams") or []
    rows = []
    for team in teams:
        if team.get("id") is None:
            continue
        rows.append(
            {
                "team_id": int(team["id"]),
                "team_name": team.get("name") or team.get("shortName"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["team_id", "team_name"]), None
    return pd.DataFrame(rows).drop_duplicates(subset=["team_id"]).sort_values("team_name"), None


def laliga_logo_source():
    # Use a local wordmark asset (new LaLiga branding style) for a stable UI.
    return str(DEFAULT_LALIGA_BADGE)


def render_team_hero(selected_team_name: str, season_label: str) -> None:
    title = "Vue equipe" if selected_team_name == "Tous les clubs" else f"Vue equipe - {selected_team_name}"
    subtitle = (
        f"LaLiga {season_label} | KPI, forme recente, domicile/exterieur et calendrier. "
        "Selectionne un club pour une analyse detaillee."
    )
    st.markdown(
        f"""
        <div class="fdp-hero">
          <div class="fdp-hero-title">{title}</div>
          <div class="fdp-hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_live_team_squad(team_id: int):
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    base_url = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
    if not token:
        return None, "Missing FOOTBALL_DATA_TOKEN"

    try:
        response = requests.get(
            f"{base_url}/teams/{int(team_id)}",
            headers={"X-Auth-Token": token},
            timeout=20,
        )
    except requests.RequestException as exc:
        return None, f"API request failed: {exc}"

    if response.status_code != 200:
        return None, f"API returned status={response.status_code}"

    try:
        payload = response.json()
    except ValueError:
        return None, "Invalid JSON from API"

    team = payload.get("team") or {}
    squad = payload.get("squad") or []
    rows = []
    for player in squad:
        pid = player.get("id")
        if pid is None:
            continue
        rows.append(
            {
                "player_id": int(pid),
                "full_name": player.get("name"),
                "position": player.get("position"),
                "nationality": player.get("nationality"),
                "birth_date": player.get("dateOfBirth"),
                "team_id": int(team.get("id", team_id)),
                "team_name": team.get("name"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "player_id", "full_name", "position", "nationality", "birth_date", "team_id", "team_name"
        ]), None
    return pd.DataFrame(rows), None


def compute_team_kpis(df: pd.DataFrame, team_id: int):
    played = wins = draws = losses = gf = ga = 0
    for _, row in df.iterrows():
        hs = row["home_score"]
        aw = row["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue

        played += 1
        if int(row["home_team_id"]) == team_id:
            gf += int(hs)
            ga += int(aw)
            if hs > aw:
                wins += 1
            elif hs == aw:
                draws += 1
            else:
                losses += 1
        else:
            gf += int(aw)
            ga += int(hs)
            if aw > hs:
                wins += 1
            elif aw == hs:
                draws += 1
            else:
                losses += 1

    points = wins * 3 + draws
    return played, wins, draws, losses, gf, ga, points


def build_team_match_view(df_matches: pd.DataFrame, team_id: int) -> pd.DataFrame:
    rows = []
    for _, row in df_matches.iterrows():
        hs = row["home_score"]
        aw = row["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue

        is_home = int(row["home_team_id"]) == int(team_id)
        gf = int(hs) if is_home else int(aw)
        ga = int(aw) if is_home else int(hs)
        if gf > ga:
            result = "W"
            points = 3
        elif gf == ga:
            result = "D"
            points = 1
        else:
            result = "L"
            points = 0

        rows.append(
            {
                "date_dt": pd.to_datetime(row["date_id"], errors="coerce"),
                "match_id": row["match_id"],
                "venue": "Domicile" if is_home else "Exterieur",
                "opponent": row["away_team"] if is_home else row["home_team"],
                "GF": gf,
                "GA": ga,
                "Result": result,
                "Points": points,
                "GoalsTotal": gf + ga,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["date_dt", "match_id", "venue", "opponent", "GF", "GA", "Result", "Points"]
        )

    df = pd.DataFrame(rows).sort_values(["date_dt", "match_id"]).reset_index(drop=True)
    df["CumulativePoints"] = df["Points"].cumsum()
    return df


def render_form_timeline(last_matches: pd.DataFrame) -> None:
    if last_matches.empty:
        st.info("Aucun match disponible pour la timeline de forme.")
        return

    color_map = {"W": "#1f7a1f", "D": "#b58a00", "L": "#a12622"}
    parts = []
    for _, row in last_matches.iterrows():
        result = str(row["Result"])
        color = color_map.get(result, "#666666")
        label = f"{row['date']} {result}"
        parts.append(
            "<span title='{}' style='display:inline-block;margin:2px;padding:6px 10px;"
            "border-radius:8px;background:{};color:white;font-weight:600;font-size:12px;'>"
            "{}</span>".format(label, color, result)
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_quality_badges(summary: dict[str, int]) -> None:
    badges = []
    palette = {
        "OK": ("#1f7a1f", "#ffffff"),
        "INCOMPLET": ("#a12622", "#ffffff"),
        "A_VERIFIER": ("#8a6d1d", "#ffffff"),
        "TOTAL": ("#2b3a55", "#ffffff"),
    }
    labels = {
        "OK": "OK",
        "INCOMPLET": "Incomplet",
        "A_VERIFIER": "A verifier",
        "TOTAL": "Total",
    }
    ordered_keys = ["OK", "INCOMPLET", "A_VERIFIER", "TOTAL"]
    for key in ordered_keys:
        if key not in summary:
            continue
        bg, fg = palette[key]
        badges.append(
            "<span style='display:inline-block;margin:2px 6px 2px 0;padding:6px 10px;"
            f"border-radius:999px;background:{bg};color:{fg};font-weight:600;font-size:12px;'>"
            f"{labels[key]}: {summary[key]}</span>"
        )
    if badges:
        st.markdown("".join(badges), unsafe_allow_html=True)


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


def style_ligue_table(df: pd.DataFrame):
    styler = df.style

    def color_gd(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v):
                styles.append("")
            elif float(v) < 0:
                styles.append("color:#B42318;font-weight:700;")
            elif float(v) > 0:
                styles.append("color:#067647;font-weight:700;")
            else:
                styles.append("color:#6b7280;")
        return styles

    def highlight_gf(col: pd.Series):
        if col.dropna().empty:
            return ["" for _ in col]
        q75 = col.dropna().quantile(0.75)
        q90 = col.dropna().quantile(0.90)
        out = []
        for v in col:
            if pd.isna(v):
                out.append("")
            elif float(v) >= q90:
                out.append("background-color:#d1fadf;color:#065f46;font-weight:700;")
            elif float(v) >= q75:
                out.append("background-color:#ecfdf3;color:#065f46;")
            else:
                out.append("")
        return out

    def highlight_ga(col: pd.Series):
        if col.dropna().empty:
            return ["" for _ in col]
        q75 = col.dropna().quantile(0.75)
        q90 = col.dropna().quantile(0.90)
        out = []
        for v in col:
            if pd.isna(v):
                out.append("")
            elif float(v) >= q90:
                out.append("background-color:#fee4e2;color:#991b1b;font-weight:700;")
            elif float(v) >= q75:
                out.append("background-color:#fff1f0;color:#991b1b;")
            else:
                out.append("")
        return out

    def highlight_pts(col: pd.Series):
        if col.dropna().empty:
            return ["" for _ in col]
        q75 = col.dropna().quantile(0.75)
        out = []
        for v in col:
            if pd.isna(v):
                out.append("")
            elif float(v) >= q75:
                out.append("background-color:#e9f2ff;color:#0b4fb4;font-weight:700;")
            else:
                out.append("")
        return out

    for c in ["GD", "Diff", "Diff."]:
        if c in df.columns:
            styler = styler.apply(color_gd, subset=[c])
            break
    if "GF" in df.columns:
        styler = styler.apply(highlight_gf, subset=["GF"])
    if "GA" in df.columns:
        styler = styler.apply(highlight_ga, subset=["GA"])
    if "Pts" in df.columns:
        styler = styler.apply(highlight_pts, subset=["Pts"])
    return styler


def add_leader_star(df: pd.DataFrame, team_col: str = "Equipe") -> pd.DataFrame:
    if df is None or df.empty or team_col not in df.columns:
        return df
    out = df.copy()
    out.iloc[0, out.columns.get_loc(team_col)] = f"★ {out.iloc[0][team_col]}"
    return out


def add_podium_icons(df: pd.DataFrame, team_col: str = "Equipe") -> pd.DataFrame:
    if df is None or df.empty or team_col not in df.columns:
        return df
    out = df.copy()
    icons = ["🥇", "🥈", "🥉"]
    for idx, icon in enumerate(icons):
        if idx >= len(out):
            break
        out.iloc[idx, out.columns.get_loc(team_col)] = f"{icon} {out.iloc[idx][team_col]}"
    return out


def add_podium_icons_generic(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df is None or df.empty or label_col not in df.columns:
        return df
    out = df.copy()
    icons = ["🥇", "🥈", "🥉"]
    for idx, icon in enumerate(icons):
        if idx >= len(out):
            break
        out.iloc[idx, out.columns.get_loc(label_col)] = f"{icon} {out.iloc[idx][label_col]}"
    return out


@st.cache_data(show_spinner=False, ttl=10 * 60)
def load_fbref_study_datasets():
    files = {
        "player_match": STUDY_FBREF_DIR / "player_match.csv",
        "player_season": STUDY_FBREF_DIR / "player_season.csv",
        "regularity": STUDY_FBREF_DIR / "regularity.csv",
        "progression": STUDY_FBREF_DIR / "progression.csv",
        "meta": STUDY_FBREF_DIR / "meta.json",
    }
    if not files["regularity"].exists():
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


def _season_label_from_start(start_year: int) -> str:
    return f"{int(start_year)}-{int(start_year) + 1}"


def _season_picker_label_from_start(start_year: int) -> str:
    start_year = int(start_year)
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def _study_expected_complete_seasons(n: int = 3) -> list[int]:
    current_start = current_season_start_year_dash()
    return [current_start - i for i in range(n, 0, -1)]


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


def render_player_study_tab() -> None:
    st.header("Etude Joueurs (FBref - 3 saisons completes)")

    study = load_fbref_study_datasets()
    expected = _study_expected_complete_seasons(3)
    expected_labels = ", ".join(_season_label_from_start(s) for s in expected)

    if not study:
        st.info(
            "Donnees d'etude FBref non generees. Lance `make study-fbref` (scraping direct) "
            "ou utilise le mode manuel CSV (`FBREF_STUDY_SOURCE=manual_csv`) puis rebuild le dashboard."
        )
        st.caption(
            "Mode manuel: depose `data/study/fbref_input/player_match_manual.csv` "
            "puis lance `make study-fbref-manual-docker`."
        )
        st.caption(f"Saisons ciblees (3 dernieres saisons completes): {expected_labels}")
        return

    meta = study.get("meta") or {}
    reg_df = study.get("regularity", pd.DataFrame()).copy()
    prog_df = study.get("progression", pd.DataFrame()).copy()
    season_df = study.get("player_season", pd.DataFrame()).copy()
    match_df = study.get("player_match", pd.DataFrame()).copy()

    if reg_df.empty and prog_df.empty:
        st.warning("Les fichiers FBref sont presents mais vides. Verifie l'extraction.")
        return

    generated_seasons = meta.get("season_labels") if isinstance(meta, dict) else None
    if generated_seasons:
        st.caption(
            "Saisons etudiees (FBref, saisons completes hors saison en cours): "
            + ", ".join(generated_seasons)
        )
    else:
        st.caption(f"Saisons cibles (attendues): {expected_labels}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saisons", int(len(set(reg_df.get("season_start", pd.Series(dtype=int)).dropna().tolist()))))
    c2.metric("Joueurs (regularite)", int(reg_df["player_id"].nunique()) if "player_id" in reg_df.columns else 0)
    c3.metric("Lignes progression", int(len(prog_df)))
    c4.metric("Matchs joueurs", int(len(match_df)) if not match_df.empty else 0)

    season_candidates = set()
    for df in (season_df, reg_df, prog_df):
        if isinstance(df, pd.DataFrame) and (not df.empty) and ("season_start" in df.columns):
            season_candidates.update(df["season_start"].dropna().astype(int).tolist())
    available_seasons = sorted(season_candidates, reverse=True)
    season_label_map = {_season_picker_label_from_start(s): s for s in available_seasons}
    default_season_label = next(iter(season_label_map.keys()), None)

    st.markdown("### Selection de l'etude")
    sel1, sel2, sel3 = st.columns([0.32, 0.32, 0.36])
    with sel1:
        selected_global_season_label = (
            st.selectbox("Saison", list(season_label_map.keys()), key="study_global_season")
            if season_label_map
            else None
        )
    selected_global_season = season_label_map.get(selected_global_season_label) if selected_global_season_label else None

    players_for_selector = pd.DataFrame()
    if not season_df.empty and selected_global_season is not None and "season_start" in season_df.columns:
        players_for_selector = season_df[season_df["season_start"].astype(int) == int(selected_global_season)].copy()
    elif not reg_df.empty and selected_global_season is not None and "season_start" in reg_df.columns:
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

        player_labels = players_for_selector["player_label"].tolist()
        with sel2:
            selected_player_label = st.selectbox(
                "Joueur (clique / recherche)",
                player_labels,
                key="study_player_selector",
            )
        player_lookup = dict(zip(players_for_selector["player_label"], players_for_selector["player_id"]))
        selected_player_id = player_lookup.get(selected_player_label)
    else:
        with sel2:
            st.selectbox("Joueur (clique / recherche)", ["Aucun joueur disponible"], key="study_player_selector_empty")
        selected_player_label = None
        selected_player_id = None

    position_options = ["Tous"] + sorted(
        [p for p in reg_df.get("position_group", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if p]
    )
    with sel3:
        selected_pos = st.selectbox("Poste (classement regularite)", position_options, key="study_regularite_position")

    selected_season = selected_global_season

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
        st.info("Selectionne une saison puis un joueur pour afficher sa performance et sa regularite.")
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
                season_perf["Saison"] = season_perf["season_start"].astype(int).map(_season_picker_label_from_start)
                season_perf = season_perf.sort_values("season_start", ascending=True)

                pcols = ["Saison", "team_name", "position_group", "minutes_total", "matches_played", "starts", "goals_total", "assists_total", "ga_total", "ga_p90", "passes_p90"]
                available_pcols = [c for c in pcols if c in season_perf.columns]
                perf_table = season_perf[available_pcols].rename(
                    columns={
                        "team_name": "Club",
                        "position_group": "Poste",
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
                    st.caption(f"Matchs (saison {selected_global_season_label})")
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
                st.info("Aucune donnee de regularite pour ce joueur (souvent seuil de minutes non atteint).")
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
                reg_table["Saison"] = reg_table["season_start"].map(_season_picker_label_from_start)
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
                prog_table["Saison N"] = prog_table["season_start"].map(_season_picker_label_from_start)
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
    prog_label_map = {_season_picker_label_from_start(s): s for s in prog_seasons}
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
            top_prog["Δ GA/90"] = top_prog["delta_ga_p90"].round(2)
            top_prog["Δ Passes/90"] = top_prog["delta_passes_p90"].round(2)
            st.dataframe(
                top_prog[["Joueur", "team_name", "position_group", "minutes_total_prev", "minutes_total", "Δ GA/90", "Δ Passes/90", "Score"]]
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
            worst_prog["Δ GA/90"] = worst_prog["delta_ga_p90"].round(2)
            worst_prog["Δ Passes/90"] = worst_prog["delta_passes_p90"].round(2)
            st.dataframe(
                worst_prog[["Joueur", "team_name", "position_group", "minutes_total_prev", "minutes_total", "Δ GA/90", "Δ Passes/90", "Score"]]
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

    if not season_df.empty:
        st.expander("Base joueurs-saison (aperçu)", expanded=False).dataframe(
            season_df.head(100),
            use_container_width=True,
            hide_index=True,
        )


def upsert_players_to_db(engine_obj, players_df: pd.DataFrame) -> int:
    if players_df is None or players_df.empty:
        return 0

    rows = []
    for _, row in players_df.iterrows():
        if pd.isna(row.get("player_id")):
            continue
        rows.append(
            {
                "player_id": int(row["player_id"]),
                "full_name": row.get("full_name"),
                "position": row.get("position"),
                "nationality": row.get("nationality"),
                "birth_date": row.get("birth_date"),
                "team_id": int(row["team_id"]) if pd.notna(row.get("team_id")) else None,
            }
        )

    if not rows:
        return 0

    with engine_obj.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO dim_player (player_id, full_name, position, nationality, birth_date, team_id)
                VALUES (:player_id, :full_name, :position, :nationality, :birth_date, :team_id)
                ON CONFLICT (player_id) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    position = EXCLUDED.position,
                    nationality = EXCLUDED.nationality,
                    birth_date = EXCLUDED.birth_date,
                    team_id = EXCLUDED.team_id
                """
            ),
            rows,
        )
    return len(rows)


def build_local_league_table(df_matches: pd.DataFrame) -> pd.DataFrame:
    played = df_matches.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        return pd.DataFrame(columns=["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"])

    table = {}

    def ensure(team_name: str):
        if team_name not in table:
            table[team_name] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
        return table[team_name]

    for _, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        aw = int(row["away_score"])

        h = ensure(home)
        a = ensure(away)

        h["P"] += 1
        a["P"] += 1
        h["GF"] += hs
        h["GA"] += aw
        a["GF"] += aw
        a["GA"] += hs

        if hs > aw:
            h["W"] += 1
            h["Pts"] += 3
            a["L"] += 1
        elif hs < aw:
            a["W"] += 1
            a["Pts"] += 3
            h["L"] += 1
        else:
            h["D"] += 1
            a["D"] += 1
            h["Pts"] += 1
            a["Pts"] += 1

    out = []
    for team, stats in table.items():
        out.append(
            {
                "Team": team,
                **stats,
                "GD": stats["GF"] - stats["GA"],
            }
        )
    return (
        pd.DataFrame(out)
        .sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )


engine = get_engine()
season_start_year = current_season_start_year_dash()
season_start, season_end = current_season_bounds(season_start_year)
competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")

inject_dashboard_styles()

header_left, header_right = st.columns([6.0, 1.3], vertical_alignment="center")
with header_left:
    st.title("Plateforme Data Football - LaLiga")
    st.caption(f"Saison en cours utilisee partout dans le dashboard: {current_season_label(season_start_year)}")
with header_right:
    st.image(laliga_logo_source(), use_column_width=True)

# -----------------------
# Shared selectors / data
# -----------------------
teams_df, teams_live_err = fetch_laliga_teams_live(competition_code, season_start_year)
if teams_df is None or teams_df.empty:
    teams_df = pd.read_sql(
        text(
            """
            SELECT DISTINCT t.team_id, t.team_name
            FROM dim_team t
            JOIN (
              SELECT home_team_id AS team_id
              FROM fact_match
              WHERE date_id BETWEEN :season_start AND :season_end
              UNION
              SELECT away_team_id AS team_id
              FROM fact_match
              WHERE date_id BETWEEN :season_start AND :season_end
            ) s ON s.team_id = t.team_id
            ORDER BY t.team_name;
            """
        ),
        engine,
        params={"season_start": season_start, "season_end": season_end},
    )
if teams_df.empty:
    teams_df = pd.read_sql("SELECT team_id, team_name FROM dim_team ORDER BY team_name;", engine)
team_names = ["Tous les clubs"] + teams_df["team_name"].dropna().tolist()

if "selected_team_name" not in st.session_state:
    st.session_state["selected_team_name"] = "Tous les clubs"
if st.session_state["selected_team_name"] not in team_names:
    st.session_state["selected_team_name"] = "Tous les clubs"

st.selectbox("Filtrer par club", team_names, key="selected_team_name")
selected_team_name = st.session_state["selected_team_name"]
selected_team_id = None
if selected_team_name != "Tous les clubs":
    selected_team_id = int(
        teams_df.loc[teams_df["team_name"] == selected_team_name, "team_id"].iloc[0]
    )

matches_all_query = """
SELECT
  m.match_id,
  m.date_id,
  m.home_team_id,
  ht.team_name AS home_team,
  m.away_team_id,
  at.team_name AS away_team,
  m.home_score,
  m.away_score
FROM fact_match m
JOIN dim_team ht ON ht.team_id = m.home_team_id
JOIN dim_team at ON at.team_id = m.away_team_id
WHERE m.date_id BETWEEN :season_start AND :season_end
ORDER BY m.date_id DESC NULLS LAST, m.match_id DESC;
"""
df_matches_all_season = pd.read_sql(
    text(matches_all_query),
    engine,
    params={"season_start": season_start, "season_end": season_end},
)
if not df_matches_all_season.empty:
    df_matches_all_season["date_dt"] = pd.to_datetime(df_matches_all_season["date_id"], errors="coerce")
league_local_all_season = build_local_league_table(df_matches_all_season)

match_conditions = ["m.date_id BETWEEN :season_start AND :season_end"]
match_params = {"season_start": season_start, "season_end": season_end}
if selected_team_id is not None:
    match_conditions.append("(m.home_team_id = :tid OR m.away_team_id = :tid)")
    match_params["tid"] = selected_team_id

matches_query = f"""
SELECT
  m.match_id,
  m.date_id,
  m.home_team_id,
  ht.team_name AS home_team,
  m.away_team_id,
  at.team_name AS away_team,
  m.home_score,
  m.away_score
FROM fact_match m
JOIN dim_team ht ON ht.team_id = m.home_team_id
JOIN dim_team at ON at.team_id = m.away_team_id
WHERE {' AND '.join(match_conditions)}
ORDER BY m.date_id DESC NULLS LAST, m.match_id DESC;
"""
df_matches = pd.read_sql(text(matches_query), engine, params=match_params)
if not df_matches.empty:
    df_matches["date_dt"] = pd.to_datetime(df_matches["date_id"], errors="coerce")

players_query = """
SELECT
  p.player_id, p.full_name, p.position, p.nationality, p.birth_date,
  p.team_id, t.team_name
FROM dim_player p
LEFT JOIN dim_team t ON t.team_id = p.team_id
"""
df_players_all = pd.read_sql(text(players_query), engine)

if not df_players_all.empty:
    df_players_all = (
        df_players_all.sort_values(["team_name", "full_name"], ascending=[True, True])
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

if selected_team_id is not None:
    df_players = df_players_all[df_players_all["team_id"] == selected_team_id].copy()
else:
    df_players = df_players_all.copy()

club_summary = (
    teams_df.copy()
    .merge(
        df_players_all.groupby("team_id", dropna=True)["player_id"].nunique().rename("players_count"),
        how="left",
        on="team_id",
    )
)
club_summary["players_count"] = club_summary["players_count"].fillna(0).astype(int)

if not df_matches_all_season.empty:
    home_cnt = df_matches_all_season.groupby("home_team_id")["match_id"].nunique()
    away_cnt = df_matches_all_season.groupby("away_team_id")["match_id"].nunique()
    club_summary["matches_in_scope"] = (
        club_summary["team_id"].map(home_cnt).fillna(0) + club_summary["team_id"].map(away_cnt).fillna(0)
    ).astype(int)
else:
    club_summary["matches_in_scope"] = 0

club_summary["data_quality_status"] = "OK"
club_summary.loc[
    (club_summary["matches_in_scope"] <= 0),
    "data_quality_status",
] = "A_VERIFIER"
club_summary.loc[
    (club_summary["matches_in_scope"] > 0) & (club_summary["players_count"] <= 0),
    "data_quality_status",
] = "INCOMPLET"
club_summary = club_summary.sort_values("team_name").reset_index(drop=True)

tab_team, tab_study, tab_standings, tab_clubs = st.tabs(["Equipe", "Etude Joueurs", "Ligue", "Clubs"])

with tab_team:
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

with tab_study:
    render_player_study_tab()

with tab_standings:
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

with tab_clubs:
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
