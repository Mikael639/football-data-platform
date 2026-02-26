import os
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
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
PHOTO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}
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


def has_photo_url(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_image_bytes(url: str):
    try:
        response = requests.get(url, headers=PHOTO_HEADERS, timeout=12)
    except requests.RequestException:
        return None

    content_type = (response.headers.get("content-type") or "").lower()
    if response.status_code != 200 or not content_type.startswith("image/"):
        return None
    return response.content or None


def laliga_logo_source():
    # Use a local wordmark asset (new LaLiga branding style) for a stable UI.
    return str(DEFAULT_LALIGA_BADGE)


def render_team_hero(selected_team_name: str, season_label: str) -> None:
    left, right = st.columns([5.0, 1.4], vertical_alignment="center")
    with left:
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
    with right:
        st.image(laliga_logo_source(), use_column_width=True)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def wikipedia_thumbnail_url(player_name: str):
    if not player_name or not player_name.strip():
        return None

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{player_name} footballer",
        "gsrlimit": 1,
        "prop": "pageimages",
        "pithumbsize": 480,
        "format": "json",
    }
    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers={"User-Agent": PHOTO_HEADERS["User-Agent"]},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    pages = (payload.get("query") or {}).get("pages") or {}
    for page in pages.values():
        thumb = (page or {}).get("thumbnail") or {}
        src = thumb.get("source")
        if isinstance(src, str) and src.startswith("http"):
            return src
    return None


def player_image_source(player_name, photo_url):
    if has_photo_url(photo_url):
        img = fetch_image_bytes(photo_url.strip())
        if img:
            return img, "remote"

    wiki_url = wikipedia_thumbnail_url(str(player_name))
    if wiki_url:
        img = fetch_image_bytes(wiki_url)
        if img:
            return img, "wiki"

    if has_photo_url(photo_url):
        return str(DEFAULT_PLAYER_PLACEHOLDER), "failed"
    return str(DEFAULT_PLAYER_PLACEHOLDER), "missing"


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
                "photo_url": None,
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "player_id", "full_name", "position", "nationality", "birth_date", "team_id", "team_name", "photo_url"
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
                "photo_url": row.get("photo_url"),
                "team_id": int(row["team_id"]) if pd.notna(row.get("team_id")) else None,
            }
        )

    if not rows:
        return 0

    with engine_obj.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO dim_player (player_id, full_name, position, nationality, birth_date, photo_url, team_id)
                VALUES (:player_id, :full_name, :position, :nationality, :birth_date, :photo_url, :team_id)
                ON CONFLICT (player_id) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    position = EXCLUDED.position,
                    nationality = EXCLUDED.nationality,
                    birth_date = EXCLUDED.birth_date,
                    photo_url = COALESCE(dim_player.photo_url, EXCLUDED.photo_url),
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

st.title("Plateforme Data Football - LaLiga")
st.caption(f"Saison en cours utilisee partout dans le dashboard: {current_season_label(season_start_year)}")

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
  p.team_id, t.team_name, p.photo_url
FROM dim_player p
LEFT JOIN dim_team t ON t.team_id = p.team_id
"""
df_players_all = pd.read_sql(text(players_query), engine)

if not df_players_all.empty:
    df_players_all["photo_rank"] = df_players_all["photo_url"].apply(lambda v: 1 if has_photo_url(v) else 0)
    df_players_all = (
        df_players_all.sort_values(["team_name", "full_name", "photo_rank"], ascending=[True, True, False])
        .drop_duplicates(subset=["player_id"], keep="first")
        .drop(columns=["photo_rank"])
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

tab_team, tab_standings, tab_clubs = st.tabs(["Equipe", "Ligue", "Clubs"])

with tab_team:
    st.header("Indicateurs equipe (Niveau A)")
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
                        add_leader_star(
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
                    add_leader_star(
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
                add_leader_star(league_local_all_season[top_cols].rename(columns={"Team": "Equipe"}))
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
