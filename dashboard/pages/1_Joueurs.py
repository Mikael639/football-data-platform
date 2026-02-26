import os
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Joueurs - Plateforme Data Football", layout="wide")

PAGE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = PAGE_DIR.parent
DEFAULT_PLAYER_PLACEHOLDER = DASHBOARD_DIR / "assets" / "player-placeholder.svg"
STUDY_FBREF_DIR = DASHBOARD_DIR.parent / "data" / "study" / "fbref"


def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "football_dw")
    user = os.getenv("DB_USER", "football")
    pwd = os.getenv("DB_PASSWORD", "football")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, pool_pre_ping=True)


def current_season_start_year() -> int:
    today = datetime.utcnow().date()
    return today.year if today.month >= 7 else today.year - 1


def current_season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_laliga_teams_live(competition_code: str, season_start_year: int):
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    base_url = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
    if not token:
        return None, "FOOTBALL_DATA_TOKEN manquant"
    try:
        response = requests.get(
            f"{base_url}/competitions/{competition_code}/teams",
            headers={"X-Auth-Token": token},
            params={"season": season_start_year},
            timeout=20,
        )
    except requests.RequestException as exc:
        return None, f"Echec requete API: {exc}"
    if response.status_code != 200:
        return None, f"API statut={response.status_code}"
    payload = response.json()
    rows = []
    for team in payload.get("teams") or []:
        if team.get("id") is None:
            continue
        rows.append({"team_id": int(team["id"]), "team_name": team.get("name") or team.get("shortName")})
    if not rows:
        return pd.DataFrame(columns=["team_id", "team_name"]), None
    return pd.DataFrame(rows).drop_duplicates(subset=["team_id"]).sort_values("team_name"), None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_live_team_squad(team_id: int):
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    base_url = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
    if not token:
        return None, "FOOTBALL_DATA_TOKEN manquant"
    try:
        response = requests.get(
            f"{base_url}/teams/{int(team_id)}",
            headers={"X-Auth-Token": token},
            timeout=20,
        )
    except requests.RequestException as exc:
        return None, f"Echec requete API: {exc}"
    if response.status_code != 200:
        return None, f"API statut={response.status_code}"
    payload = response.json()
    team = payload.get("team") or {}
    rows = []
    for player in payload.get("squad") or []:
        if player.get("id") is None:
            continue
        rows.append(
            {
                "player_id": int(player["id"]),
                "full_name": player.get("name"),
                "position": player.get("position"),
                "nationality": player.get("nationality"),
                "birth_date": player.get("dateOfBirth"),
                "team_id": int(team.get("id", team_id)),
                "team_name": team.get("name"),
            }
        )
    return pd.DataFrame(rows), None


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    txt = unicodedata.normalize("NFKD", str(value))
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    return " ".join(txt.lower().strip().split())


@st.cache_data(show_spinner=False, ttl=60)
def load_study_player_season_df():
    supabase_db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("STUDY_SUPABASE_DB_URL")
    backend = (os.getenv("FBREF_STUDY_BACKEND") or "local").strip().lower()
    if backend in {"supabase", "postgres"} and supabase_db_url:
        try:
            engine = create_engine(supabase_db_url, pool_pre_ping=True)
            df = pd.read_sql(text("SELECT * FROM public.study_fbref_player_season"), engine)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass
    csv_path = STUDY_FBREF_DIR / "player_season.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def build_player_palmares(player_name: str, study_season_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if study_season_df is None or study_season_df.empty:
        return pd.DataFrame(), {"gold": 0, "silver": 0, "bronze": 0, "total": 0}

    df = study_season_df.copy()
    if "player_name" not in df.columns or "season_start" not in df.columns:
        return pd.DataFrame(), {"gold": 0, "silver": 0, "bronze": 0, "total": 0}

    df["__player_norm"] = df["player_name"].astype(str).map(_normalize_name)
    target_norm = _normalize_name(player_name)
    player_rows = df[df["__player_norm"] == target_norm].copy()
    if player_rows.empty:
        return pd.DataFrame(), {"gold": 0, "silver": 0, "bronze": 0, "total": 0}

    metric_map = {
        "Buts": "goals_total",
        "Passes D": "assists_total",
        "G+A": "ga_total",
        "Buts (PK)": "pk_goals_total",
        "Buts (hors PK)": "goals_non_pk_total",
        "Jaunes": "yellow_cards_total",
        "Rouges": "red_cards_total",
    }
    rows = []
    for metric_label, metric_col in metric_map.items():
        if metric_col not in df.columns:
            continue
        base = df[["season_start", "player_name", "team_name", metric_col]].copy()
        base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce").fillna(0)
        base = base[base[metric_col] > 0].copy()
        if base.empty:
            continue
        base["rank_saison"] = base.groupby("season_start")[metric_col].rank(method="dense", ascending=False)
        top = base[base["rank_saison"] <= 3].copy()
        top["__player_norm"] = top["player_name"].astype(str).map(_normalize_name)
        top = top[top["__player_norm"] == target_norm].copy()
        if top.empty:
            continue
        for _, r in top.iterrows():
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(int(r["rank_saison"]), "")
            season_start = int(r["season_start"])
            rows.append(
                {
                    "Saison": f"{season_start}/{str(season_start + 1)[-2:]}",
                    "Medaille": medal,
                    "Categorie": metric_label,
                    "Valeur": int(r[metric_col]) if float(r[metric_col]).is_integer() else round(float(r[metric_col]), 2),
                    "Club": r.get("team_name"),
                    "Rang": int(r["rank_saison"]),
                    "season_sort": season_start,
                }
            )

    if not rows:
        return pd.DataFrame(), {"gold": 0, "silver": 0, "bronze": 0, "total": 0}

    out = pd.DataFrame(rows).sort_values(["season_sort", "Rang", "Categorie"], ascending=[False, True, True]).reset_index(drop=True)
    counts = {
        "gold": int((out["Rang"] == 1).sum()),
        "silver": int((out["Rang"] == 2).sum()),
        "bronze": int((out["Rang"] == 3).sum()),
        "total": int(len(out)),
    }
    return out.drop(columns=["season_sort"]), counts


engine = get_engine()
season_start_year = current_season_start_year()
competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")
teams_df, teams_live_err = fetch_laliga_teams_live(competition_code, season_start_year)
if teams_df is None or teams_df.empty:
    teams_df = pd.read_sql("SELECT team_id, team_name FROM dim_team ORDER BY team_name;", engine)

team_names = ["Tous les clubs"] + teams_df["team_name"].dropna().tolist()
incoming_team_name = st.session_state.get("players_selected_team_name")
if incoming_team_name in team_names:
    st.session_state["players_team_selector"] = incoming_team_name
elif "players_team_selector" not in st.session_state or st.session_state["players_team_selector"] not in team_names:
    st.session_state["players_team_selector"] = "Tous les clubs"

st.title("Joueurs")
st.caption(f"Saison courante: {current_season_label(season_start_year)}")
nav_col1, nav_col2 = st.columns([1, 4])
with nav_col1:
    st.page_link("app.py", label="Retour au dashboard")
with nav_col2:
    if teams_live_err:
        st.caption(f"Liste clubs en fallback local (API indisponible): {teams_live_err}")

selected_team_name = st.selectbox("Club", team_names, key="players_team_selector")
selected_team_id = None
if selected_team_name != "Tous les clubs":
    selected_team_id = int(
        teams_df.loc[teams_df["team_name"] == selected_team_name, "team_id"].iloc[0]
    )

players_query = text(
    """
SELECT
  p.player_id, p.full_name, p.position, p.nationality, p.birth_date,
  p.team_id, t.team_name
FROM dim_player p
LEFT JOIN dim_team t ON t.team_id = p.team_id
"""
)
df_players_all = pd.read_sql(players_query, engine)
if not df_players_all.empty:
    df_players_all = (
        df_players_all.sort_values(["team_name", "full_name"], ascending=[True, True])
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

players_scope_df = df_players_all.copy()
if selected_team_id is not None:
    live_players_df, live_err = fetch_live_team_squad(selected_team_id)
    if isinstance(live_players_df, pd.DataFrame) and not live_players_df.empty:
        players_scope_df = live_players_df.copy()
        st.caption("Source joueurs: API live (club selectionne)")
    else:
        players_scope_df = df_players_all[df_players_all["team_id"] == selected_team_id].copy()
        if live_err:
            st.caption(f"Source joueurs: base locale (API live indisponible: {live_err})")

if players_scope_df.empty:
    st.info("Aucun joueur trouve pour ce club.")
    st.stop()

if selected_team_id is not None:
    st.subheader(f"Effectif - {selected_team_name}")
    st.dataframe(
        players_scope_df[["full_name", "position", "nationality", "birth_date"]]
        .sort_values(["position", "full_name"]),
        use_container_width=True,
        hide_index=True,
    )

player_options = players_scope_df.copy()
player_options["player_label"] = (
    player_options["full_name"].fillna("Inconnu")
    + " | "
    + player_options["team_name"].fillna("Club inconnu")
    + " | "
    + player_options["player_id"].astype(str)
)
player_options = player_options.sort_values(["team_name", "full_name", "player_id"]).reset_index(drop=True)

choice = st.selectbox("Selectionner un joueur", player_options["player_label"].tolist(), key="players_page_player_select")
player = player_options[player_options["player_label"] == choice].iloc[0]

c1, c2 = st.columns([1, 2])
with c1:
    st.image(str(DEFAULT_PLAYER_PLACEHOLDER), use_column_width=True)
    st.caption("Icone joueur affichee.")

with c2:
    st.write(f"**Nom:** {player['full_name']}")
    st.write(f"**Poste:** {player['position']}")
    st.write(f"**Nationalite:** {player['nationality']}")
    st.write(f"**Club:** {player['team_name']}")
    st.write(f"**Date de naissance:** {player['birth_date']}")

study_season_df = load_study_player_season_df()
palmares_df, palmares_counts = build_player_palmares(str(player.get("full_name", "")), study_season_df)

st.divider()
st.subheader("Palmares du joueur (FBref - donnees d'etude)")
if palmares_df.empty:
    st.info("Aucun palmares Top 3 detecte pour ce joueur dans les donnees d'etude disponibles.")
else:
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Palmares total", palmares_counts["total"])
    p2.metric("🥇 Or", palmares_counts["gold"])
    p3.metric("🥈 Argent", palmares_counts["silver"])
    p4.metric("🥉 Bronze", palmares_counts["bronze"])
    st.dataframe(
        palmares_df[["Saison", "Medaille", "Categorie", "Valeur", "Club"]],
        use_container_width=True,
        hide_index=True,
    )
