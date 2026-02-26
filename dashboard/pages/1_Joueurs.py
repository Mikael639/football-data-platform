import os
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
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
PHOTO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


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
        src = ((page or {}).get("thumbnail") or {}).get("source")
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
    return str(DEFAULT_PLAYER_PLACEHOLDER), "failed" if has_photo_url(photo_url) else "missing"


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
                "photo_url": None,
            }
        )
    return pd.DataFrame(rows), None


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
  p.team_id, t.team_name, p.photo_url
FROM dim_player p
LEFT JOIN dim_team t ON t.team_id = p.team_id
"""
)
df_players_all = pd.read_sql(players_query, engine)
if not df_players_all.empty:
    df_players_all["photo_rank"] = df_players_all["photo_url"].apply(lambda v: 1 if has_photo_url(v) else 0)
    df_players_all = (
        df_players_all.sort_values(["team_name", "full_name", "photo_rank"], ascending=[True, True, False])
        .drop_duplicates(subset=["player_id"], keep="first")
        .drop(columns=["photo_rank"])
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
    img_source, img_status = player_image_source(player["full_name"], player["photo_url"])
    st.image(img_source, use_column_width=True)
    if img_status == "wiki":
        st.info("Photo chargee via le fallback Wikipedia.")
    elif img_status != "remote":
        st.info("Photo indisponible, mannequin affiche.")

with c2:
    st.write(f"**Nom:** {player['full_name']}")
    st.write(f"**Poste:** {player['position']}")
    st.write(f"**Nationalite:** {player['nationality']}")
    st.write(f"**Club:** {player['team_name']}")
    st.write(f"**Date de naissance:** {player['birth_date']}")
