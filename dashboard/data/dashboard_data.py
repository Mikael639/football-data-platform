import os
from datetime import date, datetime

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text


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
        return pd.DataFrame(
            columns=["player_id", "full_name", "position", "nationality", "birth_date", "team_id", "team_name"]
        ), None
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
        return pd.DataFrame(columns=["date_dt", "match_id", "venue", "opponent", "GF", "GA", "Result", "Points"])

    df = pd.DataFrame(rows).sort_values(["date_dt", "match_id"]).reset_index(drop=True)
    df["CumulativePoints"] = df["Points"].cumsum()
    return df


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
        out.append({"Team": team, **stats, "GD": stats["GF"] - stats["GA"]})
    return (
        pd.DataFrame(out)
        .sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
