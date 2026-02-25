import os
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Football Data Platform", layout="wide")


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


engine = get_engine()

st.title("⚽ Football Data Platform — LaLiga (Current Season)")

# -----------------------
# Team selector (from DB)
# -----------------------
teams_df = pd.read_sql("SELECT team_id, team_name FROM dim_team ORDER BY team_name;", engine)
team_names = ["All teams"] + teams_df["team_name"].tolist()

selected_team_name = st.selectbox("Filter by team", team_names, index=team_names.index("All teams"))
selected_team_id = None
if selected_team_name != "All teams":
    selected_team_id = int(teams_df.loc[teams_df["team_name"] == selected_team_name, "team_id"].iloc[0])

st.divider()

# -----------------------
# TEAM KPIs (Level A)
# -----------------------
st.header("Team KPIs (Level A)")

where_clause = ""
params = {}
if selected_team_id is not None:
    where_clause = "WHERE m.home_team_id = :tid OR m.away_team_id = :tid"
    params = {"tid": selected_team_id}

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
{where_clause}
ORDER BY m.date_id DESC NULLS LAST, m.match_id DESC;
"""
df_matches = pd.read_sql(text(matches_query), engine, params=params)

if df_matches.empty:
    st.warning("No matches found in the warehouse for this filter yet.")
    st.stop()

# Convert date_id to datetime for upcoming logic
df_matches["date_dt"] = pd.to_datetime(df_matches["date_id"], errors="coerce")

def compute_team_kpis(df: pd.DataFrame, team_id: int):
    played = wins = draws = losses = gf = ga = 0
    for _, r in df.iterrows():
        hs = r["home_score"]
        aw = r["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue

        played += 1
        if int(r["home_team_id"]) == team_id:
            gf += int(hs); ga += int(aw)
            if hs > aw: wins += 1
            elif hs == aw: draws += 1
            else: losses += 1
        else:
            gf += int(aw); ga += int(hs)
            if aw > hs: wins += 1
            elif aw == hs: draws += 1
            else: losses += 1

    points = wins * 3 + draws
    return played, wins, draws, losses, gf, ga, points


col1, col2, col3, col4 = st.columns(4)

if selected_team_id is None:
    total_matches = df_matches["match_id"].nunique()
    played_matches = df_matches.dropna(subset=["home_score", "away_score"])["match_id"].nunique()
    upcoming_matches = total_matches - played_matches

    col1.metric("Matches (total)", int(total_matches))
    col2.metric("Played (with score)", int(played_matches))
    col3.metric("Upcoming (in window)", int(upcoming_matches))
    col4.metric("Teams in DB", int(teams_df["team_id"].nunique()))
else:
    played, wins, draws, losses, gf, ga, points = compute_team_kpis(df_matches, selected_team_id)
    col1.metric("Played", played)
    col2.metric("Points", points)
    col3.metric("GF / GA", f"{gf} / {ga}")
    col4.metric("W - D - L", f"{wins}-{draws}-{losses}")

st.subheader("Recent matches")
df_played = df_matches.dropna(subset=["home_score", "away_score"]).copy()
st.dataframe(df_played.head(15), use_container_width=True)

st.subheader("Upcoming matches (date in the future OR missing score)")
today = pd.Timestamp.now().normalize()
df_upcoming = df_matches[
    (df_matches["date_dt"] >= today)
    | (df_matches["home_score"].isna())
    | (df_matches["away_score"].isna())
].sort_values("date_dt", ascending=True)

st.dataframe(df_upcoming.head(15), use_container_width=True)

st.divider()

# -----------------------
# Standings (current season) - fetched live from football-data.org
# -----------------------
st.header("LaLiga Standings (Current Season)")

token = os.getenv("FOOTBALL_DATA_TOKEN")
base_url = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
competition_code = os.getenv("FOOTBALL_DATA_COMPETITION", "PD")

season_env = os.getenv("FOOTBALL_DATA_SEASON")
season = int(season_env) if season_env else current_season_start_year_dash()

if not token:
    st.info("FOOTBALL_DATA_TOKEN is missing in the dashboard container environment.")
else:
    r = requests.get(
        f"{base_url}/competitions/{competition_code}/standings",
        headers={"X-Auth-Token": token},
        params={"season": season},
        timeout=30,
    )

    if r.status_code != 200:
        st.warning(f"Standings not available (status={r.status_code}).")
    else:
        data = r.json()
        table = None
        for s in data.get("standings", []):
            if s.get("type") == "TOTAL":
                table = s.get("table", [])
                break

        if not table:
            st.info("No TOTAL standings table found.")
        else:
            df = pd.DataFrame([{
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
            } for row in table])

            st.dataframe(df, use_container_width=True)

            rm_row = df[df["Team"].str.contains("Real Madrid", na=False)]
            if not rm_row.empty:
                st.success(f"Real Madrid position: {int(rm_row.iloc[0]['Pos'])} | Points: {int(rm_row.iloc[0]['Pts'])}")

st.divider()

# -----------------------
# Players (Level A: no per-match player stats)
# -----------------------
st.header("Players (Level A)")

players_query = """
SELECT p.player_id, p.full_name, p.position, p.nationality, t.team_name, p.photo_url
FROM dim_player p
LEFT JOIN dim_team t ON t.team_id = p.team_id
"""
if selected_team_id is not None:
    players_query += " WHERE p.team_id = :tid"
    df_players = pd.read_sql(text(players_query), engine, params={"tid": selected_team_id})
else:
    df_players = pd.read_sql(text(players_query), engine)

if df_players.empty:
    st.info("No players found for this filter.")
else:
    choice = st.selectbox("Select a player", df_players["full_name"].sort_values().tolist())
    p = df_players[df_players["full_name"] == choice].iloc[0]

    c1, c2 = st.columns([1, 2])
    with c1:
        if p["photo_url"]:
            st.image(p["photo_url"], use_container_width=True)
        else:
            st.info("No photo yet (we will enrich via JSON mapping, then Wikidata).")

    with c2:
        st.write(f"**Name:** {p['full_name']}")
        st.write(f"**Position:** {p['position']}")
        st.write(f"**Nationality:** {p['nationality']}")
        st.write(f"**Team:** {p['team_name']}")
        st.caption("Player match stats (minutes/goals/assists) will come in Level B with a lineup/stats provider.")