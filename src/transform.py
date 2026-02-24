from __future__ import annotations
from typing import Dict, Any, List, Tuple, Set
from datetime import datetime

def transform(payload: Dict[str, Any]) -> Dict[str, Any]:
    fixtures = payload.get("fixtures", [])

    dates: Set[str] = set()
    teams: Dict[int, Dict[str, Any]] = {}
    competitions: Dict[int, Dict[str, Any]] = {}
    players: Dict[int, Dict[str, Any]] = {}

    matches: List[Dict[str, Any]] = []
    player_match_stats: List[Dict[str, Any]] = []

    for fx in fixtures:
        match_id = int(fx["match_id"])
        date_str = fx["date"]  # YYYY-MM-DD
        dates.add(date_str)

        comp_id = int(fx["competition_id"])
        competitions[comp_id] = {
            "competition_id": comp_id,
            "competition_name": fx.get("competition_name"),
            "country": None,
        }

        home = fx["home_team"]
        away = fx["away_team"]
        for t in (home, away):
            tid = int(t["team_id"])
            teams[tid] = {
                "team_id": tid,
                "team_name": t.get("team_name"),
                "country": t.get("country"),
            }

        score = fx.get("score", {})
        matches.append(
            {
                "match_id": match_id,
                "date_id": date_str,
                "competition_id": comp_id,
                "home_team_id": int(home["team_id"]),
                "away_team_id": int(away["team_id"]),
                "home_score": score.get("home"),
                "away_score": score.get("away"),
            }
        )

        for ps in fx.get("player_stats", []):
            p = ps["player"]
            pid = int(p["player_id"])
            players[pid] = {
                "player_id": pid,
                "full_name": p.get("full_name"),
                "position": p.get("position"),
                "nationality": p.get("nationality"),
                "birth_date": p.get("birth_date"),
                "team_id": int(p.get("team_id")) if p.get("team_id") is not None else None,
            }

            s = ps["stats"]
            player_match_stats.append(
                {
                    "match_id": match_id,
                    "player_id": pid,
                    "minutes": s.get("minutes"),
                    "goals": s.get("goals"),
                    "assists": s.get("assists"),
                    "shots": s.get("shots"),
                    "passes": s.get("passes"),
                    "pass_accuracy": s.get("pass_accuracy"),
                }
            )

    # build dim_date rows
    dim_dates = []
    for d in sorted(dates):
        dt = datetime.strptime(d, "%Y-%m-%d")
        dim_dates.append({"date_id": d, "year": dt.year, "month": dt.month, "day": dt.day})

    return {
        "dim_date": dim_dates,
        "dim_team": list(teams.values()),
        "dim_competition": list(competitions.values()),
        "dim_player": list(players.values()),
        "fact_match": matches,
        "fact_player_match_stats": player_match_stats,
    }

def count_loaded(transformed: Dict[str, Any]) -> int:
    # simple metric: total rows to load
    return sum(len(v) for v in transformed.values())