from __future__ import annotations
from typing import Dict, Any, List, Set
from datetime import datetime


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float_ratio(value):
    try:
        if value is None:
            return 0.0
        v = float(value)
        # Clamp entre 0 et 1
        return max(0.0, min(v, 1.0))
    except Exception:
        return 0.0


def transform(payload: Dict[str, Any]) -> Dict[str, Any]:
    fixtures = payload.get("fixtures", [])

    dates: Set[str] = set()
    teams: Dict[int, Dict[str, Any]] = {}
    competitions: Dict[int, Dict[str, Any]] = {}
    players: Dict[int, Dict[str, Any]] = {}

    matches: List[Dict[str, Any]] = []
    player_match_stats: List[Dict[str, Any]] = []

    for fx in fixtures:
        match_id = _safe_int(fx.get("match_id"))
        date_str = fx.get("date")
        dates.add(date_str)

        # Competition
        comp_id = _safe_int(fx.get("competition_id"))
        competitions[comp_id] = {
            "competition_id": comp_id,
            "competition_name": fx.get("competition_name"),
            "country": None,
        }

        # Teams
        home = fx.get("home_team", {})
        away = fx.get("away_team", {})

        for t in (home, away):
            tid = _safe_int(t.get("team_id"))
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
                "home_team_id": _safe_int(home.get("team_id")),
                "away_team_id": _safe_int(away.get("team_id")),
                "home_score": _safe_int(score.get("home")),
                "away_score": _safe_int(score.get("away")),
            }
        )

        # Player stats
        for ps in fx.get("player_stats", []):
            p = ps.get("player", {})
            pid = _safe_int(p.get("player_id"))

            players[pid] = {
                "player_id": pid,
                "full_name": p.get("full_name"),
                "position": p.get("position"),
                "nationality": p.get("nationality"),
                "birth_date": p.get("birth_date"),
                "team_id": _safe_int(p.get("team_id")),
            }

            s = ps.get("stats", {})

            minutes = _safe_int(s.get("minutes"))
            if minutes > 130:
                raise ValueError(f"Invalid minutes value detected: {minutes}")

            player_match_stats.append(
                {
                    "match_id": match_id,
                    "player_id": pid,
                    "minutes": minutes,
                    "goals": _safe_int(s.get("goals")),
                    "assists": _safe_int(s.get("assists")),
                    "shots": _safe_int(s.get("shots")),
                    "passes": _safe_int(s.get("passes")),
                    "pass_accuracy": _safe_float_ratio(s.get("pass_accuracy")),
                }
            )

    # dim_date
    dim_dates = []
    for d in sorted(dates):
        dt = datetime.strptime(d, "%Y-%m-%d")
        dim_dates.append(
            {
                "date_id": d,
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
            }
        )

    return {
        "dim_date": dim_dates,
        "dim_team": list(teams.values()),
        "dim_competition": list(competitions.values()),
        "dim_player": list(players.values()),
        "fact_match": matches,
        "fact_player_match_stats": player_match_stats,
    }


def count_loaded(transformed: Dict[str, Any]) -> int:
    return sum(len(v) for v in transformed.values())