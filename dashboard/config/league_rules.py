from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueZoneConfig:
    champions_league: set[int]
    europa_league: set[int]
    conference_league: set[int]
    danger: set[int]
    relegation: set[int]
    note: str


LEAGUE_ZONE_CONFIG = {
    "LaLiga": LeagueZoneConfig(
        champions_league={1, 2, 3, 4},
        europa_league={5, 6},
        conference_league={7},
        danger={17},
        relegation={18, 19, 20},
        note="LaLiga zones use league-position guidance only. Domestic cup winners and UEFA rebalancing are not modeled here.",
    ),
    "Premier League": LeagueZoneConfig(
        champions_league={1, 2, 3, 4, 5},
        europa_league={6},
        conference_league={7},
        danger={17},
        relegation={18, 19, 20},
        note="Premier League zones use current as-stands access, including the extra Champions League spot.",
    ),
    "Bundesliga": LeagueZoneConfig(
        champions_league={1, 2, 3, 4, 5},
        europa_league={6},
        conference_league={7},
        danger={15, 16},
        relegation={17, 18},
        note="Bundesliga zones use current as-stands access, including the extra Champions League spot.",
    ),
    "Serie A": LeagueZoneConfig(
        champions_league={1, 2, 3, 4},
        europa_league={5, 6},
        conference_league={7},
        danger={17},
        relegation={18, 19, 20},
        note="Serie A zones use league-position guidance only. Domestic cup winners and UEFA rebalancing are not modeled here.",
    ),
    "Ligue 1": LeagueZoneConfig(
        champions_league={1, 2, 3},
        europa_league={4},
        conference_league={5},
        danger={15, 16},
        relegation={17, 18},
        note="Ligue 1 zones use league-position guidance only. Domestic cup winners and UEFA rebalancing are not modeled here.",
    ),
}


def get_zone_config(competition_name: str, team_count: int) -> LeagueZoneConfig:
    config = LEAGUE_ZONE_CONFIG.get(competition_name)
    if config is not None:
        return config

    return LeagueZoneConfig(
        champions_league={1, 2, 3, 4},
        europa_league={5},
        conference_league={6},
        danger={team_count - 3} if team_count >= 4 else set(),
        relegation=set(range(max(team_count - 2, 1), team_count + 1)) if team_count >= 4 else set(),
        note="Qualification zones use generic league-position guidance.",
    )
