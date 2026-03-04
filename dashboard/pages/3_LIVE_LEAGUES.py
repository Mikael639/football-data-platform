import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config.league_rules import get_zone_config
from data.dashboard_data import get_live_league_form, get_live_league_tables, is_european_competition_name
from ui.display import render_note_card, render_page_banner, render_section_heading
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="LIVE LEAGUES - Football Data Platform", layout="wide")


def _zone_for_position(position: int, competition_name: str, team_count: int) -> str:
    config = get_zone_config(competition_name, team_count)
    if position in config.champions_league:
        return "UCL"
    if position in config.europa_league:
        return "UEL"
    if position in config.conference_league:
        return "UECL"
    if position in config.danger:
        return "DANGER"
    if position in config.relegation:
        return "RELEGATION"
    return ""


def _row_class(zone: str) -> str:
    return {
        "UCL": "fdp-zone-ucl",
        "UEL": "fdp-zone-uel",
        "UECL": "fdp-zone-uecl",
        "DANGER": "fdp-zone-danger",
        "RELEGATION": "fdp-zone-relegation",
    }.get(zone, "")


def _badge_class(zone: str) -> str:
    return {
        "UCL": "fdp-pos-ucl",
        "UEL": "fdp-pos-uel",
        "UECL": "fdp-pos-uecl",
        "DANGER": "fdp-pos-danger",
        "RELEGATION": "fdp-pos-relegation",
    }.get(zone, "")


def _zone_label_class(zone: str) -> str:
    return {
        "UCL": "fdp-zone-label-ucl",
        "UEL": "fdp-zone-label-uel",
        "UECL": "fdp-zone-label-uecl",
        "DANGER": "fdp-zone-label-danger",
        "RELEGATION": "fdp-zone-label-relegation",
    }.get(zone, "")


def _zone_display_name(zone: str) -> str:
    return {
        "UCL": "Champions League",
        "UEL": "Europa League",
        "UECL": "Conference League",
        "DANGER": "Danger",
        "RELEGATION": "Relegation",
    }.get(zone, "")


def _trend_symbol(delta: object) -> str:
    if pd.isna(delta):
        return "="
    delta_value = int(delta)
    if delta_value > 0:
        return f"↑{delta_value}"
    if delta_value < 0:
        return f"↓{abs(delta_value)}"
    return "="


def _trend_class(delta: object) -> str:
    if pd.isna(delta):
        return "fdp-trend-flat"
    delta_value = int(delta)
    if delta_value > 0:
        return "fdp-trend-up"
    if delta_value < 0:
        return "fdp-trend-down"
    return "fdp-trend-flat"


def _embedded_table_css() -> str:
    return """
    <style>
      :root {
        --bg: #ffffff;
        --line: #e5ebf3;
        --ink: #16283d;
        --muted: #637a94;
        --ucl-bg: #ebf3ff;
        --ucl-accent: #0d63dd;
        --uel-bg: #ebfbf5;
        --uel-accent: #109b74;
        --uecl-bg: #f2edff;
        --uecl-accent: #7c3aed;
        --danger-bg: #fff6e6;
        --danger-accent: #bb7a00;
        --rel-bg: #feeeee;
        --rel-accent: #cb3a3a;
        --for: #118a57;
        --against: #d14343;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --bg: #111d31;
          --line: rgba(155, 184, 217, 0.14);
          --ink: #edf4ff;
          --muted: #9db5d0;
          --ucl-bg: rgba(13, 99, 221, 0.18);
          --ucl-accent: #7db1ff;
          --uel-bg: rgba(16, 155, 116, 0.18);
          --uel-accent: #67d7b2;
          --uecl-bg: rgba(124, 58, 237, 0.18);
          --uecl-accent: #b393ff;
          --danger-bg: rgba(187, 122, 0, 0.18);
          --danger-accent: #f2be5c;
          --rel-bg: rgba(203, 58, 58, 0.18);
          --rel-accent: #ff9b9b;
          --for: #46d68e;
          --against: #ff8d8d;
        }
      }

      body {
        margin: 0;
        background: transparent;
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
        color: var(--ink);
        color-scheme: light dark;
      }

      .fdp-standings-wrap {
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        background: var(--bg);
        box-shadow: 0 10px 24px rgba(17, 37, 62, 0.06);
      }

      .fdp-standings-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 14px;
      }

      .fdp-standings-table thead th {
        text-align: left;
        padding: 12px 10px;
        background: color-mix(in srgb, var(--bg) 84%, white 16%);
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--line);
      }

      .fdp-standings-table thead th:nth-child(1) { width: 54px; }
      .fdp-standings-table thead th:nth-child(2) { width: 58px; }
      .fdp-standings-table thead th:nth-child(3) { width: 250px; }
      .fdp-standings-table thead th:nth-child(4) { width: 120px; }
      .fdp-standings-table thead th:nth-child(5) { width: 58px; }
      .fdp-standings-table thead th:nth-child(6),
      .fdp-standings-table thead th:nth-child(7),
      .fdp-standings-table thead th:nth-child(8),
      .fdp-standings-table thead th:nth-child(9) { width: 42px; }
      .fdp-standings-table thead th:nth-child(10),
      .fdp-standings-table thead th:nth-child(11),
      .fdp-standings-table thead th:nth-child(12) { width: 52px; }
      .fdp-standings-table thead th:nth-child(13) { width: 150px; }

      .fdp-standings-table tbody td {
        padding: 11px 10px;
        border-bottom: 1px solid var(--line);
        vertical-align: middle;
        color: var(--ink);
      }

      .fdp-standings-table tbody tr:last-child td {
        border-bottom: none;
      }

      .fdp-zone-ucl {
        background: linear-gradient(90deg, rgba(13,99,221,0.06), transparent 24%);
        box-shadow: inset 4px 0 0 var(--ucl-accent);
      }

      .fdp-zone-uel {
        background: linear-gradient(90deg, rgba(16,155,116,0.06), transparent 24%);
        box-shadow: inset 4px 0 0 var(--uel-accent);
      }

      .fdp-zone-danger {
        background: linear-gradient(90deg, rgba(187,122,0,0.07), transparent 24%);
        box-shadow: inset 4px 0 0 var(--danger-accent);
      }

      .fdp-zone-uecl {
        background: linear-gradient(90deg, rgba(124,58,237,0.06), transparent 24%);
        box-shadow: inset 4px 0 0 var(--uecl-accent);
      }

      .fdp-zone-relegation {
        background: linear-gradient(90deg, rgba(203,58,58,0.07), transparent 24%);
        box-shadow: inset 4px 0 0 var(--rel-accent);
      }

      .fdp-pos-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 999px;
        background: color-mix(in srgb, var(--bg) 80%, white 20%);
        color: var(--ink);
        font-weight: 600;
        font-size: 13px;
      }

      .fdp-pos-ucl { background: var(--ucl-bg); color: var(--ucl-accent); }
      .fdp-pos-uel { background: var(--uel-bg); color: var(--uel-accent); }
      .fdp-pos-uecl { background: var(--uecl-bg); color: var(--uecl-accent); }
      .fdp-pos-danger { background: var(--danger-bg); color: var(--danger-accent); }
      .fdp-pos-relegation { background: var(--rel-bg); color: var(--rel-accent); }

      .fdp-trend {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 34px;
        height: 28px;
        border-radius: 999px;
        padding: 0 8px;
        font-size: 13px;
        font-weight: 600;
      }

      .fdp-trend-up {
        background: rgba(23,163,74,0.12);
        color: #15803d;
      }

      .fdp-trend-down {
        background: rgba(214,75,75,0.12);
        color: #c23b3b;
      }

      .fdp-trend-flat {
        background: rgba(148,163,184,0.16);
        color: color-mix(in srgb, var(--ink) 78%, #7b8ea5 22%);
      }

      .fdp-team-cell {
        display: flex;
        align-items: center;
        gap: 10px;
      }

      .fdp-team-crest {
        width: 24px;
        height: 24px;
        object-fit: contain;
        flex: 0 0 auto;
      }

      .fdp-team-fallback {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        background: color-mix(in srgb, var(--bg) 72%, white 28%);
        color: var(--ink);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 800;
        flex: 0 0 auto;
      }

      .fdp-team-name {
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .fdp-form-strip {
        display: flex;
        gap: 5px;
        align-items: center;
      }

      .fdp-form-pill {
        width: 20px;
        height: 20px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: #fff;
      }

      .fdp-form-win { background: #17a34a; }
      .fdp-form-draw { background: #d2a117; }
      .fdp-form-loss { background: #d64b4b; }
      .fdp-form-empty { background: #c8d3df; }

      .fdp-stat-for { color: var(--for); font-weight: 600; }
      .fdp-stat-against { color: var(--against); font-weight: 600; }
      .fdp-points { font-weight: 600; }

      .fdp-zone-label {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid transparent;
        white-space: nowrap;
      }

      .fdp-zone-label-ucl { background: var(--ucl-bg); color: var(--ucl-accent); border-color: #cfe0ff; }
      .fdp-zone-label-uel { background: var(--uel-bg); color: var(--uel-accent); border-color: #cbeee3; }
      .fdp-zone-label-uecl { background: var(--uecl-bg); color: var(--uecl-accent); border-color: #ddd0ff; }
      .fdp-zone-label-danger { background: var(--danger-bg); color: var(--danger-accent); border-color: #f1ddb0; }
      .fdp-zone-label-relegation { background: var(--rel-bg); color: var(--rel-accent); border-color: #f2c8c8; }
    </style>
    """


def _prepare_league_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.copy().sort_values(["position", "team_name"]).reset_index(drop=True)
    team_count = len(table)
    competition_name = str(table.iloc[0]["competition_name"])
    table["zone"] = table["position"].astype(int).map(
        lambda value: _zone_for_position(int(value), competition_name, team_count)
    )
    return table


def _crest_html(team_name: str, crest_url: str | None) -> str:
    if crest_url:
        return f'<img class="fdp-team-crest" src="{html.escape(str(crest_url))}" alt="{html.escape(team_name)} crest" />'
    initials = "".join(part[:1] for part in team_name.split()[:2]).upper() or "?"
    return f'<span class="fdp-team-fallback">{html.escape(initials)}</span>'


def _form_html(results: list[str]) -> str:
    pills: list[str] = []
    # Show the oldest result on the left and the most recent on the right.
    for result in list(results[:5])[::-1]:
        css_class = {
            "W": "fdp-form-win",
            "D": "fdp-form-draw",
            "L": "fdp-form-loss",
        }.get(result, "fdp-form-empty")
        pills.append(f'<span class="fdp-form-pill {css_class}">{html.escape(result)}</span>')
    while len(pills) < 5:
        pills.append('<span class="fdp-form-pill fdp-form-empty">-</span>')
    return f"<div class='fdp-form-strip'>{''.join(pills)}</div>"


def _render_zone_legend() -> None:
    st.markdown(
        """
        <div class="fdp-chip-row" style="margin-top:0;">
          <span class="fdp-zone-label fdp-zone-label-ucl">Champions League</span>
          <span class="fdp-zone-label fdp-zone-label-uel">Europa League</span>
          <span class="fdp-zone-label fdp-zone-label-uecl">Conference League</span>
          <span class="fdp-zone-label fdp-zone-label-danger">Danger</span>
          <span class="fdp-zone-label fdp-zone-label-relegation">Relegation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_rule_note(competition_name: str) -> None:
    st.caption(get_zone_config(competition_name, team_count=20).note)


def _render_standings_table(df: pd.DataFrame, recent_form: dict[int, list[str]]) -> None:
    if df.empty:
        st.info("Aucune table disponible pour cette ligue.")
        return

    rows_html: list[str] = []
    for _, row in df.iterrows():
        zone = str(row["zone"] or "")
        team_name = str(row["team_name"])
        row_class = _row_class(zone)
        badge_class = _badge_class(zone)
        zone_label_class = _zone_label_class(zone)
        zone_label = (
            f"<span class='fdp-zone-label {zone_label_class}'>{html.escape(_zone_display_name(zone))}</span>"
            if zone
            else ""
        )
        trend_html = f"<span class='fdp-trend {_trend_class(row.get('position_delta'))}'>{_trend_symbol(row.get('position_delta'))}</span>"
        crest_html = _crest_html(team_name, row.get("crest_url"))
        team_form_html = _form_html(recent_form.get(int(row["team_id"]), []))
        rows_html.append(
            f"""
            <tr class="{row_class}">
              <td><span class="fdp-pos-badge {badge_class}">{int(row["position"])}</span></td>
              <td>{trend_html}</td>
              <td>
                <div class="fdp-team-cell">
                  {crest_html}
                  <span class="fdp-team-name">{html.escape(team_name)}</span>
                </div>
              </td>
              <td>{team_form_html}</td>
              <td><span class="fdp-points">{int(row["points"])}</span></td>
              <td>{int(row["played_games"])}</td>
              <td>{int(row["won"])}</td>
              <td>{int(row["draw"])}</td>
              <td>{int(row["lost"])}</td>
              <td><span class="fdp-stat-for">{int(row["goals_for"])}</span></td>
              <td><span class="fdp-stat-against">{int(row["goals_against"])}</span></td>
              <td>{int(row["goal_difference"])}</td>
              <td>{zone_label}</td>
            </tr>
            """
        )

    table_html = f"""
    {_embedded_table_css()}
    <div class="fdp-standings-wrap">
      <table class="fdp-standings-table">
        <thead>
          <tr>
            <th>Pos</th>
            <th>Trend</th>
            <th>Equipe</th>
            <th>Form</th>
            <th>Pts</th>
            <th>MJ</th>
            <th>G</th>
            <th>N</th>
            <th>P</th>
            <th>BP</th>
            <th>BC</th>
            <th>Diff</th>
            <th>Zone</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    height = 120 + (len(df) * 50)
    components.html(table_html, height=min(max(height, 420), 1200), scrolling=True)


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "LIVE LEAGUES",
        "Vue comparateur multi-ligues: un onglet par championnat, basee sur le dernier classement disponible en base.",
        "Live Leagues.png",
    )
    render_note_card(
        "Contrairement a OVERVIEW, cette page ne sert pas a filtrer un perimetre equipe/date mais a lire la table courante de chaque ligue."
    )

    if st.button("Reload from DB"):
        st.cache_data.clear()
        st.rerun()

    league_tables = get_live_league_tables()
    league_tables = {
        competition_name: table
        for competition_name, table in league_tables.items()
        if not is_european_competition_name(competition_name)
    }
    if not league_tables:
        st.info(
            "Aucune ligue domestique disponible ici pour le moment. "
            "Les competitions UEFA sont visibles dans l'onglet EUROPE."
        )
        return

    if len(league_tables) == 1:
        st.warning(
            "Une seule competition est disponible en base pour l'instant. "
            "La page est prete pour plusieurs championnats des que le pipeline les charge."
        )

    league_names = list(league_tables.keys())
    tabs = st.tabs(league_names)
    for tab, competition_name in zip(tabs, league_names):
        with tab:
            table = _prepare_league_table(league_tables[competition_name])
            season = table.iloc[0]["season"]
            matchday = table.iloc[0]["matchday"]
            recent_form = get_live_league_form(int(table.iloc[0]["competition_id"]), int(season))
            render_section_heading("League Snapshot")
            top = st.columns(3)
            top[0].metric("Competition", competition_name)
            top[1].metric("Season", str(season))
            top[2].metric("Matchday", "-" if matchday is None else int(matchday))
            _render_zone_legend()
            _render_rule_note(competition_name)
            _render_standings_table(table, recent_form)


if __name__ == "__main__":
    main()
