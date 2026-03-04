from __future__ import annotations

import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data.dashboard_data import (
    get_competitions,
    get_current_standings,
    get_european_competitions,
    get_matches,
    get_seasons,
    split_recent_and_upcoming_matches,
)
from ui.display import render_note_card, render_page_banner, render_section_heading
from ui.styles import inject_dashboard_styles

st.set_page_config(page_title="EUROPE - Football Data Platform", layout="wide")


STAGE_LABELS = {
    "QUALIFICATION": "Qualification",
    "QUALIFICATION_ROUND_1": "Qualification R1",
    "QUALIFICATION_ROUND_2": "Qualification R2",
    "QUALIFICATION_ROUND_3": "Qualification R3",
    "PLAYOFF_ROUND_1": "Playoff R1",
    "PLAYOFF_ROUND_2": "Playoff R2",
    "PLAYOFFS": "Playoffs",
    "LEAGUE_STAGE": "League stage",
    "GROUP_STAGE": "Group stage",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-finals",
    "SEMI_FINALS": "Semi-finals",
    "THIRD_PLACE": "Third place",
    "FINAL": "Final",
}


def _format_stage(value: object) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return "Unknown stage"
    return STAGE_LABELS.get(text_value, text_value.replace("_", " ").title())


def _phase_label(row: pd.Series) -> str:
    parts: list[str] = []
    stage = _format_stage(row.get("stage"))
    group_name = str(row.get("group_name") or "").strip()
    if stage and stage != "Unknown stage":
        parts.append(stage)
    if group_name:
        parts.append(group_name)
    if not parts and pd.notna(row.get("matchday")):
        parts.append(f"Matchday {int(row['matchday'])}")
    return " | ".join(parts) if parts else "Schedule"


def _format_match_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.copy()
    kickoff = (
        pd.to_datetime(table["kickoff_utc"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    fallback = pd.to_datetime(table["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    table["kickoff"] = kickoff.fillna(fallback).fillna("Unknown")
    table["phase"] = table.apply(_phase_label, axis=1)
    table["score"] = table.apply(
        lambda row: "-" if pd.isna(row["home_score"]) or pd.isna(row["away_score"]) else f"{int(row['home_score'])}-{int(row['away_score'])}",
        axis=1,
    )
    table["status"] = table["status"].fillna("UNKNOWN")
    table["matchday"] = table["matchday"].fillna("--")
    return table[["kickoff", "phase", "status", "matchday", "home_team", "score", "away_team"]]


def _format_standings_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.rename(
        columns={
            "team_name": "Equipe",
            "position": "Pos",
            "points": "Pts",
            "played_games": "MJ",
            "won": "G",
            "draw": "N",
            "lost": "P",
            "goals_for": "BP",
            "goals_against": "BC",
            "goal_difference": "Diff",
        }
    )[["Pos", "Equipe", "Pts", "MJ", "G", "N", "P", "BP", "BC", "Diff"]]


def _standings_embedded_css() -> str:
    return """
    <style>
      :root {
        --bg: #ffffff;
        --line: #e5ebf3;
        --ink: #16283d;
        --muted: #637a94;
        --top8-bg: rgba(13, 99, 221, 0.08);
        --top8-accent: #0d63dd;
        --playoff-bg: rgba(16, 155, 116, 0.08);
        --playoff-accent: #109b74;
        --elim-bg: rgba(203, 58, 58, 0.08);
        --elim-accent: #cb3a3a;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --bg: #101a2d;
          --line: rgba(155, 184, 217, 0.14);
          --ink: #edf4ff;
          --muted: #9db5d0;
          --top8-bg: rgba(13, 99, 221, 0.2);
          --top8-accent: #7db1ff;
          --playoff-bg: rgba(16, 155, 116, 0.2);
          --playoff-accent: #67d7b2;
          --elim-bg: rgba(203, 58, 58, 0.2);
          --elim-accent: #ff9b9b;
        }
      }

      body {
        margin: 0;
        background: transparent;
        color: var(--ink);
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      }

      .fdp-europe-wrap {
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        background: var(--bg);
        box-shadow: 0 10px 24px rgba(17, 37, 62, 0.06);
      }

      .fdp-europe-legend {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        padding: 12px;
        border-bottom: 1px solid var(--line);
      }

      .fdp-europe-chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 700;
      }

      .fdp-chip-top8 { background: var(--top8-bg); color: var(--top8-accent); }
      .fdp-chip-playoff { background: var(--playoff-bg); color: var(--playoff-accent); }
      .fdp-chip-elim { background: var(--elim-bg); color: var(--elim-accent); }

      .fdp-europe-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 14px;
      }

      .fdp-europe-table thead th {
        text-align: left;
        padding: 12px 10px;
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--line);
      }

      .fdp-europe-table thead th:nth-child(1) { width: 60px; }
      .fdp-europe-table thead th:nth-child(2) { width: 280px; }
      .fdp-europe-table thead th:nth-child(3) { width: 65px; }
      .fdp-europe-table thead th:nth-child(4) { width: 65px; }
      .fdp-europe-table thead th:nth-child(5),
      .fdp-europe-table thead th:nth-child(6),
      .fdp-europe-table thead th:nth-child(7) { width: 55px; }
      .fdp-europe-table thead th:nth-child(8),
      .fdp-europe-table thead th:nth-child(9),
      .fdp-europe-table thead th:nth-child(10) { width: 65px; }

      .fdp-europe-table tbody td {
        padding: 11px 10px;
        border-bottom: 1px solid var(--line);
        vertical-align: middle;
      }

      .fdp-europe-table tbody tr:last-child td { border-bottom: none; }

      .fdp-row-top8 {
        background: linear-gradient(90deg, var(--top8-bg), transparent 30%);
        box-shadow: inset 4px 0 0 var(--top8-accent);
      }

      .fdp-row-playoff {
        background: linear-gradient(90deg, var(--playoff-bg), transparent 30%);
        box-shadow: inset 4px 0 0 var(--playoff-accent);
      }

      .fdp-row-elim {
        background: linear-gradient(90deg, var(--elim-bg), transparent 30%);
        box-shadow: inset 4px 0 0 var(--elim-accent);
      }

      .fdp-pos-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 13px;
        background: color-mix(in srgb, var(--bg) 80%, white 20%);
      }

      .fdp-pos-top8 { color: var(--top8-accent); background: var(--top8-bg); }
      .fdp-pos-playoff { color: var(--playoff-accent); background: var(--playoff-bg); }
      .fdp-pos-elim { color: var(--elim-accent); background: var(--elim-bg); }

      .fdp-team-cell {
        display: flex;
        align-items: center;
        gap: 10px;
      }

      .fdp-team-name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 600;
      }

      .fdp-crest {
        width: 24px;
        height: 24px;
        object-fit: contain;
        flex: 0 0 auto;
      }

      .fdp-crest-fallback {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 800;
        background: color-mix(in srgb, var(--bg) 72%, white 28%);
      }

      .fdp-num { font-weight: 600; }
      .fdp-diff-pos { color: #15964e; font-weight: 700; }
      .fdp-diff-neg { color: #cb3a3a; font-weight: 700; }
      .fdp-diff-zero { color: var(--muted); font-weight: 700; }
    </style>
    """


def _row_tier(position: int) -> str:
    if position <= 8:
        return "top8"
    if position <= 24:
        return "playoff"
    return "elim"


def _crest_html(team_name: str, crest_url: object) -> str:
    if crest_url and str(crest_url).strip():
        return f'<img class="fdp-crest" src="{html.escape(str(crest_url))}" alt="{html.escape(team_name)} crest" />'
    initials = "".join(part[:1] for part in team_name.split()[:2]).upper() or "?"
    return f'<span class="fdp-crest-fallback">{html.escape(initials)}</span>'


def _diff_class(diff: int) -> str:
    if diff > 0:
        return "fdp-diff-pos"
    if diff < 0:
        return "fdp-diff-neg"
    return "fdp-diff-zero"


def _render_standings_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Aucun classement disponible pour cette competition et cette saison.")
        return

    table = df.copy().sort_values(["position", "team_name"]).reset_index(drop=True)
    rows_html: list[str] = []

    for _, row in table.iterrows():
        position = int(row["position"])
        team_name = str(row["team_name"])
        tier = _row_tier(position)
        crest = _crest_html(team_name, row.get("crest_url"))
        diff = int(row["goal_difference"])
        diff_prefix = "+" if diff > 0 else ""

        rows_html.append(
            f"""
            <tr class="fdp-row-{tier}">
              <td><span class="fdp-pos-badge fdp-pos-{tier}">{position}</span></td>
              <td>
                <div class="fdp-team-cell">
                  {crest}
                  <span class="fdp-team-name">{html.escape(team_name)}</span>
                </div>
              </td>
              <td class="fdp-num">{int(row["points"])}</td>
              <td>{int(row["played_games"])}</td>
              <td>{int(row["won"])}</td>
              <td>{int(row["draw"])}</td>
              <td>{int(row["lost"])}</td>
              <td>{int(row["goals_for"])}</td>
              <td>{int(row["goals_against"])}</td>
              <td class="{_diff_class(diff)}">{diff_prefix}{diff}</td>
            </tr>
            """
        )

    table_html = f"""
    {_standings_embedded_css()}
    <div class="fdp-europe-wrap">
      <div class="fdp-europe-legend">
        <span class="fdp-europe-chip fdp-chip-top8">Top 8: qualification directe</span>
        <span class="fdp-europe-chip fdp-chip-playoff">9-24: playoffs</span>
        <span class="fdp-europe-chip fdp-chip-elim">25-36: elimines</span>
      </div>
      <table class="fdp-europe-table">
        <thead>
          <tr>
            <th>Pos</th>
            <th>Equipe</th>
            <th>Pts</th>
            <th>MJ</th>
            <th>G</th>
            <th>N</th>
            <th>P</th>
            <th>BP</th>
            <th>BC</th>
            <th>Diff</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    height = 140 + (len(table.index) * 48)
    components.html(table_html, height=min(max(height, 420), 1400), scrolling=True)


def _schedule_embedded_css() -> str:
    return """
    <style>
      :root {
        --bg: #ffffff;
        --line: #e5ebf3;
        --ink: #16283d;
        --muted: #637a94;
        --head: #f7faff;
        --live-bg: rgba(218, 74, 74, 0.14);
        --live-fg: #b32525;
        --done-bg: rgba(16, 155, 116, 0.16);
        --done-fg: #0e785a;
        --next-bg: rgba(13, 99, 221, 0.14);
        --next-fg: #0d63dd;
        --other-bg: rgba(98, 122, 148, 0.16);
        --other-fg: #3f5874;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --bg: #101a2d;
          --line: rgba(155, 184, 217, 0.14);
          --ink: #edf4ff;
          --muted: #9db5d0;
          --head: #0f1f36;
          --live-bg: rgba(255, 109, 109, 0.24);
          --live-fg: #ffaeae;
          --done-bg: rgba(89, 214, 176, 0.18);
          --done-fg: #89e9cc;
          --next-bg: rgba(111, 173, 255, 0.2);
          --next-fg: #b7d7ff;
          --other-bg: rgba(157, 181, 208, 0.2);
          --other-fg: #cbdcf0;
        }
      }

      body {
        margin: 0;
        background: transparent;
        color: var(--ink);
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      }

      .fdp-sched-wrap {
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        background: var(--bg);
      }

      .fdp-sched-head {
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        background: var(--head);
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
        color: var(--muted);
      }

      .fdp-sched-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 13px;
      }

      .fdp-sched-table thead th {
        text-align: left;
        padding: 10px 8px;
        border-bottom: 1px solid var(--line);
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .fdp-sched-table thead th:nth-child(1) { width: 112px; }
      .fdp-sched-table thead th:nth-child(2) { width: 150px; }
      .fdp-sched-table thead th:nth-child(3) { width: 88px; }
      .fdp-sched-table thead th:nth-child(4) { width: 48px; text-align: center; }
      .fdp-sched-table thead th:nth-child(5) { width: 132px; }
      .fdp-sched-table thead th:nth-child(6) { width: 54px; text-align: center; }
      .fdp-sched-table thead th:nth-child(7) { width: 132px; }

      .fdp-sched-table tbody td {
        padding: 10px 8px;
        border-bottom: 1px solid var(--line);
        vertical-align: middle;
      }

      .fdp-sched-table tbody tr:last-child td { border-bottom: none; }
      .fdp-sched-day { text-align: center; font-weight: 700; }
      .fdp-sched-score { text-align: center; font-weight: 700; }
      .fdp-sched-team {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 600;
      }
      .fdp-sched-phase {
        color: var(--muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .fdp-status {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
      }
      .fdp-status-live { background: var(--live-bg); color: var(--live-fg); }
      .fdp-status-done { background: var(--done-bg); color: var(--done-fg); }
      .fdp-status-next { background: var(--next-bg); color: var(--next-fg); }
      .fdp-status-other { background: var(--other-bg); color: var(--other-fg); }
    </style>
    """


def _status_badge(status: object) -> tuple[str, str]:
    raw = str(status or "").strip().upper()
    if raw in {"IN_PLAY", "PAUSED", "LIVE"}:
        return ("En direct", "fdp-status-live")
    if raw in {"FINISHED", "AFTER_EXTRA_TIME", "PENALTY_SHOOTOUT", "AWARDED"}:
        return ("Termine", "fdp-status-done")
    if raw in {"SCHEDULED", "TIMED"}:
        return ("A venir", "fdp-status-next")
    if raw == "POSTPONED":
        return ("Reporte", "fdp-status-other")
    if raw == "CANCELLED":
        return ("Annule", "fdp-status-other")
    if raw == "SUSPENDED":
        return ("Suspendu", "fdp-status-other")
    if raw:
        return (raw.replace("_", " "), "fdp-status-other")
    return ("Inconnu", "fdp-status-other")


def _render_schedule_table(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("Aucune donnee de match disponible.")
        return

    table = _format_match_table(df)
    rows_html: list[str] = []
    for _, row in table.iterrows():
        label, badge_cls = _status_badge(row.get("status"))
        matchday = str(row.get("matchday") if pd.notna(row.get("matchday")) else "--")
        rows_html.append(
            f"""
            <tr>
              <td>{html.escape(str(row.get("kickoff") or "-"))}</td>
              <td class="fdp-sched-phase">{html.escape(str(row.get("phase") or "-"))}</td>
              <td><span class="fdp-status {badge_cls}">{html.escape(label)}</span></td>
              <td class="fdp-sched-day">{html.escape(matchday)}</td>
              <td class="fdp-sched-team">{html.escape(str(row.get("home_team") or "-"))}</td>
              <td class="fdp-sched-score">{html.escape(str(row.get("score") or "-"))}</td>
              <td class="fdp-sched-team">{html.escape(str(row.get("away_team") or "-"))}</td>
            </tr>
            """
        )

    table_html = f"""
    {_schedule_embedded_css()}
    <div class="fdp-sched-wrap">
      <div class="fdp-sched-head">{html.escape(title)}</div>
      <table class="fdp-sched-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Phase</th>
            <th>Statut</th>
            <th>J</th>
            <th>Domicile</th>
            <th>Score</th>
            <th>Exterieur</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    height = 102 + (len(table.index) * 44)
    components.html(table_html, height=min(max(height, 250), 760), scrolling=True)


def _phase_embedded_css() -> str:
    return """
    <style>
      :root {
        --bg: #ffffff;
        --line: #e5ebf3;
        --ink: #16283d;
        --muted: #637a94;
        --head: #f7faff;
        --bar: #0d63dd;
        --bar-bg: rgba(13, 99, 221, 0.14);
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --bg: #101a2d;
          --line: rgba(155, 184, 217, 0.14);
          --ink: #edf4ff;
          --muted: #9db5d0;
          --head: #0f1f36;
          --bar: #7db1ff;
          --bar-bg: rgba(111, 173, 255, 0.18);
        }
      }

      body {
        margin: 0;
        background: transparent;
        color: var(--ink);
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      }

      .fdp-phase-wrap {
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        background: var(--bg);
      }

      .fdp-phase-head {
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        background: var(--head);
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
        color: var(--muted);
      }

      .fdp-phase-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 13px;
      }

      .fdp-phase-table thead th {
        text-align: left;
        padding: 10px 9px;
        border-bottom: 1px solid var(--line);
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .fdp-phase-table tbody td {
        padding: 10px 9px;
        border-bottom: 1px solid var(--line);
        vertical-align: middle;
      }

      .fdp-phase-table tbody tr:last-child td { border-bottom: none; }

      .fdp-phase-table thead th:nth-child(1) { width: 44%; }
      .fdp-phase-table thead th:nth-child(2) { width: 23%; }
      .fdp-phase-table thead th:nth-child(3) { width: 11%; text-align: right; }
      .fdp-phase-table thead th:nth-child(4) { width: 22%; }

      .fdp-phase-name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 600;
      }

      .fdp-phase-count {
        text-align: right;
        font-weight: 700;
      }

      .fdp-volume {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: var(--bar-bg);
        overflow: hidden;
      }

      .fdp-volume > span {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, var(--bar), color-mix(in srgb, var(--bar) 65%, white 35%));
      }
    </style>
    """


def _render_phase_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Pas assez de donnees pour construire une vue par phase.")
        return

    table = df.copy()
    table["Matches"] = pd.to_numeric(table["Matches"], errors="coerce").fillna(0).astype(int)
    table = table.sort_values(["Matches", "Phase", "Status"], ascending=[False, True, True]).reset_index(drop=True)
    max_matches = max(int(table["Matches"].max()), 1)

    rows_html: list[str] = []
    for _, row in table.iterrows():
        label, badge_cls = _status_badge(row.get("Status"))
        matches = int(row["Matches"])
        width = int(round((matches / max_matches) * 100))
        rows_html.append(
            f"""
            <tr>
              <td class="fdp-phase-name">{html.escape(str(row.get("Phase") or "-"))}</td>
              <td><span class="fdp-status {badge_cls}">{html.escape(label)}</span></td>
              <td class="fdp-phase-count">{matches}</td>
              <td><div class="fdp-volume"><span style="width:{width}%"></span></div></td>
            </tr>
            """
        )

    table_html = f"""
    {_schedule_embedded_css()}
    {_phase_embedded_css()}
    <div class="fdp-phase-wrap">
      <div class="fdp-phase-head">Lecture par phase</div>
      <table class="fdp-phase-table">
        <thead>
          <tr>
            <th>Phase</th>
            <th>Statut</th>
            <th>Matchs</th>
            <th>Volume relatif</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    height = 110 + (len(table.index) * 44)
    components.html(table_html, height=min(max(height, 250), 760), scrolling=True)


def _phase_summary(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    summary = matches.copy()
    summary["phase"] = summary.apply(_phase_label, axis=1)
    out = (
        summary.groupby(["phase", "status"], dropna=False, as_index=False)
        .agg(matches=("match_id", "count"))
        .sort_values(["phase", "status"])
        .reset_index(drop=True)
    )
    return out.rename(columns={"phase": "Phase", "status": "Status", "matches": "Matches"})


def main() -> None:
    inject_dashboard_styles()
    render_page_banner(
        "EUROPE",
        "Suivi dedie des competitions UEFA: classements quand ils existent, calendrier et lecture par phase.",
    )
    render_note_card(
        "Les competitions UEFA sont suivies a part des ligues domestiques. "
        "Quand aucun classement n est expose par la source, la page reste utile via les phases et le calendrier."
    )

    competitions = get_european_competitions()
    if competitions.empty:
        available = get_competitions()
        if available.empty:
            st.info("Aucune competition disponible en base. Lance la pipeline pour charger les competitions UEFA.")
        else:
            st.info("Aucune competition UEFA n est encore chargee. Verifie LIVE_COMPETITION_CODES puis relance la pipeline.")
        return

    competition_labels = competitions["competition_name"].astype(str).tolist()
    competition_map = {
        str(row["competition_name"]): int(row["competition_id"])
        for _, row in competitions.dropna(subset=["competition_id"]).iterrows()
    }

    default_competition = competition_labels[0]
    if "europe_competition" not in st.session_state or st.session_state["europe_competition"] not in competition_labels:
        st.session_state["europe_competition"] = default_competition
    selected_competition = st.selectbox("Competition UEFA", competition_labels, key="europe_competition")
    competition_id = competition_map[selected_competition]

    seasons = get_seasons(competition_id)
    season_labels = ["Derniere saison"] + seasons["season"].astype(str).tolist()
    if "europe_season" not in st.session_state or st.session_state["europe_season"] not in season_labels:
        st.session_state["europe_season"] = season_labels[0]
    selected_season = st.selectbox("Saison", season_labels, key="europe_season")
    season = None if selected_season == "Derniere saison" else selected_season

    matches = get_matches(competition_id=competition_id, season=season)
    recent_matches, upcoming_matches = split_recent_and_upcoming_matches(matches, recent_limit=8, upcoming_limit=8)
    standings = get_current_standings(competition_id=competition_id, season=season)

    render_section_heading("Classement UEFA", "Affiche le dernier snapshot quand la source fournit un classement.")
    _render_standings_table(standings)

    render_section_heading("Calendrier", "Derniers matchs joues et prochaines affiches UEFA pour la competition selectionnee.")
    left, right = st.columns(2)
    with left:
        _render_schedule_table(recent_matches, "Derniers 8 matchs")
    with right:
        _render_schedule_table(upcoming_matches, "Prochains 8 matchs")

    render_section_heading("Lecture par phase")
    phase_summary = _phase_summary(matches)
    _render_phase_table(phase_summary)


if __name__ == "__main__":
    main()
