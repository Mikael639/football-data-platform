from __future__ import annotations

import html
from typing import Callable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


BadgeRenderer = Callable[[object], tuple[str, str]]
RowClassRenderer = Callable[[pd.Series], str]


def _table_css() -> str:
    return """
    <style>
      :root {
        --fdp-bg: #ffffff;
        --fdp-head: #f7faff;
        --fdp-line: #e5ebf3;
        --fdp-ink: #16283d;
        --fdp-muted: #637a94;
        --fdp-row-alt: #f9fcff;
        --fdp-pill-neutral-bg: rgba(98, 122, 148, 0.16);
        --fdp-pill-neutral-fg: #3f5874;
        --fdp-pill-good-bg: rgba(16, 155, 116, 0.16);
        --fdp-pill-good-fg: #0e785a;
        --fdp-pill-warn-bg: rgba(210, 146, 24, 0.16);
        --fdp-pill-warn-fg: #9a6b00;
        --fdp-pill-bad-bg: rgba(203, 58, 58, 0.14);
        --fdp-pill-bad-fg: #b32525;
        --fdp-pill-info-bg: rgba(13, 99, 221, 0.14);
        --fdp-pill-info-fg: #0d63dd;
        --fdp-row-top: rgba(13, 99, 221, 0.08);
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --fdp-bg: #101a2d;
          --fdp-head: #0f1f36;
          --fdp-line: rgba(155, 184, 217, 0.14);
          --fdp-ink: #edf4ff;
          --fdp-muted: #9db5d0;
          --fdp-row-alt: rgba(255, 255, 255, 0.02);
          --fdp-pill-neutral-bg: rgba(157, 181, 208, 0.2);
          --fdp-pill-neutral-fg: #cbdcf0;
          --fdp-pill-good-bg: rgba(89, 214, 176, 0.18);
          --fdp-pill-good-fg: #89e9cc;
          --fdp-pill-warn-bg: rgba(255, 200, 94, 0.18);
          --fdp-pill-warn-fg: #ffd99a;
          --fdp-pill-bad-bg: rgba(255, 109, 109, 0.22);
          --fdp-pill-bad-fg: #ffaeae;
          --fdp-pill-info-bg: rgba(111, 173, 255, 0.2);
          --fdp-pill-info-fg: #b7d7ff;
          --fdp-row-top: rgba(111, 173, 255, 0.11);
        }
      }

      body {
        margin: 0;
        background: transparent;
        color: var(--fdp-ink);
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      }

      .fdp-adapt-wrap {
        border: 1px solid var(--fdp-line);
        border-radius: 16px;
        overflow: hidden;
        background: var(--fdp-bg);
      }

      .fdp-adapt-head {
        padding: 10px 12px;
        border-bottom: 1px solid var(--fdp-line);
        background: var(--fdp-head);
        color: var(--fdp-muted);
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
      }

      .fdp-adapt-scroll {
        overflow-x: auto;
      }

      .fdp-adapt-table {
        width: 100%;
        min-width: 720px;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 13px;
      }

      .fdp-adapt-table thead th {
        text-align: left;
        padding: 10px 9px;
        border-bottom: 1px solid var(--fdp-line);
        color: var(--fdp-muted);
        font-size: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .fdp-adapt-table tbody td {
        padding: 10px 9px;
        border-bottom: 1px solid var(--fdp-line);
        vertical-align: middle;
        word-break: break-word;
      }

      .fdp-adapt-table tbody tr:nth-child(even) {
        background: var(--fdp-row-alt);
      }

      .fdp-adapt-table tbody tr:last-child td {
        border-bottom: none;
      }

      .fdp-row-top {
        background: linear-gradient(90deg, var(--fdp-row-top), transparent 36%);
        box-shadow: inset 3px 0 0 #0d63dd;
      }

      .fdp-right {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }

      .fdp-cell-strong {
        font-weight: 700;
      }

      .fdp-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
      }

      .fdp-pill-neutral { background: var(--fdp-pill-neutral-bg); color: var(--fdp-pill-neutral-fg); }
      .fdp-pill-good { background: var(--fdp-pill-good-bg); color: var(--fdp-pill-good-fg); }
      .fdp-pill-warn { background: var(--fdp-pill-warn-bg); color: var(--fdp-pill-warn-fg); }
      .fdp-pill-bad { background: var(--fdp-pill-bad-bg); color: var(--fdp-pill-bad-fg); }
      .fdp-pill-info { background: var(--fdp-pill-info-bg); color: var(--fdp-pill-info-fg); }

      @media (max-width: 760px) {
        .fdp-adapt-table {
          min-width: 620px;
          font-size: 12px;
        }

        .fdp-adapt-table thead th,
        .fdp-adapt-table tbody td {
          padding: 8px 7px;
        }
      }
    </style>
    """


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    if isinstance(value, (int, bool)):
        return str(int(value)) if isinstance(value, bool) else str(value)
    text = str(value).strip()
    return text if text else "-"


def _status_badge(value: object) -> tuple[str, str]:
    raw = str(value or "").strip().upper()
    if raw in {"IN_PLAY", "PAUSED", "LIVE"}:
        return ("En direct", "fdp-pill-bad")
    if raw in {"FINISHED", "AFTER_EXTRA_TIME", "PENALTY_SHOOTOUT", "AWARDED"}:
        return ("Termine", "fdp-pill-good")
    if raw in {"SCHEDULED", "TIMED"}:
        return ("A venir", "fdp-pill-info")
    if raw == "POSTPONED":
        return ("Reporte", "fdp-pill-warn")
    if raw == "CANCELLED":
        return ("Annule", "fdp-pill-neutral")
    if raw == "SUSPENDED":
        return ("Suspendu", "fdp-pill-warn")
    if raw:
        return (raw.replace("_", " "), "fdp-pill-neutral")
    return ("Inconnu", "fdp-pill-neutral")


def _result_badge(value: object) -> tuple[str, str]:
    raw = str(value or "").strip().upper()
    if raw == "W":
        return ("Win", "fdp-pill-good")
    if raw == "D":
        return ("Draw", "fdp-pill-warn")
    if raw == "L":
        return ("Loss", "fdp-pill-bad")
    if raw == "-":
        return ("-", "fdp-pill-neutral")
    return (raw or "-", "fdp-pill-neutral")


def _trend_badge(value: object) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw or raw == "=":
        return ("=", "fdp-pill-neutral")
    if any(token in raw for token in ("↑", "â†‘", "+")):
        return (raw.replace("â†‘", "↑"), "fdp-pill-good")
    if any(token in raw for token in ("↓", "â†“", "-")):
        return (raw.replace("â†“", "↓"), "fdp-pill-bad")
    return (raw, "fdp-pill-neutral")


def _venue_badge(value: object) -> tuple[str, str]:
    raw = str(value or "").strip().upper()
    if raw in {"HOME", "H", "DOMICILE"}:
        return ("Domicile", "fdp-pill-info")
    if raw in {"AWAY", "A", "EXTERIEUR", "EXT"}:
        return ("Exterieur", "fdp-pill-neutral")
    return (raw or "-", "fdp-pill-neutral")


def _delta_badge(value: object) -> tuple[str, str]:
    if pd.isna(value):
        return ("-", "fdp-pill-neutral")
    try:
        delta = float(value)
    except (TypeError, ValueError):
        return (str(value).strip() or "-", "fdp-pill-neutral")

    if delta > 0:
        label = f"+{int(delta)}" if delta.is_integer() else f"+{delta:.2f}"
        return (label, "fdp-pill-good")
    if delta < 0:
        label = f"{int(delta)}" if delta.is_integer() else f"{delta:.2f}"
        return (label, "fdp-pill-bad")
    return ("0", "fdp-pill-neutral")


def _delta_position_badge(value: object) -> tuple[str, str]:
    if pd.isna(value):
        return ("-", "fdp-pill-neutral")
    try:
        delta = float(value)
    except (TypeError, ValueError):
        return (str(value).strip() or "-", "fdp-pill-neutral")

    if delta < 0:
        steps = abs(int(delta)) if float(delta).is_integer() else abs(delta)
        label = f"up {steps}" if isinstance(steps, int) else f"up {steps:.2f}"
        return (label, "fdp-pill-good")
    if delta > 0:
        steps = int(delta) if float(delta).is_integer() else delta
        label = f"down {steps}" if isinstance(steps, int) else f"down {steps:.2f}"
        return (label, "fdp-pill-bad")
    return ("=", "fdp-pill-neutral")


BADGE_RENDERERS: dict[str, BadgeRenderer] = {
    "status": _status_badge,
    "result": _result_badge,
    "trend": _trend_badge,
    "venue": _venue_badge,
    "delta": _delta_badge,
    "delta_position": _delta_position_badge,
}


def render_adaptive_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    empty_message: str = "Aucune donnee disponible.",
    badge_columns: dict[str, str] | None = None,
    right_align_columns: set[str] | None = None,
    strong_columns: set[str] | None = None,
    row_class_renderer: RowClassRenderer | None = None,
    row_height: int = 42,
    min_height: int = 220,
    max_height: int = 860,
) -> None:
    if df.empty:
        st.info(empty_message)
        return

    table = df.copy().reset_index(drop=True)
    badge_columns = badge_columns or {}
    right_align_columns = right_align_columns or set()
    strong_columns = strong_columns or set()

    numeric_columns = {
        column
        for column in table.columns
        if pd.api.types.is_numeric_dtype(table[column])
    }
    right_columns = right_align_columns.union(numeric_columns)

    rows_html: list[str] = []
    for _, row in table.iterrows():
        row_class = row_class_renderer(row) if row_class_renderer else ""
        cell_html: list[str] = []
        for column in table.columns:
            raw_value = row.get(column)
            text_value = _format_cell(raw_value)
            classes: list[str] = []
            if column in right_columns:
                classes.append("fdp-right")
            if column in strong_columns:
                classes.append("fdp-cell-strong")

            badge_kind = badge_columns.get(str(column))
            if badge_kind:
                renderer = BADGE_RENDERERS.get(badge_kind)
                if renderer:
                    badge_label, badge_class = renderer(raw_value)
                    rendered = f'<span class="fdp-pill {badge_class}">{html.escape(badge_label)}</span>'
                else:
                    rendered = html.escape(text_value)
            else:
                rendered = html.escape(text_value)
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            cell_html.append(f"<td{class_attr}>{rendered}</td>")
        row_attr = f' class="{html.escape(row_class)}"' if row_class else ""
        rows_html.append(f"<tr{row_attr}>{''.join(cell_html)}</tr>")

    headers_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in table.columns)
    title_html = f'<div class="fdp-adapt-head">{html.escape(title)}</div>' if title else ""
    table_html = f"""
    {_table_css()}
    <div class="fdp-adapt-wrap">
      {title_html}
      <div class="fdp-adapt-scroll">
        <table class="fdp-adapt-table">
          <thead>
            <tr>{headers_html}</tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </div>
    """
    height = 78 + (len(table.index) * row_height)
    if title:
        height += 36
    components.html(table_html, height=min(max(height, min_height), max_height), scrolling=True)
