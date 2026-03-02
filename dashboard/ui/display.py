from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_LALIGA_BADGE_PNG = ASSETS_DIR / "laliga-badge.png"
DEFAULT_LALIGA_BADGE_SVG = ASSETS_DIR / "laliga-badge.svg"


def laliga_logo_source() -> str:
    if DEFAULT_LALIGA_BADGE_PNG.exists():
        return str(DEFAULT_LALIGA_BADGE_PNG)
    return str(DEFAULT_LALIGA_BADGE_SVG)


def render_badge(status: str) -> str:
    palette = {
        "PASS": ("#e8f7ef", "#147a45"),
        "WARN": ("#fff4df", "#9a6700"),
        "FAIL": ("#fde8e8", "#b42318"),
        "SUCCESS": ("#e8f7ef", "#147a45"),
        "FAILED": ("#fde8e8", "#b42318"),
        "STARTED": ("#e8effa", "#1d4ed8"),
    }
    background, foreground = palette.get(status, ("#eef2f7", "#344054"))
    return (
        f"<span style='padding:4px 8px;border-radius:999px;background:{background};"
        f"color:{foreground};font-weight:700;font-size:12px;'>{status}</span>"
    )


def render_status_badge(status: str) -> None:
    st.markdown(render_badge(status), unsafe_allow_html=True)


def render_team_header(team: dict[str, object] | None) -> None:
    if not team:
        st.subheader("Equipe")
        return

    cols = st.columns([1, 5], vertical_alignment="center")
    with cols[0]:
        crest_url = team.get("crest_url")
        if crest_url:
            st.image(str(crest_url), use_column_width=True)
        else:
            st.markdown(
                f"""
                <div style="height:78px;border-radius:18px;background:#eef3fa;
                display:flex;align-items:center;justify-content:center;font-weight:700;color:#29415f;">
                  {str(team.get("short_name") or team.get("team_name") or "?")[:3].upper()}
                </div>
                """,
                unsafe_allow_html=True,
            )
    with cols[1]:
        st.title(str(team.get("team_name") or "Equipe"))
        subtitle_parts = [part for part in [team.get("short_name"), team.get("country")] if part]
        if subtitle_parts:
            st.caption(" | ".join(str(part) for part in subtitle_parts))


def render_result_strip(results: list[str]) -> None:
    if not results:
        st.info("Aucun match joue pour afficher la forme.")
        return

    color_map = {"W": "#1FA774", "D": "#C58B1A", "L": "#D64B4B"}
    parts = []
    for result in results:
        parts.append(
            f"<span style='display:inline-block;margin-right:6px;padding:6px 10px;border-radius:10px;"
            f"background:{color_map.get(result, '#94a3b8')};color:#fff;font-weight:700;'>{result}</span>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def style_monitoring_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    if df.empty:
        return df.style

    def _status_style(series: pd.Series) -> list[str]:
        output = []
        for value in series:
            if value in {"FAIL", "FAILED"}:
                output.append("background-color:#fde8e8;color:#b42318;font-weight:700;")
            elif value == "WARN":
                output.append("background-color:#fff4df;color:#9a6700;font-weight:700;")
            elif value in {"PASS", "SUCCESS"}:
                output.append("background-color:#e8f7ef;color:#147a45;font-weight:700;")
            else:
                output.append("")
        return output

    target_col = "status" if "status" in df.columns else "severity"
    return df.style.apply(_status_style, subset=[target_col])
