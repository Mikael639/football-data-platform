from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_LALIGA_BADGE = Path(__file__).resolve().parents[1] / "assets" / "laliga-badge.svg"


def laliga_logo_source() -> str:
    return str(DEFAULT_LALIGA_BADGE)


def render_team_hero(selected_team_name: str, season_label: str) -> None:
    title = "Vue equipe" if selected_team_name == "Tous les clubs" else f"Vue equipe - {selected_team_name}"
    subtitle = (
        f"LaLiga {season_label} | KPI, forme recente, domicile/exterieur et calendrier. "
        "Selectionne un club pour une analyse detaillee."
    )
    st.markdown(
        f"""
        <div class="fdp-hero">
          <div class="fdp-hero-title">{title}</div>
          <div class="fdp-hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_form_timeline(last_matches: pd.DataFrame) -> None:
    if last_matches.empty:
        st.info("Aucun match disponible pour la timeline de forme.")
        return

    color_map = {"W": "#1f7a1f", "D": "#b58a00", "L": "#a12622"}
    parts = []
    for _, row in last_matches.iterrows():
        result = str(row["Result"])
        color = color_map.get(result, "#666666")
        label = f"{row['date']} {result}"
        parts.append(
            "<span title='{}' style='display:inline-block;margin:2px;padding:6px 10px;"
            "border-radius:8px;background:{};color:white;font-weight:600;font-size:12px;'>"
            "{}</span>".format(label, color, result)
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_quality_badges(summary: dict[str, int]) -> None:
    badges = []
    palette = {
        "OK": ("#1f7a1f", "#ffffff"),
        "INCOMPLET": ("#a12622", "#ffffff"),
        "A_VERIFIER": ("#8a6d1d", "#ffffff"),
        "TOTAL": ("#2b3a55", "#ffffff"),
    }
    labels = {
        "OK": "OK",
        "INCOMPLET": "Incomplet",
        "A_VERIFIER": "A verifier",
        "TOTAL": "Total",
    }
    ordered_keys = ["OK", "INCOMPLET", "A_VERIFIER", "TOTAL"]
    for key in ordered_keys:
        if key not in summary:
            continue
        bg, fg = palette[key]
        badges.append(
            "<span style='display:inline-block;margin:2px 6px 2px 0;padding:6px 10px;"
            f"border-radius:999px;background:{bg};color:{fg};font-weight:600;font-size:12px;'>"
            f"{labels[key]}: {summary[key]}</span>"
        )
    if badges:
        st.markdown("".join(badges), unsafe_allow_html=True)


def style_ligue_table(df: pd.DataFrame):
    styler = df.style

    def color_gd(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v):
                styles.append("")
            elif float(v) < 0:
                styles.append("color:#B42318;font-weight:700;")
            elif float(v) > 0:
                styles.append("color:#067647;font-weight:700;")
            else:
                styles.append("color:#6b7280;")
        return styles

    def highlight_gf(col: pd.Series):
        if col.dropna().empty:
            return ["" for _ in col]
        q75 = col.dropna().quantile(0.75)
        q90 = col.dropna().quantile(0.90)
        out = []
        for v in col:
            if pd.isna(v):
                out.append("")
            elif float(v) >= q90:
                out.append("background-color:#d1fadf;color:#065f46;font-weight:700;")
            elif float(v) >= q75:
                out.append("background-color:#ecfdf3;color:#065f46;")
            else:
                out.append("")
        return out

    def highlight_ga(col: pd.Series):
        if col.dropna().empty:
            return ["" for _ in col]
        q75 = col.dropna().quantile(0.75)
        q90 = col.dropna().quantile(0.90)
        out = []
        for v in col:
            if pd.isna(v):
                out.append("")
            elif float(v) >= q90:
                out.append("background-color:#fee4e2;color:#991b1b;font-weight:700;")
            elif float(v) >= q75:
                out.append("background-color:#fff1f0;color:#991b1b;")
            else:
                out.append("")
        return out

    def highlight_pts(col: pd.Series):
        if col.dropna().empty:
            return ["" for _ in col]
        q75 = col.dropna().quantile(0.75)
        out = []
        for v in col:
            if pd.isna(v):
                out.append("")
            elif float(v) >= q75:
                out.append("background-color:#e9f2ff;color:#0b4fb4;font-weight:700;")
            else:
                out.append("")
        return out

    for c in ["GD", "Diff", "Diff."]:
        if c in df.columns:
            styler = styler.apply(color_gd, subset=[c])
            break
    if "GF" in df.columns:
        styler = styler.apply(highlight_gf, subset=["GF"])
    if "GA" in df.columns:
        styler = styler.apply(highlight_ga, subset=["GA"])
    if "Pts" in df.columns:
        styler = styler.apply(highlight_pts, subset=["Pts"])
    return styler


def add_leader_star(df: pd.DataFrame, team_col: str = "Equipe") -> pd.DataFrame:
    if df is None or df.empty or team_col not in df.columns:
        return df
    out = df.copy()
    out.iloc[0, out.columns.get_loc(team_col)] = f"★ {out.iloc[0][team_col]}"
    return out


def add_podium_icons(df: pd.DataFrame, team_col: str = "Equipe") -> pd.DataFrame:
    if df is None or df.empty or team_col not in df.columns:
        return df
    out = df.copy()
    icons = ["🥇", "🥈", "🥉"]
    for idx, icon in enumerate(icons):
        if idx >= len(out):
            break
        out.iloc[idx, out.columns.get_loc(team_col)] = f"{icon} {out.iloc[idx][team_col]}"
    return out
