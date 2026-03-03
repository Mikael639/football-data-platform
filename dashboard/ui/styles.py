import streamlit as st


VISUAL_COLORS = {
    "points": "#0E6FFF",
    "attack": "#1FA774",
    "defense": "#D64B4B",
    "neutral": "#C58B1A",
    "teal": "#1697A6",
    "violet": "#5B6CF0",
}


def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --fdp-bg-soft: #f3f6fb;
          --fdp-ink: #132238;
          --fdp-accent: #0e6fff;
          --fdp-accent-2: #16b39a;
          --fdp-border: #d9e2ef;
          --fdp-card: #ffffff;
        }

        .stApp {
          background:
            radial-gradient(circle at 0% 0%, rgba(14,111,255,0.08), transparent 38%),
            radial-gradient(circle at 100% 20%, rgba(22,179,154,0.08), transparent 40%),
            linear-gradient(180deg, #f7f9fd 0%, #eef3fa 100%);
        }

        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #f6f8fc 0%, #eef3fa 100%);
          border-right: 1px solid #d9e2ef;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a p,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button p {
          color: #2a3f5b !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
          color: #314b6a !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stDateInput input,
        section[data-testid="stSidebar"] .stTextInput input {
          background: #172843 !important;
          color: #ffffff !important;
          border-color: #22395b !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
        section[data-testid="stSidebar"] .stDateInput svg {
          fill: #dbe8f6 !important;
        }

        section[data-testid="stSidebar"] .stDateInput label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stTextInput label {
          color: #4d6684 !important;
          font-weight: 700 !important;
        }

        div[data-testid="stMetric"] {
          background: var(--fdp-card);
          border: 1px solid var(--fdp-border);
          border-radius: 14px;
          padding: 10px 12px;
          box-shadow: 0 3px 12px rgba(19,34,56,0.05);
        }

        div[data-testid="stTabs"] button[role="tab"] {
          border-radius: 12px;
          padding: 8px 14px;
          border: 1px solid transparent;
          color: #29415f;
          font-weight: 600;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
          background: rgba(14,111,255,0.10);
          border-color: rgba(14,111,255,0.25);
          color: #0b4fb4;
        }

        div.stButton > button {
          border-radius: 12px;
          border: 1px solid #c9d7ea;
          background: #ffffff;
          color: #163252;
          font-weight: 600;
        }

        div.stButton > button:hover {
          border-color: #0e6fff;
          color: #0e6fff;
        }

        .fdp-hero {
          background:
            radial-gradient(circle at top right, rgba(77, 211, 191, 0.22), transparent 34%),
            linear-gradient(135deg, rgba(8, 39, 79, 0.96), rgba(11, 93, 217, 0.88));
          border: 1px solid rgba(38, 86, 148, 0.55);
          border-radius: 26px;
          padding: 26px 28px;
          box-shadow: 0 24px 50px rgba(19,34,56,0.16);
          margin: 4px 0 16px 0;
        }

        .fdp-page-banner {
          background:
            radial-gradient(circle at top right, rgba(77, 211, 191, 0.18), transparent 34%),
            linear-gradient(135deg, rgba(8, 39, 79, 0.94), rgba(11, 93, 217, 0.84));
          border: 1px solid rgba(38, 86, 148, 0.5);
          border-radius: 24px;
          padding: 20px 22px;
          box-shadow: 0 18px 36px rgba(19,34,56,0.12);
          margin: 2px 0 10px 0;
        }

        .fdp-page-banner-kicker {
          display: inline-flex;
          align-items: center;
          padding: 5px 9px;
          margin-bottom: 10px;
          border-radius: 999px;
          background: rgba(255,255,255,0.12);
          color: #dbeafe;
          font-size: 0.68rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          border: 1px solid rgba(255,255,255,0.16);
        }

        .fdp-page-banner-title {
          color: #ffffff;
          font-size: 1.38rem;
          font-weight: 800;
          letter-spacing: -0.02em;
          margin-bottom: 6px;
        }

        .fdp-page-banner-sub {
          color: rgba(255,255,255,0.82);
          font-size: 0.96rem;
          line-height: 1.5;
          max-width: 720px;
        }

        .fdp-hero-title {
          font-size: 1.5rem;
          font-weight: 800;
          color: #ffffff;
          margin-bottom: 6px;
          letter-spacing: -0.02em;
        }

        .fdp-hero-sub {
          color: rgba(255,255,255,0.82);
          font-size: 0.98rem;
          line-height: 1.5;
          max-width: 720px;
        }

        .fdp-home-ribbon {
          display: inline-flex;
          align-items: center;
          padding: 6px 10px;
          margin-bottom: 12px;
          border-radius: 999px;
          background: rgba(255,255,255,0.12);
          color: #dbeafe;
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          border: 1px solid rgba(255,255,255,0.18);
        }

        .fdp-section-title {
          font-size: 0.82rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #56708f;
          margin: 8px 0 10px 0;
        }

        .fdp-section-sub {
          margin: -2px 0 12px 0;
          color: #4a637e;
          font-size: 0.93rem;
          line-height: 1.45;
        }

        .fdp-note-card {
          background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(243,247,252,0.95));
          border: 1px solid var(--fdp-border);
          border-radius: 16px;
          padding: 12px 14px;
          color: #425d78;
          font-size: 0.92rem;
          line-height: 1.5;
          box-shadow: 0 8px 20px rgba(19,34,56,0.04);
          margin: 8px 0 14px 0;
        }

        .fdp-page-card {
          background: rgba(255,255,255,0.92);
          border: 1px solid var(--fdp-border);
          border-radius: 22px;
          padding: 12px 12px 16px 12px;
          min-height: 320px;
          box-shadow: 0 16px 30px rgba(19,34,56,0.06);
          overflow: hidden;
          backdrop-filter: blur(6px);
        }

        .fdp-page-card-compact {
          min-height: auto;
          padding: 12px 14px 14px 14px;
          box-shadow: 0 12px 24px rgba(19,34,56,0.05);
          background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(247,250,254,0.95));
        }

        .fdp-page-card-media {
          width: 100%;
          height: 166px;
          border-radius: 16px;
          overflow: hidden;
          background: linear-gradient(135deg, rgba(14,111,255,0.14), rgba(22,179,154,0.12));
          margin-bottom: 14px;
          border: 1px solid rgba(217,226,239,0.8);
          position: relative;
        }

        .fdp-page-card-media img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
          transform: scale(1.02);
        }

        .fdp-page-card-overlay {
          position: absolute;
          inset: auto 0 0 0;
          padding: 18px 16px 14px 16px;
          background: linear-gradient(180deg, rgba(10, 23, 40, 0.02), rgba(10, 23, 40, 0.78));
          display: flex;
          flex-direction: column;
          gap: 3px;
        }

        .fdp-page-card-kicker {
          color: rgba(255,255,255,0.78);
          font-size: 0.68rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .fdp-page-card-title {
          color: #ffffff;
          font-size: 1rem;
          font-weight: 800;
          letter-spacing: -0.01em;
        }

        .fdp-page-card-media-fallback {
          background:
            radial-gradient(circle at 18% 22%, rgba(14,111,255,0.2), transparent 32%),
            radial-gradient(circle at 82% 36%, rgba(22,179,154,0.18), transparent 30%),
            linear-gradient(135deg, rgba(14,111,255,0.08), rgba(22,179,154,0.08));
        }

        .fdp-page-card h3 {
          margin: 0 0 8px 0;
          color: var(--fdp-ink);
          font-size: 1.04rem;
          letter-spacing: -0.01em;
        }

        .fdp-page-card p {
          margin: 0 0 10px 0;
          color: #46627f;
          font-size: 0.92rem;
          line-height: 1.5;
        }

        .fdp-page-tile {
          display: flex;
          flex-direction: column;
          gap: 10px;
          height: 100%;
        }

        .fdp-page-tile-title {
          margin: 0 0 8px 0;
          color: var(--fdp-ink);
          font-size: 1.04rem;
          font-weight: 800;
          letter-spacing: -0.01em;
        }

        .fdp-page-tile-text {
          margin: 0 0 10px 0;
          color: #46627f;
          font-size: 0.92rem;
          line-height: 1.55;
        }

        .fdp-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }

        .fdp-chip {
          display: inline-block;
          border-radius: 999px;
          padding: 5px 10px;
          font-size: 0.74rem;
          font-weight: 600;
          background: #eef4fb;
          color: #29415f;
          border: 1px solid #d8e2ef;
        }

        .fdp-home-summary-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px;
          margin: 6px 0 18px 0;
        }

        .fdp-home-summary-card {
          background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(243,247,252,0.96));
          border: 1px solid var(--fdp-border);
          border-radius: 18px;
          padding: 14px 16px;
          box-shadow: 0 10px 24px rgba(19,34,56,0.05);
        }

        .fdp-home-summary-kicker {
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #65819f;
          margin-bottom: 8px;
        }

        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div[data-testid="stImage"] {
          display: flex;
          justify-content: center;
          margin-bottom: 4px;
        }

        div[data-testid="stPageLink"] a {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 9px 14px;
          margin-top: 8px;
          border-radius: 999px;
          background: #0e6fff;
          color: #ffffff !important;
          font-weight: 700;
          text-decoration: none;
          border: 1px solid rgba(14,111,255,0.28);
          box-shadow: 0 8px 18px rgba(14,111,255,0.18);
        }

        div[data-testid="stPageLink"] a:hover {
          background: #0b5fd9;
          border-color: rgba(11,95,217,0.35);
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
          align-self: stretch;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a p,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button p {
          text-transform: uppercase;
          letter-spacing: 0.04em;
          font-weight: 700;
        }

        .fdp-signal-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
          gap: 12px;
          margin: 6px 0 14px 0;
        }

        .fdp-signal-card {
          background: rgba(255,255,255,0.94);
          border: 1px solid var(--fdp-border);
          border-radius: 18px;
          padding: 14px 16px;
          box-shadow: 0 8px 24px rgba(19,34,56,0.05);
        }

        .fdp-signal-label {
          font-size: 0.76rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #6a829c;
          margin-bottom: 8px;
        }

        .fdp-signal-value {
          font-size: 1.15rem;
          font-weight: 800;
          color: var(--fdp-ink);
          line-height: 1.1;
        }

        .fdp-signal-sub {
          margin-top: 6px;
          color: #4d6783;
          font-size: 0.87rem;
          line-height: 1.35;
        }

        .fdp-run-list {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px;
          margin: 8px 0 14px 0;
        }

        .fdp-run-card {
          background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245,248,253,0.98));
          border: 1px solid var(--fdp-border);
          border-radius: 18px;
          padding: 14px 16px;
          box-shadow: 0 8px 24px rgba(19,34,56,0.04);
        }

        .fdp-run-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 10px;
        }

        .fdp-run-id {
          font-size: 0.95rem;
          font-weight: 800;
          color: var(--fdp-ink);
        }

        .fdp-run-meta {
          color: #4d6783;
          font-size: 0.86rem;
          line-height: 1.45;
        }

        .fdp-standings-wrap {
          margin-top: 10px;
          border: 1px solid var(--fdp-border);
          border-radius: 18px;
          overflow: hidden;
          background: rgba(255,255,255,0.96);
          box-shadow: 0 8px 24px rgba(19,34,56,0.04);
        }

        .fdp-standings-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.92rem;
        }

        .fdp-standings-table thead th {
          text-align: left;
          padding: 12px 14px;
          background: #f5f8fc;
          color: #5b738e;
          font-size: 0.76rem;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          border-bottom: 1px solid #e1e8f1;
        }

        .fdp-standings-table tbody td {
          padding: 12px 14px;
          border-bottom: 1px solid #eef3f8;
          color: #21364f;
          vertical-align: middle;
        }

        .fdp-team-cell {
          display: flex;
          align-items: center;
          gap: 10px;
          min-width: 220px;
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
          background: #edf2f8;
          color: #29415f;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 0.72rem;
          font-weight: 800;
          flex: 0 0 auto;
        }

        .fdp-form-strip {
          display: flex;
          gap: 6px;
          align-items: center;
          flex-wrap: nowrap;
        }

        .fdp-form-pill {
          width: 22px;
          height: 22px;
          border-radius: 999px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 0.72rem;
          font-weight: 800;
          color: #fff;
        }

        .fdp-form-win {
          background: #16a34a;
        }

        .fdp-form-draw {
          background: #d4a017;
        }

        .fdp-form-loss {
          background: #d64b4b;
        }

        .fdp-form-empty {
          background: #cbd5e1;
        }

        .fdp-standings-table tbody tr:last-child td {
          border-bottom: none;
        }

        .fdp-zone-ucl {
          background: rgba(14,111,255,0.05);
          box-shadow: inset 4px 0 0 #0e6fff;
        }

        .fdp-zone-uel {
          background: rgba(22,179,154,0.05);
          box-shadow: inset 4px 0 0 #16b39a;
        }

        .fdp-zone-danger {
          background: rgba(245,158,11,0.08);
          box-shadow: inset 4px 0 0 #d97706;
        }

        .fdp-zone-relegation {
          background: rgba(214,75,75,0.07);
          box-shadow: inset 4px 0 0 #d64b4b;
        }

        .fdp-pos-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 32px;
          height: 32px;
          border-radius: 999px;
          background: #edf2f8;
          color: #21364f;
          font-weight: 800;
          font-size: 0.88rem;
        }

        .fdp-pos-ucl {
          background: #e7f0ff;
          color: #0b5bd3;
        }

        .fdp-pos-uel {
          background: #eafaf5;
          color: #0f8b69;
        }

        .fdp-pos-danger {
          background: #fff4df;
          color: #a16207;
        }

        .fdp-pos-relegation {
          background: #fdecec;
          color: #b42318;
        }

        .fdp-stat-for {
          color: #168a5b;
          font-weight: 700;
        }

        .fdp-stat-against {
          color: #cf3f4f;
          font-weight: 700;
        }

        .fdp-zone-label {
          display: inline-block;
          border-radius: 999px;
          padding: 4px 10px;
          font-size: 0.76rem;
          font-weight: 700;
          border: 1px solid transparent;
        }

        .fdp-zone-label-ucl {
          background: #e7f0ff;
          color: #0b5bd3;
          border-color: #cfe0ff;
        }

        .fdp-zone-label-uel {
          background: #eafaf5;
          color: #0f8b69;
          border-color: #cbeee3;
        }

        .fdp-zone-label-danger {
          background: #fff4df;
          color: #a16207;
          border-color: #f3d7a1;
        }

        .fdp-zone-label-relegation {
          background: #fdecec;
          color: #b42318;
          border-color: #f2c7c7;
        }

        @media (prefers-color-scheme: dark) {
          :root {
            --fdp-bg-soft: #08111f;
            --fdp-ink: #edf4ff;
            --fdp-accent: #54a6ff;
            --fdp-accent-2: #2ad1b2;
            --fdp-border: rgba(155, 184, 217, 0.18);
            --fdp-card: rgba(11, 20, 35, 0.88);
          }

          .stApp {
            background:
              radial-gradient(circle at 0% 0%, rgba(84,166,255,0.12), transparent 38%),
              radial-gradient(circle at 100% 20%, rgba(42,209,178,0.10), transparent 40%),
              linear-gradient(180deg, #06101d 0%, #0c1728 100%);
          }

          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1627 0%, #101b2f 100%);
            border-right: 1px solid rgba(155, 184, 217, 0.16);
          }

          div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(19, 31, 51, 0.96), rgba(15, 25, 42, 0.94));
            border-color: rgba(155, 184, 217, 0.28);
            box-shadow: 0 10px 24px rgba(0,0,0,0.26);
          }

          div[data-testid="stMetric"] label,
          div[data-testid="stMetric"] [data-testid="stMetricLabel"],
          div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
            color: #d5e4f5 !important;
          }

          div[data-testid="stMetric"] [data-testid="stMetricValue"],
          div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
            color: #ffffff !important;
          }

          div[data-testid="stMetric"] [data-testid="stMetricDelta"],
          div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
            color: #e4effc !important;
          }

          div[data-testid="stTabs"] button[role="tab"] {
            color: #cfe0f5;
            background: rgba(255,255,255,0.03);
          }

          div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: rgba(84,166,255,0.18);
            border-color: rgba(84,166,255,0.28);
            color: #ffffff;
          }

          div.stButton > button {
            background: rgba(12, 21, 37, 0.92);
            border-color: rgba(155, 184, 217, 0.18);
            color: #edf4ff;
          }

          div.stButton > button:hover {
            border-color: #54a6ff;
            color: #ffffff;
          }

          label,
          .stSelectbox label,
          .stDateInput label,
          .stMultiSelect label,
          .stSlider label,
          .stRadio label {
            color: #e3eefb !important;
            font-weight: 600 !important;
          }

          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          div[data-baseweb="popover"] input,
          .stDateInput input,
          .stTextInput input {
            background: rgba(22, 35, 57, 0.98) !important;
            color: #f8fbff !important;
            border-color: rgba(155, 184, 217, 0.36) !important;
          }

          div[data-baseweb="select"] > div:hover,
          div[data-baseweb="input"] > div:hover,
          .stDateInput input:hover,
          .stTextInput input:hover {
            border-color: rgba(84,166,255,0.55) !important;
          }

          div[data-baseweb="select"] svg,
          .stDateInput svg {
            fill: #cfe0f5 !important;
          }

          div[data-baseweb="tag"] {
            background: rgba(84,166,255,0.16) !important;
            color: #eef5ff !important;
            border-color: rgba(84,166,255,0.24) !important;
          }

          section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
          section[data-testid="stSidebar"] .stDateInput input {
            background: rgba(26, 39, 61, 0.98) !important;
          }

          section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a p,
          section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span,
          section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button p,
          section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button span {
            color: #e7f0fb !important;
            opacity: 1 !important;
            font-weight: 700 !important;
          }

          section[data-testid="stSidebar"] [aria-selected="true"] p,
          section[data-testid="stSidebar"] [aria-selected="true"] span {
            color: #ffffff !important;
          }

          section[data-testid="stSidebar"] [data-testid="stMarkdown"],
          section[data-testid="stSidebar"] p,
          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] h1,
          section[data-testid="stSidebar"] h2,
          section[data-testid="stSidebar"] h3 {
            color: #e7f0fb !important;
            opacity: 1 !important;
          }

          section[data-testid="stSidebar"] .stDateInput label,
          section[data-testid="stSidebar"] .stSelectbox label,
          section[data-testid="stSidebar"] .stMultiSelect label,
          section[data-testid="stSidebar"] .stTextInput label,
          section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            color: #cfe0f5 !important;
            font-weight: 700 !important;
            opacity: 1 !important;
          }

          .fdp-note-card,
          .fdp-home-summary-card,
          .fdp-page-card,
          .fdp-page-card-compact,
          .fdp-signal-card,
          .fdp-run-card,
          .fdp-standings-wrap {
            background: linear-gradient(180deg, rgba(18,29,48,0.96), rgba(14,24,40,0.94));
            border-color: rgba(155, 184, 217, 0.24);
            box-shadow: 0 12px 28px rgba(0,0,0,0.22);
          }

          .fdp-home-summary-kicker,
          .fdp-section-title,
          .fdp-signal-label,
          .fdp-run-meta,
          .fdp-section-sub,
          .fdp-note-card,
          .fdp-page-card p,
          .fdp-page-tile-text,
          .fdp-signal-sub {
            color: #bfd0e6;
          }

          .fdp-signal-value,
          .fdp-page-card h3,
          .fdp-page-tile-title,
          .fdp-run-id {
            color: #f5f9ff;
          }

          .fdp-chip {
            background: rgba(84,166,255,0.10);
            color: #dceaff;
            border-color: rgba(84,166,255,0.18);
          }

          .fdp-page-card-media {
            background: linear-gradient(135deg, rgba(84,166,255,0.10), rgba(42,209,178,0.08));
            border-color: rgba(155, 184, 217, 0.18);
          }

          .fdp-standings-table thead th {
            background: #0f1b2f;
            color: #a9c0da;
            border-bottom-color: rgba(155, 184, 217, 0.14);
          }

          .fdp-standings-table tbody td {
            color: #ecf4ff;
            border-bottom-color: rgba(155, 184, 217, 0.10);
          }

          .fdp-team-fallback,
          .fdp-pos-badge {
            background: rgba(255,255,255,0.08);
            color: #eef5ff;
          }

          .fdp-zone-ucl {
            background: rgba(14,111,255,0.10);
          }

          .fdp-zone-uel {
            background: rgba(22,179,154,0.10);
          }

          .fdp-zone-danger {
            background: rgba(245,158,11,0.12);
          }

          .fdp-zone-relegation {
            background: rgba(214,75,75,0.12);
          }

          div[data-testid="stPageLink"] a {
            background: #267cff;
            border-color: rgba(84,166,255,0.22);
            box-shadow: 0 10px 22px rgba(0,0,0,0.24);
          }

          div[data-testid="stPageLink"] a:hover {
            background: #3b89ff;
          }
        }

        @media (max-width: 900px) {
          .fdp-hero,
          .fdp-page-banner {
            padding: 16px 18px;
            border-radius: 18px;
          }

          .fdp-hero-title,
          .fdp-page-banner-title {
            font-size: 1.18rem;
          }

          .fdp-hero-sub,
          .fdp-page-banner-sub,
          .fdp-note-card,
          .fdp-section-sub {
            font-size: 0.88rem;
          }

          .fdp-chip-row {
            gap: 6px;
          }

          .fdp-chip,
          .fdp-zone-label {
            font-size: 0.68rem;
            padding: 4px 8px;
          }

          .fdp-signal-grid,
          .fdp-home-summary-grid,
          .fdp-run-list {
            grid-template-columns: 1fr;
          }

          .fdp-page-card h3 {
            font-size: 0.96rem;
          }

          .fdp-page-card p,
          .fdp-page-tile-text,
          .fdp-signal-sub,
          .fdp-run-meta {
            font-size: 0.82rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
