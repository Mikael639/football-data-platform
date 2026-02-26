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
            linear-gradient(135deg, rgba(14,111,255,0.12), rgba(22,179,154,0.10)),
            #ffffff;
          border: 1px solid var(--fdp-border);
          border-radius: 18px;
          padding: 16px 18px;
          box-shadow: 0 8px 24px rgba(19,34,56,0.06);
          margin: 4px 0 12px 0;
        }

        .fdp-hero-title {
          font-size: 1.1rem;
          font-weight: 700;
          color: var(--fdp-ink);
          margin-bottom: 4px;
        }

        .fdp-hero-sub {
          color: #405a78;
          font-size: 0.92rem;
          line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
