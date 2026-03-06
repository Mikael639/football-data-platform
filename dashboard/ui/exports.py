from __future__ import annotations

import pandas as pd
import streamlit as st


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def render_csv_download(
    *,
    df: pd.DataFrame,
    label: str,
    filename: str,
    key: str,
) -> None:
    target_name = filename if filename.lower().endswith(".csv") else f"{filename}.csv"
    disabled = df is None or df.empty
    st.download_button(
        label=label,
        data=dataframe_to_csv_bytes(df),
        file_name=target_name,
        mime="text/csv",
        disabled=disabled,
        key=key,
    )
