from __future__ import annotations

import hmac
import os

import streamlit as st

ADMIN_AUTH_KEY = "fdp_admin_authenticated"
ADMIN_USER_KEY = "fdp_admin_user"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123!"


def _admin_username() -> str:
    return str(os.getenv("DASHBOARD_ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME)).strip() or DEFAULT_ADMIN_USERNAME


def _admin_password() -> str:
    return str(os.getenv("DASHBOARD_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)).strip() or DEFAULT_ADMIN_PASSWORD


def is_admin_authenticated() -> bool:
    return bool(st.session_state.get(ADMIN_AUTH_KEY, False))


def render_admin_access_sidebar(key_prefix: str = "admin_access") -> None:
    with st.sidebar.expander("Admin access", expanded=False):
        if is_admin_authenticated():
            current_user = str(st.session_state.get(ADMIN_USER_KEY, _admin_username()))
            st.success(f"Connecte en admin ({current_user})")
            if st.button("Se deconnecter", key=f"{key_prefix}_logout", use_container_width=True):
                st.session_state[ADMIN_AUTH_KEY] = False
                st.session_state[ADMIN_USER_KEY] = ""
                st.rerun()
            return

        user_key = f"{key_prefix}_user"
        pwd_key = f"{key_prefix}_pwd"
        st.text_input("Nom admin", key=user_key, value=_admin_username())
        st.text_input("Mot de passe admin", type="password", key=pwd_key)
        if st.button("Connexion admin", key=f"{key_prefix}_login", use_container_width=True):
            user_candidate = str(st.session_state.get(user_key, "")).strip()
            candidate = str(st.session_state.get(pwd_key, ""))
            username_ok = hmac.compare_digest(user_candidate, _admin_username())
            password_ok = hmac.compare_digest(candidate, _admin_password())
            if username_ok and password_ok:
                st.session_state[ADMIN_AUTH_KEY] = True
                st.session_state[ADMIN_USER_KEY] = user_candidate
                st.success("Connexion admin reussie.")
                st.rerun()
            else:
                st.error("Identifiants admin incorrects.")


def require_admin_access(page_name: str = "Cette page") -> bool:
    if is_admin_authenticated():
        return True
    st.warning(f"Acces admin requis pour {page_name}.")
    st.info("Utilise le panneau `Admin access` dans la sidebar pour te connecter.")
    return False
