"""Single authoritative Streamlit entry for Quant Research Workbench."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.navigation import initialize_session_state  # noqa: E402
from app.i18n import get_locale, set_locale, t  # noqa: E402
from app.services.credential_service import CredentialService, ProviderErrorKind  # noqa: E402


def _credential_sidebar(locale: str) -> None:
    st.sidebar.divider()
    st.sidebar.subheader(t("provider.section", locale=locale))
    st.sidebar.caption(t("provider.name", locale=locale))
    token = st.sidebar.text_input(
        t("provider.token", locale=locale),
        value=str(st.session_state.get("tushare_session_token", "")),
        type="password",
        help=t("provider.token_help", locale=locale),
        key="tushare_token_widget",
    )
    st.session_state["tushare_session_token"] = token
    credential = CredentialService().resolve(token)
    if credential.source == "environment":
        st.sidebar.success(t("provider.environment", locale=locale))
    elif credential.available:
        st.sidebar.success(t("provider.available", locale=locale))
    else:
        st.sidebar.info(t("provider.missing", locale=locale))
    if st.sidebar.button(t("provider.test", locale=locale), key="test_tushare_connection"):
        result = CredentialService().test_connection(token)
        if result.success:
            st.sidebar.success(t("provider.test_ok", locale=locale))
        else:
            key = {
                ProviderErrorKind.CREDENTIAL_MISSING: "provider.missing",
                ProviderErrorKind.AUTHENTICATION_INVALID: "provider.auth",
                ProviderErrorKind.PERMISSION_INSUFFICIENT: "provider.permission",
                ProviderErrorKind.NETWORK_ERROR: "provider.network",
                ProviderErrorKind.PROVIDER_ERROR: "provider.error",
            }[result.error_kind]
            st.sidebar.error(t(key, locale=locale))


def main() -> None:
    st.set_page_config(page_title="Quant Research Workbench", layout="wide")
    initialize_session_state(st.session_state)
    locale = get_locale(st.session_state)
    language = st.sidebar.selectbox(
        t("language", locale=locale),
        ("zh-CN", "en"),
        index=("zh-CN", "en").index(locale),
        format_func=lambda value: t("language.zh" if value == "zh-CN" else "language.en", locale=locale),
        key="workbench_language_selector",
    )
    if language != locale:
        set_locale(st.session_state, language)
        locale = language
    st.sidebar.title(t("app.title", locale=locale))
    _credential_sidebar(locale)

    pages = (
        st.Page("views/overview.py", title=t("nav.overview", locale=locale), icon=":material/home:", url_path="overview", default=True),
        st.Page("views/new_run.py", title=t("nav.new_run", locale=locale), icon=":material/play_arrow:", url_path="new-run"),
        st.Page("views/results.py", title=t("nav.results", locale=locale), icon=":material/analytics:", url_path="results"),
        st.Page("views/runs.py", title=t("nav.runs", locale=locale), icon=":material/history:", url_path="runs"),
        st.Page("views/data.py", title=t("nav.data", locale=locale), icon=":material/database:", url_path="data"),
    )
    st.navigation(pages, position="sidebar").run()


if __name__ == "__main__":
    main()
