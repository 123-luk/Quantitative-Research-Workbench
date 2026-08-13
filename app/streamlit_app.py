"""Single authoritative Streamlit entry for Quant Research Workbench."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.navigation import initialize_session_state, page_path  # noqa: E402
from app.i18n import get_locale, set_locale, t  # noqa: E402
from app.services.credential_service import CredentialService, ProviderErrorKind  # noqa: E402
from app.services.provider_capability_service import ProviderCapabilityService  # noqa: E402
from src.data.provider_registry import ProviderId  # noqa: E402


def _credential_sidebar(locale: str) -> None:
    st.sidebar.divider()
    st.sidebar.subheader(t("provider.section", locale=locale))
    selected = st.sidebar.selectbox(
        t("provider.selector", locale=locale),
        (ProviderId.TUSHARE_OFFICIAL.value, ProviderId.TUSHARE_PROXY.value),
        format_func=lambda value: t(f"provider.{value}", locale=locale),
        key="selected_provider_id",
    )
    token_state = "tushare_official_session_token" if selected == ProviderId.TUSHARE_OFFICIAL.value else "tushare_proxy_session_token"
    widget_key = "tushare_token_widget" if selected == ProviderId.TUSHARE_OFFICIAL.value else "tushare_proxy_token_widget"
    if selected == ProviderId.TUSHARE_PROXY.value:
        st.sidebar.warning(t("provider.proxy_risk", locale=locale))
    token = st.sidebar.text_input(
        t("provider.token" if selected == ProviderId.TUSHARE_OFFICIAL.value else "provider.proxy_token", locale=locale),
        value=str(st.session_state.get(token_state, "")),
        type="password",
        help=t("provider.token_help", locale=locale),
        key=widget_key,
    )
    st.session_state[token_state] = token
    # Keep the legacy official-only key as a non-persistent compatibility view.
    if selected == ProviderId.TUSHARE_OFFICIAL.value:
        st.session_state["tushare_session_token"] = token
    credential = CredentialService().resolve(token, provider_id=selected)
    if credential.source == "environment":
        st.sidebar.success(t("provider.environment", locale=locale))
    elif credential.available:
        st.sidebar.success(t("provider.available", locale=locale))
    else:
        st.sidebar.info(t("provider.missing", locale=locale))
    if st.sidebar.button(t("provider.test", locale=locale), key="test_tushare_connection"):
        result = CredentialService().test_connection(token, provider_id=selected)
        if result.success:
            st.sidebar.success(t("provider.test_ok", locale=locale))
        else:
            key = {
                ProviderErrorKind.CREDENTIAL_MISSING: "provider.missing",
                ProviderErrorKind.AUTHENTICATION_INVALID: "provider.auth",
                ProviderErrorKind.PERMISSION_INSUFFICIENT: "provider.permission",
                ProviderErrorKind.POINTS_INSUFFICIENT: "provider.points",
                ProviderErrorKind.RATE_LIMITED: "provider.rate_limited",
                ProviderErrorKind.NETWORK_ERROR: "provider.network",
                ProviderErrorKind.RESPONSE_INVALID: "provider.response_invalid",
                ProviderErrorKind.PROVIDER_ERROR: "provider.error",
            }[result.error_kind]
            st.sidebar.error(t(key, locale=locale))
    if st.sidebar.button(t("provider.capability_test", locale=locale), key=f"capability_{selected}"):
        if not credential.available:
            st.sidebar.error(t("provider.missing", locale=locale))
        else:
            with st.sidebar.status(t("provider.capability_running", locale=locale), expanded=True):
                report = ProviderCapabilityService().run(
                    selected, credential.reveal_for_provider()  # type: ignore[arg-type]
                )
            st.sidebar.dataframe(
                [{
                    t("readiness.dataset", locale=locale): item.dataset_id,
                    t("readiness.status", locale=locale): item.status,
                    t("readiness.range", locale=locale): ", ".join(item.dates),
                } for item in report.probes],
                hide_index=True,
                width="stretch",
            )


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
        st.Page(page_path("overview"), title=t("nav.overview", locale=locale), icon=":material/home:", url_path="overview", default=True),
        st.Page(page_path("new_run"), title=t("nav.new_run", locale=locale), icon=":material/play_arrow:", url_path="new-run"),
        st.Page(page_path("results"), title=t("nav.results", locale=locale), icon=":material/analytics:", url_path="results"),
        st.Page(page_path("runs"), title=t("nav.runs", locale=locale), icon=":material/history:", url_path="runs"),
        st.Page(page_path("data"), title=t("nav.data", locale=locale), icon=":material/database:", url_path="data"),
    )
    st.navigation(pages, position="sidebar").run()


if __name__ == "__main__":
    main()
