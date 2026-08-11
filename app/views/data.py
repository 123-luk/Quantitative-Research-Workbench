"""Read-only localized Data Layer 2.0 status page."""

from __future__ import annotations

import pandas as pd

from app.i18n import get_locale, t
from app.services.credential_service import CredentialService
from app.services.data_status_service import DataLayer2StatusService


def render(st: object) -> None:
    locale = get_locale(st.session_state)
    st.title(t("data.title", locale=locale))
    st.caption(t("data.subtitle", locale=locale))
    status = DataLayer2StatusService().get_status()
    credential = CredentialService().resolve(st.session_state.get("tushare_session_token"))
    columns = st.columns(3)
    columns[0].metric(t("data.provider", locale=locale), "TuShare Pro")
    columns[1].metric(t("overview.credential", locale=locale), t("provider.available" if credential.available else "provider.missing", locale=locale))
    columns[2].metric(t("data.ledger", locale=locale), "READY" if status.ledger_exists else "N/A")
    st.markdown(f"**{t('data.registry', locale=locale)}**")
    st.dataframe(pd.DataFrame([{"dataset_id": item.dataset_id, "schema_version": item.schema_version, t("data.complete_units", locale=locale): item.complete_units} for item in status.datasets]), width="stretch", hide_index=True)
    st.markdown(f"**CURATED:** `{status.curated_root}`")
    st.markdown(f"**SQLite:** `{status.ledger_path}`")
    if not status.ledger_exists:
        st.info(t("data.no_ledger", locale=locale))
    st.caption(t("data.read_only", locale=locale))


if __name__ == "__main__":
    import streamlit as st
    render(st)
