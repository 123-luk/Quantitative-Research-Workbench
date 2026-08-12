"""Read-only localized Data Layer 2.0 status page."""

from __future__ import annotations

import pandas as pd

from app.i18n import get_locale, t
from app.services.credential_service import CredentialService
from app.services.data_status_service import DataLayer2StatusService
from app.services.ui_metadata_service import dataset_label, dataset_unit


def render(st: object) -> None:
    locale = get_locale(st.session_state)
    st.title(t("data.title", locale=locale))
    st.caption(t("data.subtitle", locale=locale))
    status = DataLayer2StatusService().get_status()
    credential = CredentialService().resolve(st.session_state.get("tushare_session_token"))
    columns = st.columns(3)
    columns[0].metric(t("data.provider", locale=locale), "TuShare Pro")
    columns[1].metric(t("overview.credential", locale=locale), t("provider.available" if credential.available else "provider.missing", locale=locale))
    columns[2].metric(t("data.ledger", locale=locale), t("readiness.ready" if status.ledger_exists else "readiness.missing_status", locale=locale))
    st.markdown(f"**{t('data.registry', locale=locale)}**")
    st.dataframe(pd.DataFrame([{
        t("readiness.dataset", locale=locale): dataset_label(item.dataset_id, locale),
        t("data.complete_units", locale=locale): f"{item.complete_units} {dataset_unit(item.dataset_id, locale)}",
    } for item in status.datasets]), width="stretch", hide_index=True)
    if not status.ledger_exists:
        st.info(t("data.no_ledger", locale=locale))
    st.caption(t("data.read_only", locale=locale))
    with st.expander(t("task.technical", locale=locale)):
        st.markdown(f"`CURATED: {status.curated_root}`  \n`SQLite: {status.ledger_path}`")


if __name__ == "__main__":
    import streamlit as st
    render(st)
