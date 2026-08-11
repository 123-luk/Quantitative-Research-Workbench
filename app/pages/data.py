"""Read-only local data readiness page."""

from __future__ import annotations

import pandas as pd

from app.services.data_status_service import DataStatusService


def render(st: object) -> None:
    st.title("Data")
    status = DataStatusService().get_status()
    st.subheader("Local Data Readiness")
    columns = st.columns(3)
    columns[0].metric("Cache Status", status.cache_status)
    columns[1].metric("Required Datasets", len(status.required_datasets))
    columns[2].metric("Required Range", f"{status.required_start_date} — {status.required_end_date}")
    st.markdown(f"**Configured data root:** `{status.configured_data_root}`")
    st.markdown(f"**Raw data root:** `{status.raw_data_root}`")
    st.markdown(f"**Cache metadata:** `{status.cache_metadata_path}`")
    st.dataframe(pd.DataFrame([
        {
            "Dataset": item.dataset,
            "Exact Path": item.path,
            "Exact File Exists": item.exists,
            "Cached Start": item.cached_start or "N/A",
            "Cached End": item.cached_end or "N/A",
            "Updated At": item.updated_at or "N/A",
            "Missing Ranges": list(item.missing_ranges),
        }
        for item in status.datasets
    ]), use_container_width=True, hide_index=True)
    st.caption(
        "Read-only status from DataManager, DataCache, and ParquetStore. "
        "No provider call, download, update, repair, or data-lake scan is performed."
    )

