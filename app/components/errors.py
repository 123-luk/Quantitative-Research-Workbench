"""Secret-safe, user-facing exception presentation."""

from __future__ import annotations

from dataclasses import asdict

from app.services.run_service import SafeRunError


class ErrorPresenter:
    @staticmethod
    def payload(error: SafeRunError) -> dict[str, object]:
        if not isinstance(error, SafeRunError):
            raise TypeError("error must be SafeRunError.")
        return {
            "title": "Research Run Failed",
            "stage": error.stage,
            "reason": error.message,
            "technical_details": {
                key: value
                for key, value in asdict(error).items()
                if key in {"exception_class", "message", "stage", "run_id"}
            },
        }

    @staticmethod
    def render(st: object, error: SafeRunError) -> None:
        payload = ErrorPresenter.payload(error)
        st.error(payload["title"])
        if payload["stage"]:
            st.markdown(f"**Stage:** {payload['stage']}")
        st.markdown(f"**Reason:** {payload['reason']}")
        with st.expander("Technical Details"):
            st.json(payload["technical_details"])

