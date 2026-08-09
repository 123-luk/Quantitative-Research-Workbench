"""Streamlit smoke coverage for the V6 Research Backtest control surface."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


STREAMLIT_APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _pipeline_page() -> AppTest:
    app = AppTest.from_file(str(STREAMLIT_APP), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("运行研究流水线")
    app.run()
    assert not app.exception
    return app


def test_research_backtest_disabled_surface_preserves_legacy_controls() -> None:
    app = _pipeline_page()
    source = STREAMLIT_APP.read_text(encoding="utf-8")
    assert "当前能力：V6 Research Backtest Dashboard" in source
    assert "当前版本：V6-A Portfolio Dashboard" not in source
    labels = [widget.label for widget in app.checkbox]
    assert labels == ["Enable Research Backtest", "skip_fetch", "skip_plot"]
    assert not next(
        widget for widget in app.checkbox if widget.label == "Enable Research Backtest"
    ).value
    assert "Transaction cost (bps)" not in [
        widget.label for widget in app.number_input
    ]
    assert "Benchmark code" not in [widget.label for widget in app.text_input]


def test_research_backtest_enabled_controls_build_without_widget_errors() -> None:
    app = _pipeline_page()
    enable = next(
        widget for widget in app.checkbox if widget.label == "Enable Research Backtest"
    )
    enable.set_value(True)
    app.run()
    assert not app.exception
    assert "Transaction cost (bps)" in [
        widget.label for widget in app.number_input
    ]
    assert "Annual risk-free rate" in [
        widget.label for widget in app.number_input
    ]
    assert "Benchmark code" in [widget.label for widget in app.text_input]
    source = STREAMLIT_APP.read_text(encoding="utf-8").split(
        "def render_legacy_pipeline_controls", 1
    )[0]
    assert "Research Backtest End Date" not in source
    assert "Backtest frequency" not in source
