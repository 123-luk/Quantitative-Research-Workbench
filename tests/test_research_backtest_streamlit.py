"""Streamlit smoke coverage for the canonical Workbench controls."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


STREAMLIT_APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _pipeline_page() -> AppTest:
    app = AppTest.from_file(str(STREAMLIT_APP.parent / "views" / "new_run.py"), default_timeout=30).run()
    assert not app.exception
    app.session_state["locale"] = "en"
    app.run()
    assert not app.exception
    return app


def test_research_backtest_disabled_surface_has_no_legacy_controls() -> None:
    app = _pipeline_page()
    source = STREAMLIT_APP.read_text(encoding="utf-8")
    assert "legacy" not in source.lower()
    enable = next(widget for widget in app.checkbox if widget.label == "Enable Research Backtest")
    enable.set_value(False)
    app.run()
    assert not app.exception
    labels = [widget.label for widget in app.checkbox]
    assert "Set Maximum Weight" in labels
    assert "Enable Research Backtest" in labels
    assert "skip_fetch" not in labels and "skip_plot" not in labels
    assert "Transaction Cost (bps)" not in [widget.label for widget in app.number_input]


def test_research_backtest_enabled_controls_build_without_widget_errors() -> None:
    app = _pipeline_page()
    assert next(widget for widget in app.checkbox if widget.label == "Enable Research Backtest").value
    assert "Transaction Cost (bps)" in [widget.label for widget in app.number_input]
    assert "Annual Risk-Free Rate" in [widget.label for widget in app.number_input]
    assert "Benchmark Code" in [widget.label for widget in app.text_input]
    source = STREAMLIT_APP.read_text(encoding="utf-8")
    assert "Research Backtest End Date" not in source
    assert "Backtest frequency" not in source


def test_portfolio_controls_are_conditional_and_use_ui_suggestions() -> None:
    app = _pipeline_page()
    method = next(widget for widget in app.selectbox if widget.label == "Portfolio Method")
    assert method.value == "equal_weight"
    number_labels = [widget.label for widget in app.number_input]
    assert "Lookback Trading Days" not in number_labels
    assert "Minimum Observations" not in number_labels
    assert "Maximum Weight (%)" not in number_labels

    method.set_value("inverse_volatility")
    cap = next(widget for widget in app.checkbox if widget.label == "Set Maximum Weight")
    cap.set_value(True)
    app.run()
    assert not app.exception
    values = {widget.label: widget.value for widget in app.number_input}
    assert values["Lookback Trading Days"] == 60
    assert values["Minimum Observations"] == 40
    assert values["Maximum Weight (%)"] == 20.0
