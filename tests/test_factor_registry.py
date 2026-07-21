"""Tests for V2-A factor metadata, calculation, and registry behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from src.factors.base import FactorMetadata, FunctionFactor
from src.factors.examples import MOMENTUM_20D, VOLATILITY_20D, register_example_factors
from src.factors.registry import FactorRegistry, create_default_registry


def make_factor(name: str, category: str = "test", multiplier: float = 1.0) -> FunctionFactor:
    """Create a small valid factor for isolated registry tests."""
    metadata = FactorMetadata(
        name=name,
        category=category,
        direction=1,
        required_datasets=("daily",),
        source_fields=("close",),
        lookback_days=0,
        frequency="daily",
        availability_lag_days=0,
        description="Registry test factor.",
        version="1.0",
    )
    return FunctionFactor(metadata, lambda data: data["close"] * multiplier)


def make_close_data(size: int = 60) -> pd.DataFrame:
    """Create deterministic local close data with a non-default index."""
    index = pd.date_range("2024-01-01", periods=size, freq="D")
    return pd.DataFrame({"close": range(100, 100 + size)}, index=index)


def test_factor_metadata_normal_creation() -> None:
    metadata = make_factor("valid_factor").metadata

    assert metadata.name == "valid_factor"
    assert metadata.required_datasets == ("daily",)
    assert metadata.source_fields == ("close",)


def test_factor_metadata_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        FactorMetadata(name="invalid", category="test", direction=0)


def test_factor_metadata_rejects_negative_lookback() -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        FactorMetadata(name="invalid", category="test", direction=1, lookback_days=-1)


def test_factor_metadata_rejects_negative_availability_lag() -> None:
    with pytest.raises(ValueError, match="availability_lag_days"):
        FactorMetadata(
            name="invalid",
            category="test",
            direction=1,
            availability_lag_days=-1,
        )


def test_register_and_get_factor_by_name() -> None:
    registry = FactorRegistry()
    factor = make_factor("alpha")
    registry.register(factor)

    assert registry.get("alpha") is factor
    assert registry.contains("alpha")
    assert not registry.contains("missing")


def test_duplicate_registration_raises_by_default() -> None:
    registry = FactorRegistry()
    registry.register(make_factor("alpha"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_factor("alpha"))


def test_registration_can_explicitly_override() -> None:
    registry = FactorRegistry()
    original = make_factor("alpha")
    replacement = make_factor("alpha", multiplier=2.0)
    registry.register(original)

    registry.register(replacement, allow_override=True)

    assert registry.get("alpha") is replacement


def test_unregister_removes_and_returns_factor() -> None:
    registry = FactorRegistry()
    factor = make_factor("alpha")
    registry.register(factor)

    removed = registry.unregister("alpha")

    assert removed is factor
    assert not registry.contains("alpha")


def test_missing_factor_operations_raise_key_error() -> None:
    registry = FactorRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister("missing")


def test_category_filter_and_metadata_listing() -> None:
    registry = FactorRegistry()
    registry.register(make_factor("zeta", category="value"))
    registry.register(make_factor("alpha", category="momentum"))
    registry.register(make_factor("beta", category="momentum"))

    assert [factor.metadata.name for factor in registry.get_by_category("momentum")] == [
        "alpha",
        "beta",
    ]
    assert [metadata.name for metadata in registry.list_metadata()] == [
        "alpha",
        "beta",
        "zeta",
    ]


def test_list_names_is_stably_sorted_and_clear_is_isolated() -> None:
    registry = FactorRegistry()
    registry.register(make_factor("zeta"))
    registry.register(make_factor("alpha"))

    assert registry.list_names() == ["alpha", "zeta"]
    registry.clear()
    assert registry.list_names() == []


def test_registration_requires_metadata_and_compute() -> None:
    registry = FactorRegistry()

    with pytest.raises(TypeError, match="FactorMetadata"):
        registry.register(object())  # type: ignore[arg-type]


def test_missing_close_field_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="close"):
        MOMENTUM_20D.compute(pd.DataFrame({"open": [1.0, 2.0]}))


def test_momentum_output_alignment_and_warmup() -> None:
    data = make_close_data()
    result = MOMENTUM_20D.compute(data)

    assert len(result) == len(data)
    assert result.index.equals(data.index)
    assert result.iloc[:20].isna().all()
    assert result.iloc[20:].notna().all()


def test_volatility_output_alignment() -> None:
    data = make_close_data()
    result = VOLATILITY_20D.compute(data)

    assert len(result) == len(data)
    assert result.index.equals(data.index)


@pytest.mark.parametrize("factor", [MOMENTUM_20D, VOLATILITY_20D])
def test_example_factors_do_not_use_future_values(factor: FunctionFactor) -> None:
    original = make_close_data()
    changed = original.copy()
    changed.iloc[-10:, changed.columns.get_loc("close")] *= 10

    before = factor.compute(original)
    after = factor.compute(changed)

    pd.testing.assert_series_equal(before.iloc[:-10], after.iloc[:-10])


def test_example_registration_and_default_registry_are_independent() -> None:
    first = FactorRegistry()
    register_example_factors(first)
    second = create_default_registry()

    assert first.list_names() == ["momentum_20d", "volatility_20d"]
    assert second.list_names() == first.list_names()
    first.clear()
    assert second.list_names() == ["momentum_20d", "volatility_20d"]
