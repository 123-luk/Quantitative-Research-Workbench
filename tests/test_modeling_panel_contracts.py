"""Tests for the pure in-memory V4-A Modeling Panel contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone
import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.modeling_panel import (
    MODELING_PANEL_AUDIT_COLUMNS,
    MODELING_PANEL_KEY_COLUMNS,
    MODELING_PANEL_SCHEMA_VERSION,
    ModelingPanelAlignmentError,
    ModelingPanelAudit,
    ModelingPanelConfig,
    ModelingPanelConfigError,
    ModelingPanelDataError,
    ModelingPanelError,
    ModelingPanelIntegrityError,
    ModelingPanelLeakageError,
    ModelingPanelResult,
    ModelingPanelUnmatchedAudit,
)


FEATURE_NAMES = ("factor_a", "factor_b")


def _empty_unmatched() -> ModelingPanelUnmatchedAudit:
    return ModelingPanelUnmatchedAudit(0, 0, None, None)


def _config(**overrides: object) -> ModelingPanelConfig:
    values: dict[str, object] = {}
    values.update(overrides)
    return ModelingPanelConfig(**values)  # type: ignore[arg-type]


def _audit(**overrides: object) -> ModelingPanelAudit:
    config = overrides.get("config", ModelingPanelConfig())
    label = (
        config.label_column
        if isinstance(config, ModelingPanelConfig)
        else "forward_return"
    )
    values: dict[str, object] = {
        "schema_version": MODELING_PANEL_SCHEMA_VERSION,
        "config": config,
        "label_column": label,
        "factor_input_rows": 2,
        "return_input_rows": 2,
        "matched_rows": 2,
        "output_rows": 2,
        "factor_only": _empty_unmatched(),
        "return_only": _empty_unmatched(),
        "date_count": 2,
        "security_count": 2,
        "first_trade_date": pd.Timestamp("2024-01-02 12:30"),
        "last_trade_date": pd.Timestamp("2024-01-03 08:00"),
        "first_entry_trade_date": pd.Timestamp("2024-01-03"),
        "last_entry_trade_date": pd.Timestamp("2024-01-04"),
        "first_exit_trade_date": pd.Timestamp("2024-01-04"),
        "last_exit_trade_date": pd.Timestamp("2024-01-05"),
        "feature_count": 2,
        "feature_names": FEATURE_NAMES,
        "feature_missing_counts": (("factor_a", 0), ("factor_b", 1)),
        "feature_missing_rates": (("factor_a", 0.0), ("factor_b", 0.5)),
        "feature_non_finite_counts": (("factor_a", 0), ("factor_b", 0)),
        "all_missing_features": (),
        "constant_features": ("factor_a",),
        "suspicious_feature_names": (),
        "label_missing_count": 0,
        "label_non_finite_count": 0,
        "duplicate_factor_key_count": 0,
        "duplicate_return_key_count": 0,
        "entry_before_signal_count": 0,
        "entry_equal_signal_count": 0,
        "exit_not_after_entry_count": 0,
        "label_formula_mismatch_count": 0,
        "per_date_security_count_min": 1,
        "per_date_security_count_median": 1.0,
        "per_date_security_count_max": 1,
        "per_security_observation_count_min": 1,
        "per_security_observation_count_median": 1.0,
        "per_security_observation_count_max": 1,
        "warnings": ("factor_a is constant",),
    }
    values.update(overrides)
    if "config" in overrides and "label_column" not in overrides:
        supplied = overrides["config"]
        if isinstance(supplied, ModelingPanelConfig):
            values["label_column"] = supplied.label_column
    return ModelingPanelAudit(**values)  # type: ignore[arg-type]


def _panel(label: str = "forward_return") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ts_code": pd.Series(["000001.SZ", "000002.SZ"], dtype="string"),
            "factor_a": [1.0, 1.0],
            "factor_b": [np.nan, 2.0],
            "entry_trade_date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "exit_trade_date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "entry_price": [10.0, 20.0],
            "exit_price": [11.0, 22.0],
            label: [0.1, 0.1],
        }
    )


def _result(
    *,
    panel: pd.DataFrame | None = None,
    config: ModelingPanelConfig | None = None,
    audit: ModelingPanelAudit | None = None,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    label_column: str | None = None,
    schema_version: str = MODELING_PANEL_SCHEMA_VERSION,
) -> ModelingPanelResult:
    config = ModelingPanelConfig() if config is None else config
    label = config.label_column if label_column is None else label_column
    audit = _audit(config=config) if audit is None else audit
    return ModelingPanelResult(
        _panel(label) if panel is None else panel,
        feature_names=feature_names,
        label_column=label,
        audit=audit,
        config=config,
        schema_version=schema_version,
    )


def test_public_constants_and_exception_hierarchy() -> None:
    assert MODELING_PANEL_SCHEMA_VERSION == "1.0"
    assert MODELING_PANEL_KEY_COLUMNS == ("trade_date", "ts_code")
    assert MODELING_PANEL_AUDIT_COLUMNS == (
        "entry_trade_date",
        "exit_trade_date",
        "entry_price",
        "exit_price",
    )
    for exception in (
        ModelingPanelConfigError,
        ModelingPanelDataError,
        ModelingPanelAlignmentError,
        ModelingPanelLeakageError,
        ModelingPanelIntegrityError,
    ):
        assert exception.__bases__ == (ModelingPanelError,)
        assert issubclass(exception, ValueError)


def test_config_defaults_normalization_round_trip_and_identity() -> None:
    config = ModelingPanelConfig.from_dict(
        {
            "label_column": " target ",
            "include_features": [" factor_b ", "factor_a"],
            "unmatched_policy": " ERROR ",
            "require_entry_after_signal": False,
            "allow_missing_labels": False,
        }
    )
    assert config.label_column == "target"
    assert config.include_features == ("factor_b", "factor_a")
    assert config.exclude_features == ()
    assert config.unmatched_policy == "error"
    assert ModelingPanelConfig.from_dict(config) is config
    assert ModelingPanelConfig.from_dict(config.as_dict()) == config
    assert ModelingPanelConfig.from_dict(None) == ModelingPanelConfig()
    payload = config.as_dict()
    payload["include_features"].append("mutated")
    assert config.include_features == ("factor_b", "factor_a")
    json.dumps(config.as_dict(), allow_nan=False)


@pytest.mark.parametrize("value", [1, [], "x", object()])
def test_config_from_dict_requires_supported_input(value: object) -> None:
    with pytest.raises(ModelingPanelConfigError, match="mapping"):
        ModelingPanelConfig.from_dict(value)  # type: ignore[arg-type]


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ModelingPanelConfigError, match="Unknown.*extra"):
        ModelingPanelConfig.from_dict({"extra": 1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label_column", ""),
        ("label_column", 1),
        ("label_column", "trade_date"),
        ("label_column", "entry_price"),
        ("include_features", []),
        ("include_features", "factor_a"),
        ("include_features", ["factor_a", " factor_a "]),
        ("include_features", ["forward_return"]),
        ("exclude_features", None),
        ("exclude_features", "factor_a"),
        ("exclude_features", ["ts_code"]),
        ("unmatched_policy", "drop"),
        ("unmatched_policy", 1),
        ("require_entry_after_signal", 1),
        ("allow_missing_labels", "true"),
    ],
)
def test_config_rejects_invalid_field_values(field: str, value: object) -> None:
    with pytest.raises(ModelingPanelConfigError, match=field):
        _config(**{field: value})


def test_config_rejects_include_exclude_combination_and_is_frozen() -> None:
    with pytest.raises(ModelingPanelConfigError, match="cannot both"):
        _config(include_features=("a",), exclude_features=("b",))
    config = ModelingPanelConfig()
    with pytest.raises(FrozenInstanceError):
        config.label_column = "other"  # type: ignore[misc]


def test_unmatched_audit_normalizes_dates_and_serializes_samples() -> None:
    audit = ModelingPanelUnmatchedAudit(
        row_count=3,
        date_count=2,
        first_trade_date=datetime(2024, 1, 2, 12),
        last_trade_date=date(2024, 1, 3),
        sampled_keys=(
            (pd.Timestamp("2024-01-02 10:00"), " A "),
            (pd.Timestamp("2024-01-03"), "B"),
        ),
    )
    assert audit.first_trade_date == pd.Timestamp("2024-01-02")
    assert audit.sampled_keys[0] == (pd.Timestamp("2024-01-02"), "A")
    assert audit.as_dict()["sampled_keys"] == [
        {"trade_date": "2024-01-02", "ts_code": "A"},
        {"trade_date": "2024-01-03", "ts_code": "B"},
    ]
    json.dumps(audit.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"row_count": True},
        {"row_count": -1},
        {"date_count": 2, "row_count": 1},
        {"row_count": 0, "date_count": 1},
        {"row_count": 0, "first_trade_date": pd.Timestamp("2024-01-01")},
        {
            "row_count": 1,
            "date_count": 0,
            "first_trade_date": None,
            "last_trade_date": None,
        },
        {
            "row_count": 1,
            "date_count": 1,
            "first_trade_date": pd.Timestamp("2024-01-03"),
            "last_trade_date": pd.Timestamp("2024-01-02"),
        },
        {"sampled_keys": []},
        {"sampled_keys": ((pd.Timestamp("2024-01-01"),),)},
        {"sampled_keys": ((pd.NaT, "A"),)},
        {"sampled_keys": ((pd.Timestamp("2024-01-01", tz="UTC"), "A"),)},
    ],
)
def test_unmatched_audit_rejects_invalid_shapes_and_counts(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "row_count": 0,
        "date_count": 0,
        "first_trade_date": None,
        "last_trade_date": None,
        "sampled_keys": (),
    }
    values.update(kwargs)
    with pytest.raises(ModelingPanelDataError):
        ModelingPanelUnmatchedAudit(**values)  # type: ignore[arg-type]


def test_unmatched_audit_rejects_unsorted_duplicate_out_of_range_and_oversized() -> None:
    base = {
        "row_count": 2,
        "date_count": 2,
        "first_trade_date": pd.Timestamp("2024-01-02"),
        "last_trade_date": pd.Timestamp("2024-01-03"),
    }
    bad_samples = (
        (
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-02"), "A"),
        ),
        (
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
        ),
        ((pd.Timestamp("2024-01-01"), "A"),),
    )
    for samples in bad_samples:
        with pytest.raises(ModelingPanelDataError, match="sampled_keys"):
            ModelingPanelUnmatchedAudit(**base, sampled_keys=samples)
    with pytest.raises(ModelingPanelDataError, match="20"):
        ModelingPanelUnmatchedAudit(
            row_count=21,
            date_count=1,
            first_trade_date=pd.Timestamp("2024-01-02"),
            last_trade_date=pd.Timestamp("2024-01-02"),
            sampled_keys=tuple(
                (pd.Timestamp("2024-01-02"), f"{index:02d}") for index in range(21)
            ),
        )


def test_audit_normalizes_dates_is_frozen_and_json_safe() -> None:
    audit = _audit()
    assert audit.first_trade_date == pd.Timestamp("2024-01-02")
    assert audit.last_trade_date == pd.Timestamp("2024-01-03")
    payload = audit.as_dict()
    assert payload["first_trade_date"] == "2024-01-02"
    assert payload["feature_missing_counts"] == {"factor_a": 0, "factor_b": 1}
    json.dumps(payload, allow_nan=False)
    payload["feature_names"].append("mutated")
    payload["config"]["exclude_features"].append("mutated")
    assert audit.feature_names == FEATURE_NAMES
    assert audit.config.exclude_features == ()
    with pytest.raises(FrozenInstanceError):
        audit.output_rows = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", "2.0", ModelingPanelIntegrityError),
        ("config", {}, ModelingPanelIntegrityError),
        ("label_column", "other", ModelingPanelIntegrityError),
        ("factor_only", {}, ModelingPanelIntegrityError),
        ("return_only", {}, ModelingPanelIntegrityError),
        ("output_rows", 0, ModelingPanelDataError),
        ("date_count", 0, ModelingPanelDataError),
        ("security_count", 0, ModelingPanelDataError),
        ("feature_count", 0, ModelingPanelDataError),
        ("label_missing_count", 3, ModelingPanelDataError),
        ("duplicate_factor_key_count", 3, ModelingPanelDataError),
        ("per_date_security_count_min", 0, ModelingPanelDataError),
        ("per_date_security_count_median", float("nan"), ModelingPanelDataError),
        ("per_date_security_count_max", True, ModelingPanelDataError),
        ("warnings", [], ModelingPanelDataError),
    ],
)
def test_audit_rejects_invalid_scalar_fields(
    field: str, value: object, error: type[ModelingPanelError]
) -> None:
    with pytest.raises(error, match=field):
        _audit(**{field: value})


def test_audit_rejects_alignment_equation_violations() -> None:
    with pytest.raises(ModelingPanelAlignmentError, match="matched_rows"):
        _audit(matched_rows=1)
    with pytest.raises(ModelingPanelAlignmentError, match="factor_only"):
        _audit(
            factor_input_rows=3,
            factor_only=_empty_unmatched(),
        )
    with pytest.raises(ModelingPanelAlignmentError, match="return_only"):
        _audit(
            return_input_rows=3,
            return_only=_empty_unmatched(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"first_trade_date": pd.Timestamp("2024-01-04")},
        {"first_trade_date": pd.Timestamp("2024-01-02", tz="UTC")},
        {"first_entry_trade_date": None},
        {
            "first_exit_trade_date": pd.Timestamp("2024-01-06"),
            "last_exit_trade_date": pd.Timestamp("2024-01-05"),
        },
    ],
)
def test_audit_rejects_invalid_date_contracts(overrides: dict[str, object]) -> None:
    with pytest.raises(ModelingPanelDataError):
        _audit(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_names": ["factor_a", "factor_b"]},
        {"feature_names": ("factor_a", "factor_a")},
        {"feature_names": ("factor_a", "entry_price")},
        {"feature_count": 1},
        {"feature_missing_counts": (("factor_b", 1), ("factor_a", 0))},
        {"feature_missing_counts": (("factor_a", 0), ("factor_b", 3))},
        {"feature_missing_rates": (("factor_a", 0.0), ("factor_b", 0.4))},
        {"feature_missing_rates": (("factor_a", 0.0), ("factor_b", float("inf")))},
        {"feature_non_finite_counts": (("factor_a", 0), ("factor_b", True))},
        {"all_missing_features": ("factor_b",)},
        {"constant_features": ("factor_b", "factor_a")},
        {"suspicious_feature_names": ("unknown",)},
    ],
)
def test_audit_rejects_invalid_feature_metadata(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ModelingPanelError):
        _audit(**overrides)


def test_audit_accepts_bounded_unmatched_metadata() -> None:
    factor_only = ModelingPanelUnmatchedAudit(
        1,
        1,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
        ((pd.Timestamp("2024-01-01"), "A"),),
    )
    return_only = ModelingPanelUnmatchedAudit(
        2,
        1,
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-05"),
    )
    audit = _audit(
        factor_input_rows=3,
        return_input_rows=4,
        factor_only=factor_only,
        return_only=return_only,
    )
    assert audit.factor_only.row_count == 1
    assert audit.return_only.row_count == 2


def test_result_public_properties_and_defensive_copies() -> None:
    source = _panel()
    result = _result(panel=source)
    source.loc[0, "factor_a"] = -1.0
    exposed = result.panel
    exposed.loc[0, "factor_a"] = -2.0
    assert result.panel.loc[0, "factor_a"] == 1.0
    assert result.feature_names == FEATURE_NAMES
    assert result.label_column == "forward_return"
    assert result.schema_version == MODELING_PANEL_SCHEMA_VERSION
    assert isinstance(result.audit, ModelingPanelAudit)
    assert isinstance(result.config, ModelingPanelConfig)
    with pytest.raises(AttributeError):
        result.feature_names = ("other",)  # type: ignore[misc]


@pytest.mark.parametrize("panel", [None, [], pd.DataFrame()])
def test_result_requires_nonempty_dataframe(panel: object) -> None:
    with pytest.raises(ModelingPanelDataError, match="panel"):
        ModelingPanelResult(
            panel,  # type: ignore[arg-type]
            feature_names=FEATURE_NAMES,
            label_column="forward_return",
            audit=_audit(),
            config=ModelingPanelConfig(),
        )


def test_result_requires_unique_exact_ordered_columns() -> None:
    panel = _panel()
    with pytest.raises(ModelingPanelIntegrityError, match="columns"):
        _result(panel=panel[list(reversed(panel.columns))])
    with pytest.raises(ModelingPanelIntegrityError, match="columns"):
        _result(panel=panel.assign(extra=1))
    duplicate = pd.concat([panel, panel[["factor_a"]]], axis=1)
    with pytest.raises(ModelingPanelDataError, match="unique"):
        _result(panel=duplicate)


@pytest.mark.parametrize(
    "feature_names",
    [
        [],
        (),
        ("factor_a", "factor_a"),
        ("factor_a", "entry_price"),
        ("factor_a", "forward_return"),
    ],
)
def test_result_rejects_invalid_feature_names(feature_names: object) -> None:
    with pytest.raises(ModelingPanelError, match="feature_names"):
        _result(feature_names=feature_names)  # type: ignore[arg-type]


def test_result_requires_consistent_config_audit_and_schema() -> None:
    custom = _config(label_column="target")
    with pytest.raises(ModelingPanelIntegrityError, match="config"):
        _result(config=custom, audit=_audit())
    with pytest.raises(ModelingPanelIntegrityError, match="label_column"):
        _result(label_column="target")
    with pytest.raises(ModelingPanelIntegrityError, match="schema_version"):
        _result(schema_version="2.0")
    with pytest.raises(ModelingPanelIntegrityError, match="row count"):
        _result(panel=pd.concat([_panel(), _panel()], ignore_index=True))
    with pytest.raises(ModelingPanelIntegrityError, match="schema"):
        _result(feature_names=("factor_b", "factor_a"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"duplicate_factor_key_count": 1},
        {"duplicate_return_key_count": 1},
        {"entry_before_signal_count": 1},
        {"entry_equal_signal_count": 1},
        {"exit_not_after_entry_count": 1},
        {"label_formula_mismatch_count": 1},
        {"label_non_finite_count": 1},
        {
            "feature_non_finite_counts": (
                ("factor_a", 1),
                ("factor_b", 0),
            )
        },
    ],
)
def test_result_rejects_unsuccessful_audit_states(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ModelingPanelIntegrityError):
        _result(audit=_audit(**overrides))


def test_result_policy_dependent_success_state_checks() -> None:
    relaxed = _config(require_entry_after_signal=False)
    _result(config=relaxed, audit=_audit(config=relaxed, entry_equal_signal_count=1))

    strict_unmatched = _config(unmatched_policy="error")
    factor_only = ModelingPanelUnmatchedAudit(
        1,
        1,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
    )
    with pytest.raises(ModelingPanelAlignmentError, match="unmatched_policy"):
        _result(
            config=strict_unmatched,
            audit=_audit(
                config=strict_unmatched,
                factor_input_rows=3,
                factor_only=factor_only,
            ),
        )

    no_missing = _config(allow_missing_labels=False)
    with pytest.raises(ModelingPanelIntegrityError, match="allow_missing_labels"):
        _result(
            config=no_missing,
            audit=_audit(config=no_missing, label_missing_count=1),
        )


def test_result_rejects_all_missing_features() -> None:
    audit = _audit(
        feature_missing_counts=(("factor_a", 2), ("factor_b", 1)),
        feature_missing_rates=(("factor_a", 1.0), ("factor_b", 0.5)),
        all_missing_features=("factor_a",),
    )
    with pytest.raises(ModelingPanelIntegrityError, match="all-missing"):
        _result(audit=audit)


def test_contracts_expose_no_builder_or_persistence_methods() -> None:
    result = _result()
    for method_name in ("build", "write", "save", "load", "validate", "as_dict"):
        assert not hasattr(result, method_name)
    assert not hasattr(ModelingPanelUnmatchedAudit, "from_dict")
    assert not hasattr(ModelingPanelAudit, "from_dict")


def test_timezone_aware_python_datetime_is_rejected() -> None:
    with pytest.raises(ModelingPanelDataError, match="timezone-naive"):
        ModelingPanelUnmatchedAudit(
            1,
            1,
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )


def test_result_does_not_modify_panel_dtypes_or_values() -> None:
    panel = _panel()
    before = panel.copy(deep=True)
    result = _result(panel=panel)
    pdt.assert_frame_equal(result.panel, before)
    pdt.assert_frame_equal(panel, before)
