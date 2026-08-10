"""Pure in-memory Top-N selection and portfolio-weight Holdings construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.holdings.contracts import (
    HOLDINGS_OUTPUT_COLUMNS,
    HoldingsDataError,
    validate_holdings_columns,
)
from src.signals.contracts import SIGNAL_KEY_COLUMNS, SIGNAL_OUTPUT_COLUMNS
from src.portfolio_construction import (
    PortfolioConstructionConfig,
    PortfolioConstructionEngine,
    PortfolioConstructionRequest,
)


WEIGHT_SUM_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class HoldingsDateCount:
    """Deterministic per-date universe and selection audit record."""

    trade_date: pd.Timestamp
    available_count: int
    selected_count: int
    partial: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.strftime("%Y-%m-%d"),
            "available_count": self.available_count,
            "selected_count": self.selected_count,
            "partial": self.partial,
        }


@dataclass(frozen=True)
class HoldingsBuildAudit:
    """Immutable summary of one Holdings build."""

    input_rows: int
    output_rows: int
    trade_date_count: int
    first_trade_date: pd.Timestamp
    last_trade_date: pd.Timestamp
    requested_top_n: int
    insufficient_universe_policy: str
    weighting: str
    per_date_counts: tuple[HoldingsDateCount, ...]
    partial_dates: tuple[pd.Timestamp, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "trade_date_count": self.trade_date_count,
            "min_trade_date": self.first_trade_date.strftime("%Y-%m-%d"),
            "max_trade_date": self.last_trade_date.strftime("%Y-%m-%d"),
            "requested_top_n": self.requested_top_n,
            "insufficient_universe_policy": self.insufficient_universe_policy,
            "weighting": self.weighting,
            "per_date_counts": [item.as_dict() for item in self.per_date_counts],
            "partial_dates": [item.strftime("%Y-%m-%d") for item in self.partial_dates],
            "warnings": list(self.warnings),
        }


class HoldingsBuildResult:
    """Defensively expose canonical Holdings and immutable build audit."""

    __slots__ = ("_holdings", "_audit")

    def __init__(self, holdings: pd.DataFrame, audit: HoldingsBuildAudit) -> None:
        if not isinstance(holdings, pd.DataFrame) or holdings.empty:
            raise HoldingsDataError("holdings must be a non-empty DataFrame.")
        validate_holdings_columns(holdings.columns)
        if not isinstance(audit, HoldingsBuildAudit) or len(holdings) != audit.output_rows:
            raise HoldingsDataError("Holdings result and audit are inconsistent.")
        self._holdings = holdings.copy(deep=True)
        self._audit = audit

    @property
    def holdings(self) -> pd.DataFrame:
        return self._holdings.copy(deep=True)

    @property
    def audit(self) -> HoldingsBuildAudit:
        return self._audit


class HoldingsBuilder:
    """Select existing Signal ranks and assign validated long-only weights."""

    def __init__(
        self, engine: PortfolioConstructionEngine | None = None
    ) -> None:
        if engine is not None and not isinstance(
            engine, PortfolioConstructionEngine
        ):
            raise HoldingsDataError(
                "engine must be PortfolioConstructionEngine."
            )
        self._engine = engine or PortfolioConstructionEngine()

    def build(
        self,
        signals: pd.DataFrame,
        *,
        top_n: int,
        insufficient_universe_policy: str,
        weighting: str,
        portfolio_construction: PortfolioConstructionConfig | None = None,
    ) -> HoldingsBuildResult:
        """Build Holdings using explicit effective config with no business defaults."""
        frame = self._validated_signal(signals)
        requested = self._top_n(top_n)
        policy = self._policy(insufficient_universe_policy)
        weighting_value = self._weighting(weighting)
        construction = portfolio_construction or PortfolioConstructionConfig(
            "equal_weight", {}
        )
        if not isinstance(construction, PortfolioConstructionConfig):
            raise HoldingsDataError(
                "portfolio_construction must be PortfolioConstructionConfig."
            )

        available = frame.groupby("trade_date", sort=False).size()
        insufficient = available[available < requested]
        if policy == "error" and not insufficient.empty:
            trade_date = pd.Timestamp(insufficient.index[0]).date().isoformat()
            count = int(insufficient.iloc[0])
            raise HoldingsDataError(
                "insufficient universe on trade_date "
                f"{trade_date}: requested top_n={requested}, available_count={count}."
            )

        selected = frame.loc[frame["rank"] <= requested].copy(deep=True)
        selected_counts = selected.groupby("trade_date", sort=False).size()
        weighted_groups: list[pd.DataFrame] = []
        for trade_date, group in selected.groupby("trade_date", sort=False):
            candidates = group.loc[:, ["ts_code", "score", "rank"]].copy(
                deep=True
            )
            candidates["selection_position"] = np.arange(
                1, len(candidates) + 1, dtype=np.int64
            )
            request = PortfolioConstructionRequest(trade_date, candidates)
            constructed = self._engine.construct(request, construction)
            weights = constructed.weights.set_index("ts_code")["target_weight"]
            weighted = group.copy(deep=True)
            weighted.insert(
                2,
                "target_weight",
                weighted["ts_code"].map(weights).astype(np.float64),
            )
            weighted_groups.append(weighted)
        selected = pd.concat(weighted_groups, ignore_index=True)
        output = selected.loc[:, list(HOLDINGS_OUTPUT_COLUMNS)].copy(deep=True)
        output = output.sort_values(
            ["trade_date", "rank", "ts_code"], kind="mergesort"
        ).reset_index(drop=True)

        per_date: list[HoldingsDateCount] = []
        partial_dates: list[pd.Timestamp] = []
        for trade_date, count in available.items():
            timestamp = pd.Timestamp(trade_date)
            selected_count = int(selected_counts.loc[trade_date])
            partial = int(count) < requested
            per_date.append(
                HoldingsDateCount(timestamp, int(count), selected_count, partial)
            )
            if partial:
                partial_dates.append(timestamp)
        warnings = tuple(
            f"partial universe on {item.strftime('%Y-%m-%d')}"
            for item in partial_dates
        )
        audit = HoldingsBuildAudit(
            input_rows=len(frame),
            output_rows=len(output),
            trade_date_count=len(per_date),
            first_trade_date=pd.Timestamp(frame["trade_date"].min()),
            last_trade_date=pd.Timestamp(frame["trade_date"].max()),
            requested_top_n=requested,
            insufficient_universe_policy=policy,
            weighting=weighting_value,
            per_date_counts=tuple(per_date),
            partial_dates=tuple(partial_dates),
            warnings=warnings,
        )
        return HoldingsBuildResult(output, audit)

    @staticmethod
    def _top_n(value: object) -> int:
        if type(value) is not int:
            raise HoldingsDataError("top_n must be a strict int.")
        if value < 1:
            raise HoldingsDataError("top_n must be >= 1.")
        return value

    @staticmethod
    def _policy(value: object) -> str:
        if not isinstance(value, str):
            raise HoldingsDataError(
                "insufficient_universe_policy must be a string."
            )
        policy = value.strip().lower()
        if policy not in {"error", "allow_partial"}:
            raise HoldingsDataError(
                "insufficient_universe_policy must be 'error' or 'allow_partial'."
            )
        return policy

    @staticmethod
    def _weighting(value: object) -> str:
        if not isinstance(value, str):
            raise HoldingsDataError("weighting must be a string.")
        normalized = value.strip().lower()
        if normalized != "equal_weight":
            raise HoldingsDataError("weighting must be 'equal_weight' for V5.")
        return normalized

    @staticmethod
    def _validated_signal(value: object) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise HoldingsDataError("signals must be a pandas DataFrame.")
        if value.empty:
            raise HoldingsDataError("signals must not be empty.")
        if (
            isinstance(value.columns, pd.MultiIndex)
            or not value.columns.is_unique
            or tuple(value.columns) != SIGNAL_OUTPUT_COLUMNS
        ):
            raise HoldingsDataError(
                f"signals must contain exactly {SIGNAL_OUTPUT_COLUMNS!r}."
            )
        frame = value.copy(deep=True)
        dates = frame["trade_date"]
        if (
            not pd.api.types.is_datetime64_ns_dtype(dates.dtype)
            or getattr(dates.dt, "tz", None) is not None
            or dates.isna().any()
            or not dates.eq(dates.dt.normalize()).all()
        ):
            raise HoldingsDataError("trade_date contract is invalid.")
        codes = frame["ts_code"]
        if (
            codes.isna().any()
            or not codes.map(lambda item: isinstance(item, (str, np.str_))).all()
            or codes.astype("string").str.strip().eq("").any()
            or not codes.astype("string").eq(codes.astype("string").str.strip()).all()
        ):
            raise HoldingsDataError("ts_code contract is invalid.")
        score = frame["score"]
        if (
            pd.api.types.is_bool_dtype(score.dtype)
            or not pd.api.types.is_numeric_dtype(score.dtype)
            or pd.api.types.is_complex_dtype(score.dtype)
        ):
            raise HoldingsDataError("score must be real numeric data.")
        try:
            score_values = score.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise HoldingsDataError("score must be real numeric data.") from exc
        if not np.isfinite(score_values).all():
            raise HoldingsDataError("score must contain only finite values.")
        ranks = frame["rank"]
        if not pd.api.types.is_integer_dtype(ranks.dtype) or bool((ranks <= 0).any()):
            raise HoldingsDataError("rank must be positive integer data.")
        if frame.duplicated(list(SIGNAL_KEY_COLUMNS)).any():
            raise HoldingsDataError("Signal keys must be unique.")
        expected_order = frame.sort_values(
            ["trade_date", "rank", "ts_code"], kind="mergesort"
        ).reset_index(drop=True)
        try:
            pdt.assert_frame_equal(frame.reset_index(drop=True), expected_order)
        except AssertionError as exc:
            raise HoldingsDataError("Signal row order must be canonical.") from exc
        for trade_date, group in frame.groupby("trade_date", sort=False):
            expected = np.arange(1, len(group) + 1, dtype=np.int64)
            if not np.array_equal(group["rank"].to_numpy(), expected):
                date_text = pd.Timestamp(trade_date).date().isoformat()
                raise HoldingsDataError(
                    f"Signal ranks must be unique contiguous 1..N on {date_text}."
                )
        return frame
