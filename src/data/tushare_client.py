"""TuShare Pro client wrapper for project data access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from src.data.credentials import CredentialProvider, EnvironmentCredentialProvider


logger = logging.getLogger(__name__)


class TushareClient:
    """Lazy TuShare Pro client initialized from environment configuration."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(
        self,
        token: str | None = None,
        *,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        """Initialize one instance client without mutating TuShare global state."""
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ValueError("TuShare token must be a non-empty string when provided.")
        provider = credential_provider or EnvironmentCredentialProvider(self.PROJECT_ROOT)
        resolved = token.strip() if token is not None else provider.tushare_token()
        if not resolved:
            raise ValueError(
                "TuShare credential is unavailable. Provide an instance token or "
                "configure the supported environment credential source."
            )
        try:
            self.pro: Any = ts.pro_api(resolved)
        except Exception:
            raise RuntimeError("TuShare client initialization failed.") from None

    def get_stock_basic(self, list_status: str = "L") -> pd.DataFrame:
        """Fetch stock lifecycle basics for one explicit listing status."""
        fields = (
            "ts_code,symbol,name,area,industry,market,exchange,curr_type,list_status,"
            "list_date,delist_date"
        )
        return self.pro.stock_basic(
            exchange="",
            list_status=list_status,
            fields=fields,
        )

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch the exchange trading calendar for a date range.

        Args:
            start_date: Start date in YYYYMMDD format.
            end_date: End date in YYYYMMDD format.

        Returns:
            Trading calendar data with exchange, calendar date, open flag, and
            previous trading date.
        """
        fields = "exchange,cal_date,is_open,pretrade_date"
        return self.pro.trade_cal(
            exchange="",
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_index_weight(
        self,
        index_code: str = "000300.SH",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch index constituent weights for the selected index and period."""
        fields = "index_code,con_code,trade_date,weight"
        return self.pro.index_weight(
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_hs300_components(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Build an HS300 stock pool from unique constituents in a period."""
        weights = self.get_index_weight(
            index_code="000300.SH",
            start_date=start_date,
            end_date=end_date,
        )
        if weights.empty or "con_code" not in weights.columns:
            return pd.DataFrame(columns=["ts_code"])

        components = (
            weights[["con_code"]]
            .dropna()
            .drop_duplicates()
            .rename(columns={"con_code": "ts_code"})
            .reset_index(drop=True)
        )
        return components

    def get_monthly(
        self,
        ts_code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch monthly market data for a stock or date range."""
        fields = (
            "ts_code,trade_date,open,high,low,close,pre_close,change,"
            "pct_chg,vol,amount"
        )
        return self.pro.monthly(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_daily(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch raw security daily market data from TuShare Pro."""
        fields = (
            "ts_code,trade_date,open,high,low,close,pre_close,change,"
            "pct_chg,vol,amount"
        )
        return self.pro.daily(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_index_daily(
        self,
        ts_code: str,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch raw benchmark/index daily market data from TuShare Pro."""
        fields = (
            "ts_code,trade_date,open,high,low,close,pre_close,change,"
            "pct_chg,vol,amount"
        )
        return self.pro.index_daily(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_suspend_d(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        suspend_type: str | None = None,
    ) -> pd.DataFrame:
        """Fetch raw daily suspension/resumption records from TuShare Pro."""
        fields = "ts_code,trade_date,suspend_timing,suspend_type"
        result = self.pro.suspend_d(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            suspend_type=suspend_type,
            fields=fields,
        )
        # TuShare represents a proven zero-event day as an empty DataFrame and
        # may omit every column. Normalize only that explicit protocol shape;
        # non-empty malformed responses remain subject to strict validation.
        if isinstance(result, pd.DataFrame) and result.empty:
            return pd.DataFrame(columns=fields.split(","))
        return result

    def get_daily_basic(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily valuation and liquidity indicators from TuShare Pro."""
        fields = (
            "ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,"
            "pb,ps,ps_ttm,dv_ratio,dv_ttm,total_mv,circ_mv"
        )
        return self.pro.daily_basic(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def get_adj_factor(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch raw daily adjustment factors without applying adjustments."""
        return self.pro.adj_factor(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,adj_factor",
        )

    @staticmethod
    def save_csv(df: pd.DataFrame, path: str | Path) -> None:
        """Save a non-empty DataFrame to CSV using utf-8-sig encoding.

        Args:
            df: DataFrame to persist.
            path: Target CSV file path.
        """
        if df.empty:
            logger.warning("DataFrame is empty; skip saving CSV to %s.", path)
            return

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
