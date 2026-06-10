"""Command-line utility for fetching raw TuShare data."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tushare_client import TushareClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the TuShare fetch script."""
    parser = argparse.ArgumentParser(
        description="Fetch raw TuShare data for the quant factor system."
    )
    parser.add_argument(
        "--start",
        default="20150101",
        help="Start date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--end",
        default="20251231",
        help="End date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--universe",
        choices=("hs300", "all"),
        default="hs300",
        help="Stock universe: hs300 for CSI 300 or all for all A-shares.",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=50,
        help="Limit the number of stocks for debugging. Use 0 for no limit.",
    )
    parser.add_argument(
        "--skip-daily-basic",
        action="store_true",
        help="Skip fetching TuShare daily_basic data.",
    )
    parser.add_argument(
        "--skip-monthly",
        action="store_true",
        help="Skip fetching TuShare monthly data.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to wait between API calls.",
    )
    return parser.parse_args()


def limit_codes(codes: Iterable[str], max_stocks: int) -> list[str]:
    """Limit a stock code iterable for debugging runs.

    Args:
        codes: Stock codes to be requested from TuShare.
        max_stocks: Maximum number of stocks to keep. Values <= 0 disable the
            limit.

    Returns:
        A list of stock codes, optionally truncated to max_stocks.
    """
    code_list = [code for code in codes if isinstance(code, str) and code]
    if max_stocks > 0:
        return code_list[:max_stocks]
    return code_list


def fetch_monthly_data(
    client: TushareClient,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    """Fetch and combine monthly data for multiple stock codes."""
    frames: list[pd.DataFrame] = []
    for ts_code in codes:
        try:
            frame = client.get_monthly(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to fetch monthly for {ts_code}: {exc}")
        finally:
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    if not frames:
        warnings.warn("All monthly fetch results are empty; monthly.csv not saved.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_daily_basic_data(
    client: TushareClient,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    """Fetch and combine daily_basic data for multiple stock codes."""
    frames: list[pd.DataFrame] = []
    for ts_code in codes:
        try:
            frame = client.get_daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                "Failed to fetch daily_basic for "
                f"{ts_code}; TuShare may not support this parameter "
                f"combination: {exc}"
            )
        finally:
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    if not frames:
        warnings.warn(
            "All daily_basic fetch results are empty; daily_basic.csv not saved."
        )
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_and_report(
    client: TushareClient,
    df: pd.DataFrame,
    path: Path,
    outputs: list[tuple[str, int, Path]],
) -> None:
    """Save a DataFrame with the client helper and record output metadata."""
    if df.empty:
        warnings.warn(f"{path.name} is empty; file not saved.")
        return

    client.save_csv(df, path)
    outputs.append((path.name, len(df), path))


def main() -> None:
    """Fetch raw TuShare datasets according to command-line options."""
    args = parse_args()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    client = TushareClient()
    outputs: list[tuple[str, int, Path]] = []

    stock_basic = client.get_stock_basic()
    save_and_report(client, stock_basic, raw_dir / "stock_basic.csv", outputs)

    trade_cal = client.get_trade_cal(start_date=args.start, end_date=args.end)
    save_and_report(client, trade_cal, raw_dir / "trade_cal.csv", outputs)

    if args.universe == "hs300":
        index_weight = client.get_index_weight(
            index_code="000300.SH",
            start_date=args.start,
            end_date=args.end,
        )
        save_and_report(
            client,
            index_weight,
            raw_dir / "hs300_index_weight.csv",
            outputs,
        )

        components = client.get_hs300_components(
            start_date=args.start,
            end_date=args.end,
        )
        save_and_report(
            client,
            components,
            raw_dir / "hs300_components.csv",
            outputs,
        )
        codes = limit_codes(components.get("ts_code", pd.Series(dtype=str)), args.max_stocks)
    else:
        codes = limit_codes(stock_basic.get("ts_code", pd.Series(dtype=str)), args.max_stocks)

    print(f"Selected {len(codes)} stock codes from universe={args.universe}.")

    if not args.skip_monthly:
        monthly = fetch_monthly_data(
            client=client,
            codes=codes,
            start_date=args.start,
            end_date=args.end,
            sleep_seconds=args.sleep,
        )
        save_and_report(client, monthly, raw_dir / "monthly.csv", outputs)

    if not args.skip_daily_basic:
        daily_basic = fetch_daily_basic_data(
            client=client,
            codes=codes,
            start_date=args.start,
            end_date=args.end,
            sleep_seconds=args.sleep,
        )
        save_and_report(client, daily_basic, raw_dir / "daily_basic.csv", outputs)

    print("Output summary:")
    for name, rows, path in outputs:
        print(f"- {name}: {rows} rows -> {path}")


if __name__ == "__main__":
    main()
