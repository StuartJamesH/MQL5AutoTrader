"""Reporting helpers for production trade analysis from MetaTrader 5.

The main entrypoint is :func:`fetch_trade_report`, which pulls historical deal
data from a local MT5 terminal for a caller-specified date range and returns:

1. A normalized deal-level DataFrame
2. A brief per-symbol performance summary DataFrame
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - package absent in non-MT5 environments
    mt5 = None


_TRADE_COLUMNS = [
    "ticket",
    "order",
    "position_id",
    "time",
    "time_msc",
    "symbol",
    "side",
    "deal_type",
    "entry_type",
    "reason",
    "volume",
    "price",
    "profit",
    "commission",
    "swap",
    "fee",
    "net_pnl",
    "magic",
    "comment",
    "external_id",
]

_SUMMARY_COLUMNS = [
    "symbol",
    "trade_count",
    "first_trade_time",
    "last_trade_time",
    "volume_lots",
    "gross_profit",
    "gross_loss",
    "net_pnl",
    "avg_net_pnl",
    "median_net_pnl",
    "win_rate",
    "avg_win",
    "avg_loss",
    "total_commission",
    "total_swap",
    "total_fee",
]


def _require_mt5() -> Any:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not available. Install with `pip install MetaTrader5`.")

    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    return mt5


def _coerce_datetime(value: Any, *, name: str) -> datetime:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"{name} must be a valid date/time, got {value!r}")
    return ts.to_pydatetime()


def _has_explicit_time(value: Any) -> bool:
    if isinstance(value, datetime):
        return any([value.hour, value.minute, value.second, value.microsecond])
    if isinstance(value, date):
        return False
    if isinstance(value, str):
        text = value.strip()
        return any(token in text for token in ["T", " ", ":"])
    return False


def _normalize_date_range(start_date: Any, end_date: Any) -> tuple[datetime, datetime]:
    start_dt = _coerce_datetime(start_date, name="start_date")
    end_dt = _coerce_datetime(end_date, name="end_date")
    if not _has_explicit_time(end_date):
        end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_dt}) must be greater than or equal to start_date ({start_dt})")
    return start_dt, end_dt


def _label_from_constant(value: Any, prefix: str) -> str:
    if mt5 is None:
        return str(value)

    for attr in dir(mt5):
        if attr.startswith(prefix) and getattr(mt5, attr) == value:
            return attr.removeprefix(prefix)
    return str(value)


def _side_from_deal_type(deal_type: Any) -> str | None:
    if mt5 is None:
        return None
    if deal_type == mt5.DEAL_TYPE_BUY:
        return "buy"
    if deal_type == mt5.DEAL_TYPE_SELL:
        return "sell"
    return None


def _empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_TRADE_COLUMNS)


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_SUMMARY_COLUMNS)


def fetch_trade_history(
    start_date: Any,
    end_date: Any,
    *,
    group: str | None = None,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return normalized MT5 deal history for the requested date range.

    Parameters
    ----------
    start_date, end_date:
        Any datetime-like values accepted by ``pandas.to_datetime``.
    group:
        Optional MT5 group filter passed through to ``history_deals_get``.
    symbols:
        Optional iterable of symbol names used to filter the returned rows
        after retrieval.
    """

    mt5_mod = _require_mt5()
    start_dt, end_dt = _normalize_date_range(start_date, end_date)

    if group is None:
        deals = mt5_mod.history_deals_get(start_dt, end_dt)
    else:
        deals = mt5_mod.history_deals_get(start_dt, end_dt, group=group)

    if deals is None:
        raise RuntimeError(f"mt5.history_deals_get() failed: {mt5_mod.last_error()}")

    rows: list[dict[str, Any]] = []
    for deal in deals:
        profit = float(getattr(deal, "profit", 0.0))
        commission = float(getattr(deal, "commission", 0.0))
        swap = float(getattr(deal, "swap", 0.0))
        fee = float(getattr(deal, "fee", 0.0))
        row = {
            "ticket": getattr(deal, "ticket", None),
            "order": getattr(deal, "order", None),
            "position_id": getattr(deal, "position_id", None),
            "time": pd.to_datetime(getattr(deal, "time", None), unit="s", utc=True),
            "time_msc": pd.to_datetime(getattr(deal, "time_msc", None), unit="ms", utc=True),
            "symbol": getattr(deal, "symbol", None),
            "side": _side_from_deal_type(getattr(deal, "type", None)),
            "deal_type": _label_from_constant(getattr(deal, "type", None), "DEAL_TYPE_"),
            "entry_type": _label_from_constant(getattr(deal, "entry", None), "DEAL_ENTRY_"),
            "reason": _label_from_constant(getattr(deal, "reason", None), "DEAL_REASON_"),
            "volume": float(getattr(deal, "volume", 0.0)),
            "price": float(getattr(deal, "price", 0.0)),
            "profit": profit,
            "commission": commission,
            "swap": swap,
            "fee": fee,
            "net_pnl": profit + commission + swap + fee,
            "magic": getattr(deal, "magic", None),
            "comment": getattr(deal, "comment", ""),
            "external_id": getattr(deal, "external_id", ""),
        }
        rows.append(row)

    if not rows:
        return _empty_trade_frame()

    trades = pd.DataFrame(rows, columns=_TRADE_COLUMNS).sort_values("time").reset_index(drop=True)

    if symbols is not None:
        if isinstance(symbols, str):
            symbol_set = {symbols}
        else:
            symbol_set = {str(symbol) for symbol in symbols}
        trades = trades[trades["symbol"].astype(str).isin(symbol_set)].reset_index(drop=True)

    return trades


def summarize_trade_history_by_symbol(
    trades: pd.DataFrame,
    *,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Return brief per-symbol summary stats from a normalized deal DataFrame."""

    if trades.empty:
        return _empty_summary_frame()

    working = trades.copy()
    if closed_only:
        working = working[working["entry_type"].isin(["OUT", "OUT_BY", "INOUT"])].copy()

    if working.empty:
        return _empty_summary_frame()

    working["is_win"] = working["net_pnl"] > 0

    grouped = working.groupby("symbol", dropna=False)
    summary = grouped.apply(
        lambda frame: pd.Series(
            {
                "trade_count": int(len(frame)),
                "first_trade_time": frame["time"].min(),
                "last_trade_time": frame["time"].max(),
                "volume_lots": float(frame["volume"].sum()),
                "gross_profit": float(frame.loc[frame["net_pnl"] > 0, "net_pnl"].sum()),
                "gross_loss": float(frame.loc[frame["net_pnl"] < 0, "net_pnl"].sum()),
                "net_pnl": float(frame["net_pnl"].sum()),
                "avg_net_pnl": float(frame["net_pnl"].mean()),
                "median_net_pnl": float(frame["net_pnl"].median()),
                "win_rate": float(frame["is_win"].mean()),
                "avg_win": float(frame.loc[frame["net_pnl"] > 0, "net_pnl"].mean()),
                "avg_loss": float(frame.loc[frame["net_pnl"] < 0, "net_pnl"].mean()),
                "total_commission": float(frame["commission"].sum()),
                "total_swap": float(frame["swap"].sum()),
                "total_fee": float(frame["fee"].sum()),
            }
        )
    )

    summary = summary.reset_index().rename(columns={"index": "symbol"})
    summary["win_rate"] = summary["win_rate"].fillna(0.0)
    summary = summary.sort_values(["net_pnl", "trade_count"], ascending=[False, False]).reset_index(drop=True)
    return summary[_SUMMARY_COLUMNS]


def fetch_trade_report(
    start_date: Any,
    end_date: Any,
    *,
    group: str | None = None,
    symbols: Iterable[str] | None = None,
    closed_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(trades, summary)`` for the requested MT5 date range.

    This is the notebook-facing convenience function for production reporting.
    """

    trades = fetch_trade_history(
        start_date=start_date,
        end_date=end_date,
        group=group,
        symbols=symbols,
    )
    summary = summarize_trade_history_by_symbol(trades, closed_only=closed_only)
    return trades, summary
