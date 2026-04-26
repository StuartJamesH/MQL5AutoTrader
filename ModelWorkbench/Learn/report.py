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


def build_position_pairs(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair IN and OUT deals into complete round-trip trades.

    Returns one row per closed position with columns:
    position_id, symbol, side, entry_time, exit_time,
    entry_price, exit_price, volume, gross_pnl,
    commission, swap, fee, net_pnl, duration_mins, result
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "position_id", "symbol", "side", "entry_time", "exit_time",
                "entry_price", "exit_price", "volume", "gross_pnl",
                "commission", "swap", "fee", "net_pnl", "duration_mins", "result",
            ]
        )

    entries = (
        trades[trades["entry_type"] == "IN"]
        .rename(columns={"time": "entry_time", "price": "entry_price"})
        [["position_id", "symbol", "side", "entry_time", "entry_price", "volume",
          "commission", "swap", "fee"]]
        .copy()
    )

    exits = (
        trades[trades["entry_type"].isin(["OUT", "OUT_BY", "INOUT"])]
        .rename(columns={"time": "exit_time", "price": "exit_price", "profit": "gross_pnl"})
        [["position_id", "exit_time", "exit_price", "gross_pnl",
          "commission", "swap", "fee"]]
        .copy()
    )

    merged = entries.merge(exits, on="position_id", suffixes=("_in", "_out"))

    merged["commission"] = merged["commission_in"] + merged["commission_out"]
    merged["swap"] = merged["swap_in"] + merged["swap_out"]
    merged["fee"] = merged["fee_in"] + merged["fee_out"]
    merged["net_pnl"] = merged["gross_pnl"] + merged["commission"] + merged["swap"] + merged["fee"]
    merged["duration_mins"] = (
        (merged["exit_time"] - merged["entry_time"]).dt.total_seconds() / 60
    )
    merged["result"] = merged["net_pnl"].apply(
        lambda v: "win" if v > 0 else ("loss" if v < 0 else "breakeven")
    )

    cols = [
        "position_id", "symbol", "side", "entry_time", "exit_time",
        "entry_price", "exit_price", "volume", "gross_pnl",
        "commission", "swap", "fee", "net_pnl", "duration_mins", "result",
    ]
    return merged[cols].sort_values("entry_time").reset_index(drop=True)


def compute_trade_quality_metrics(pairs: pd.DataFrame) -> dict:
    """Return profit factor, expectancy, win rate, payoff ratio, and commission stats."""
    if pairs.empty:
        return {
            "trade_count": 0, "win_count": 0, "loss_count": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "payoff_ratio": 0.0, "total_commission": 0.0, "commission_per_trade": 0.0,
            "total_net_pnl": 0.0,
        }

    wins = pairs[pairs["result"] == "win"]["net_pnl"]
    losses = pairs[pairs["result"] == "loss"]["net_pnl"]

    total_win = wins.sum()
    total_loss = abs(losses.sum())
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    win_rate = len(wins) / len(pairs) if len(pairs) > 0 else 0.0
    profit_factor = total_win / total_loss if total_loss != 0 else float("inf")
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")
    total_commission = float(pairs["commission"].sum() + pairs["swap"].sum() + pairs["fee"].sum())

    return {
        "trade_count": int(len(pairs)),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": float(pairs["net_pnl"].mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "total_commission": total_commission,
        "commission_per_trade": total_commission / len(pairs),
        "total_net_pnl": float(pairs["net_pnl"].sum()),
    }


def equity_curve_and_drawdown(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute equity curve and drawdown from position pairs sorted by exit_time.

    Returns (curve_df, metrics_dict).
    curve_df columns: exit_time, net_pnl, equity, drawdown, drawdown_pct
    metrics_dict keys: max_drawdown, max_drawdown_pct, final_equity, longest_drawdown_trades, calmar_ratio
    """
    if pairs.empty:
        return pd.DataFrame(
            columns=["exit_time", "net_pnl", "equity", "drawdown", "drawdown_pct"]
        ), {
            "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "final_equity": 0.0, "longest_drawdown_trades": 0, "calmar_ratio": 0.0,
        }

    df = pairs[["exit_time", "net_pnl"]].sort_values("exit_time").copy()
    df["equity"] = df["net_pnl"].cumsum()
    df["running_peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] - df["running_peak"]
    df["drawdown_pct"] = df.apply(
        lambda r: (r["drawdown"] / r["running_peak"] * 100) if r["running_peak"] != 0 else 0.0,
        axis=1,
    )

    max_drawdown = float(df["drawdown"].min())
    max_drawdown_pct = float(df["drawdown_pct"].min())
    final_equity = float(df["equity"].iloc[-1])

    # Longest consecutive drawdown streak
    in_dd = (df["drawdown"] < 0).astype(int)
    streak = longest = current = 0
    for v in in_dd:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    calmar_ratio = final_equity / abs(max_drawdown) if max_drawdown != 0 else 0.0

    curve_df = df[["exit_time", "net_pnl", "equity", "drawdown", "drawdown_pct"]].reset_index(drop=True)
    metrics = {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "final_equity": final_equity,
        "longest_drawdown_trades": longest,
        "calmar_ratio": calmar_ratio,
    }
    return curve_df, metrics


def compute_rolling_metrics(pairs: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling win rate and average PnL over a sliding trade window."""
    if pairs.empty:
        return pd.DataFrame(columns=["exit_time", "net_pnl", "rolling_win_rate", "rolling_avg_pnl"])

    df = pairs[["exit_time", "net_pnl"]].sort_values("exit_time").copy()
    df["is_win"] = (pairs.sort_values("exit_time")["result"] == "win").values
    df["rolling_win_rate"] = df["is_win"].rolling(window, min_periods=1).mean()
    df["rolling_avg_pnl"] = df["net_pnl"].rolling(window, min_periods=1).mean()
    return df[["exit_time", "net_pnl", "rolling_win_rate", "rolling_avg_pnl"]].reset_index(drop=True)


def compute_hourly_performance(pairs: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Aggregate performance by hour of day (entry time, all 24 hours present)."""
    all_hours = pd.DataFrame({"hour": range(24)})
    if pairs.empty:
        return all_hours.assign(trade_count=0, win_rate=0.0, avg_pnl=0.0, total_pnl=0.0)

    df = pairs.copy()
    entry = pd.to_datetime(df["entry_time"])
    if entry.dt.tz is None:
        entry = entry.dt.tz_localize("UTC")
    if tz != "UTC":
        entry = entry.dt.tz_convert(tz)
    df["hour"] = entry.dt.hour
    df["is_win"] = df["result"] == "win"

    agg = df.groupby("hour").agg(
        trade_count=("net_pnl", "count"),
        win_rate=("is_win", "mean"),
        avg_pnl=("net_pnl", "mean"),
        total_pnl=("net_pnl", "sum"),
    ).reset_index()

    result = all_hours.merge(agg, on="hour", how="left").fillna(
        {"trade_count": 0, "win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}
    )
    result["trade_count"] = result["trade_count"].astype(int)
    return result


def compute_weekday_performance(pairs: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Aggregate performance by day of week (entry time)."""
    _day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    all_days = pd.DataFrame({"weekday_num": range(7), "weekday": _day_names})

    if pairs.empty:
        return all_days.assign(trade_count=0, win_rate=0.0, avg_pnl=0.0, total_pnl=0.0)

    df = pairs.copy()
    entry = pd.to_datetime(df["entry_time"])
    if entry.dt.tz is None:
        entry = entry.dt.tz_localize("UTC")
    if tz != "UTC":
        entry = entry.dt.tz_convert(tz)
    df["weekday_num"] = entry.dt.dayofweek
    df["is_win"] = df["result"] == "win"

    agg = df.groupby("weekday_num").agg(
        trade_count=("net_pnl", "count"),
        win_rate=("is_win", "mean"),
        avg_pnl=("net_pnl", "mean"),
        total_pnl=("net_pnl", "sum"),
    ).reset_index()

    result = all_days.merge(agg, on="weekday_num", how="left").fillna(
        {"trade_count": 0, "win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}
    )
    result["trade_count"] = result["trade_count"].astype(int)
    return result.sort_values("weekday_num").reset_index(drop=True)


def compute_side_performance(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance split by trade side (buy/sell)."""
    if pairs.empty:
        return pd.DataFrame(
            columns=["side", "trade_count", "win_rate", "avg_pnl", "total_pnl", "avg_duration_mins"]
        )

    df = pairs.copy()
    df["is_win"] = df["result"] == "win"

    agg = df.groupby("side").agg(
        trade_count=("net_pnl", "count"),
        win_rate=("is_win", "mean"),
        avg_pnl=("net_pnl", "mean"),
        total_pnl=("net_pnl", "sum"),
        avg_duration_mins=("duration_mins", "mean"),
    ).reset_index()
    return agg


def compute_mae_mfe(
    pairs: pd.DataFrame,
    ohlcv: pd.DataFrame,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Compute Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
    for each completed trade using 1-minute OHLCV bars.

    MAE = worst price move against the trade direction during the trade's lifetime (always <= 0 in price pts)
    MFE = best price move in the trade direction during the trade's lifetime (always >= 0 in price pts)

    For buys:  MAE = Low.min() - entry_price,  MFE = High.max() - entry_price
    For sells: MAE = -(High.max() - entry_price), MFE = -(Low.min() - entry_price)

    Returns DataFrame with columns:
    position_id, side, entry_time, exit_time, duration_mins, net_pnl, result,
    mae_pts, mfe_pts, entry_price, mfe_to_mae_ratio
    """
    import numpy as np

    empty_cols = [
        "position_id", "side", "entry_time", "exit_time", "duration_mins",
        "net_pnl", "result", "mae_pts", "mfe_pts", "entry_price", "mfe_to_mae_ratio",
    ]
    if pairs.empty or ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=empty_cols)

    working = pairs.copy()
    if symbol is not None:
        working = working[working["symbol"].str.startswith(symbol)].copy()
    if working.empty:
        return pd.DataFrame(columns=empty_cols)

    # Ensure ohlcv index is tz-aware UTC
    idx = ohlcv.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ohlcv_tz = ohlcv.copy()
    ohlcv_tz.index = idx

    rows = []
    for _, trade in working.iterrows():
        t0 = pd.Timestamp(trade["entry_time"])
        t1 = pd.Timestamp(trade["exit_time"])
        if t0.tz is None:
            t0 = t0.tz_localize("UTC")
        if t1.tz is None:
            t1 = t1.tz_localize("UTC")

        bars = ohlcv_tz.loc[t0:t1]
        if bars.empty:
            continue

        ep = float(trade["entry_price"])
        side = str(trade["side"]).lower() if trade["side"] else ""

        if side == "buy":
            mae_pts = float(bars["Low"].min()) - ep
            mfe_pts = float(bars["High"].max()) - ep
        else:
            mae_pts = -(float(bars["High"].max()) - ep)
            mfe_pts = -(float(bars["Low"].min()) - ep)

        mfe_to_mae = mfe_pts / abs(mae_pts) if mae_pts != 0 else np.nan

        rows.append({
            "position_id": trade["position_id"],
            "side": trade["side"],
            "entry_time": trade["entry_time"],
            "exit_time": trade["exit_time"],
            "duration_mins": trade["duration_mins"],
            "net_pnl": trade["net_pnl"],
            "result": trade["result"],
            "mae_pts": mae_pts,
            "mfe_pts": mfe_pts,
            "entry_price": ep,
            "mfe_to_mae_ratio": mfe_to_mae,
        })

    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows, columns=empty_cols).reset_index(drop=True)
