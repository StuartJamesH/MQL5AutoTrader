from __future__ import annotations

import numpy as np
import pandas as pd

from Learn.labels import causal_market_regime, causal_triple_barrier_hilow_trend_labeler


def generate_label_events(
    df: pd.DataFrame,
    regime_params: dict,
    label_params: dict,
) -> pd.DataFrame:
    params = dict(label_params)
    params["regime_params"] = dict(regime_params)
    events = causal_triple_barrier_hilow_trend_labeler(df.copy(), **params)
    if events.empty:
        return pd.DataFrame(columns=["side", "z", "regime", "tp", "sl", "t_end", "label"])
    return events.sort_index()


def events_to_bar_labels(
    df: pd.DataFrame,
    events: pd.DataFrame,
    rollover_window: tuple[str, str] = ("21:30", "22:00"),
) -> pd.Series:
    target = pd.Series(1, index=df.index, dtype=int)

    if events is not None and not events.empty:
        winning = events[events["label"] == 1]
        buy_idx = winning[winning["side"] == 1].index
        sell_idx = winning[winning["side"] == -1].index
        target.loc[target.index.isin(buy_idx)] = 2
        target.loc[target.index.isin(sell_idx)] = 0

    time_col = pd.to_datetime(df["Time"])
    t_start = pd.to_datetime(rollover_window[0]).time()
    t_end = pd.to_datetime(rollover_window[1]).time()
    rollover_mask = (time_col.dt.time >= t_start) & (time_col.dt.time < t_end)
    target.loc[rollover_mask] = 1
    return target


def summarise_label_quality(
    df: pd.DataFrame,
    regime_params: dict,
    label_params: dict,
    events: pd.DataFrame | None = None,
    rollover_window: tuple[str, str] = ("21:30", "22:00"),
) -> dict:
    if events is None:
        events = generate_label_events(df, regime_params, label_params)

    total_bars = int(len(df))
    regime = causal_market_regime(df.copy(), **regime_params)
    bar_labels = events_to_bar_labels(df, events, rollover_window=rollover_window)

    result = {
        "total_bars": total_bars,
        "regime_coverage": float((regime != 0).mean()) if total_bars else 0.0,
        "uptrend_pct": float((regime == 1).mean()) if total_bars else 0.0,
        "range_pct": float((regime == 0).mean()) if total_bars else 0.0,
        "downtrend_pct": float((regime == -1).mean()) if total_bars else 0.0,
        "candidate_density": 0.0,
        "label_density": float((bar_labels != 1).mean()) if total_bars else 0.0,
        "tp_rate": 0.0,
        "sl_rate": 0.0,
        "timeout_rate": 0.0,
        "n_candidates": 0,
        "n_buy_labels": int((bar_labels == 2).sum()),
        "n_sell_labels": int((bar_labels == 0).sum()),
        "avg_bars_to_resolution": np.nan,
    }

    if events is None or events.empty:
        return result

    tp_events = events[events["label"] == 1]
    sl_events = events[events["label"] == -1]
    timeout_events = events[events["label"] == 0]
    n_events = len(events)

    result.update(
        {
            "candidate_density": n_events / total_bars if total_bars else 0.0,
            "label_density": len(tp_events) / total_bars if total_bars else 0.0,
            "tp_rate": len(tp_events) / n_events if n_events else 0.0,
            "sl_rate": len(sl_events) / n_events if n_events else 0.0,
            "timeout_rate": len(timeout_events) / n_events if n_events else 0.0,
            "n_candidates": int(n_events),
            "n_buy_labels": int(len(tp_events[tp_events["side"] == 1])),
            "n_sell_labels": int(len(tp_events[tp_events["side"] == -1])),
            "avg_bars_to_resolution": float((events["t_end"] - events.index.to_series()).mean()),
        }
    )
    return result
