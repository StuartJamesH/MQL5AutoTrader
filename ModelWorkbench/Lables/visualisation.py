from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


_LABEL_NAMES = {0: "SELL", 1: "FLAT", 2: "BUY"}
_LABEL_COLOURS = {0: "#d62728", 1: "#7f7f7f", 2: "#2ca02c"}


def plot_label_distribution(bar_labels, ax=None):
    ax = ax or plt.gca()
    series = pd.Series(bar_labels).map(_LABEL_NAMES).value_counts().reindex(["SELL", "FLAT", "BUY"], fill_value=0)
    colours = [_LABEL_COLOURS[0], _LABEL_COLOURS[1], _LABEL_COLOURS[2]]
    series.plot(kind="bar", color=colours, ax=ax)
    ax.set_title("Label Distribution")
    ax.set_ylabel("Bars")
    ax.tick_params(axis="x", rotation=0)
    return ax


def plot_signal_chart(df: pd.DataFrame, events: pd.DataFrame, bars: int = 1500, ax=None):
    ax = ax or plt.gca()
    window = df.tail(bars).copy()
    ax.plot(window["Time"], window["Close"], color="black", linewidth=1.0, label="Close")

    if events is not None and not events.empty:
        winning = events[(events["label"] == 1) & (events.index.isin(window.index))]
        buys = winning[winning["side"] == 1]
        sells = winning[winning["side"] == -1]

        if not buys.empty:
            ax.scatter(window.loc[buys.index, "Time"], window.loc[buys.index, "High"], s=16, color="#2ca02c", label="BUY")
        if not sells.empty:
            ax.scatter(window.loc[sells.index, "Time"], window.loc[sells.index, "Low"], s=16, color="#d62728", label="SELL")

    ax.set_title(f"Signal Chart (last {len(window):,} bars)")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    return ax


def plot_regime_performance(regime_breakdown: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    if regime_breakdown is None or regime_breakdown.empty:
        ax.set_title("Regime Performance")
        ax.text(0.5, 0.5, "No regime breakdown available", ha="center", va="center")
        ax.axis("off")
        return ax

    frame = regime_breakdown.copy()
    cols = [col for col in ["trade_precision", "trade_macro_f1", "directional_accuracy_on_trades"] if col in frame.columns]
    frame.plot(x="regime_name", y=cols, kind="bar", ax=ax)
    ax.set_title("Regime Trade Performance")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    return ax


def plot_feature_importance(mi_result: dict, top_n: int = 15, ax=None):
    ax = ax or plt.gca()
    top_features = mi_result.get("top_features", [])[:top_n]
    if not top_features:
        ax.set_title("Feature Importance")
        ax.text(0.5, 0.5, "No MI results available", ha="center", va="center")
        ax.axis("off")
        return ax

    labels = [name for name, _ in top_features][::-1]
    values = [score for _, score in top_features][::-1]
    ax.barh(labels, values, color="#1f77b4")
    ax.set_title("Top Mutual Information Features")
    ax.set_xlabel("Mutual Information")
    return ax


def plot_projection(projection_frame: pd.DataFrame, ax=None, method: str = "PCA"):
    ax = ax or plt.gca()
    if projection_frame is None or projection_frame.empty:
        ax.set_title(f"{method} Projection")
        ax.text(0.5, 0.5, "No projection data available", ha="center", va="center")
        ax.axis("off")
        return ax

    for label_value, label_name in _LABEL_NAMES.items():
        subset = projection_frame[projection_frame["target"] == label_value]
        if subset.empty:
            continue
        ax.scatter(
            subset["component_1"],
            subset["component_2"],
            s=8,
            alpha=0.5,
            label=label_name,
            color=_LABEL_COLOURS[label_value],
        )
    ax.set_title(f"{method} Projection")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best")
    return ax
