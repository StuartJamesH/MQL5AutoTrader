from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .baselines import calculate_trade_metrics

_REGIME_NAMES = {-1: "Downtrend", 0: "Range", 1: "Uptrend"}


def evaluate_regime_stability(
    X: np.ndarray,
    y: np.ndarray,
    regimes,
    train_frac: float = 0.70,
    max_train_rows: int = 100_000,
    max_test_rows: int = 50_000,
) -> dict:
    regimes = np.asarray(regimes, dtype=int)
    if len(regimes) != len(y):
        raise ValueError("Regime array must align with the feature rows.")

    split = int(len(X) * train_frac)
    if split < 100 or len(X) - split < 100:
        raise ValueError("Not enough rows to run regime stability analysis.")

    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    regimes_test = regimes[split:]
    regimes_all = regimes

    if max_train_rows and len(X_train) > max_train_rows:
        X_train = X_train[-max_train_rows:]
        y_train = y_train[-max_train_rows:]
    if max_test_rows and len(X_test) > max_test_rows:
        X_test = X_test[-max_test_rows:]
        y_test = y_test[-max_test_rows:]
        regimes_test = regimes_test[-max_test_rows:]

    model = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=300,
        tol=1e-3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    rows = []
    for regime_value in sorted(np.unique(regimes_test)):
        mask = regimes_test == regime_value
        if mask.sum() < 50:
            continue

        trade_metrics = calculate_trade_metrics(
            y_test[mask],
            preds[mask],
            proba=proba[mask],
            classes=getattr(model, "classes_", None),
        )
        trade_regime_score = float(
            np.nanmean(
                [
                    trade_metrics["trade_precision"],
                    trade_metrics["trade_macro_f1"],
                    trade_metrics["directional_accuracy_on_trades"],
                    trade_metrics["trade_ovr_auc"],
                ]
            )
        )

        rows.append(
            {
                "regime": int(regime_value),
                "regime_name": _REGIME_NAMES.get(int(regime_value), str(regime_value)),
                "sample_count": int(mask.sum()),
                "label_density": float((regimes_all == regime_value).mean()),
                "trade_precision": trade_metrics["trade_precision"],
                "trade_recall": trade_metrics["trade_recall"],
                "trade_macro_f1": trade_metrics["trade_macro_f1"],
                "directional_accuracy_on_trades": trade_metrics["directional_accuracy_on_trades"],
                "trade_ovr_auc": trade_metrics["trade_ovr_auc"],
                "predicted_trade_rate": trade_metrics["predicted_trade_rate"],
                "false_trade_rate": trade_metrics["false_trade_rate"],
                "trade_regime_score": trade_regime_score,
            }
        )

    breakdown = pd.DataFrame(rows)
    if breakdown.empty:
        return {
            "best_regime_trade_score": float("nan"),
            "worst_regime_trade_score": float("nan"),
            "regime_stability_score": float("nan"),
            "regime_breakdown": breakdown,
        }

    score_series = breakdown["trade_regime_score"]
    best_score = float(score_series.max())
    worst_score = float(score_series.min())
    spread = best_score - worst_score
    stability_score = float(max(0.0, score_series.mean() * (1.0 - min(1.0, spread))))

    return {
        "best_regime_trade_score": best_score,
        "worst_regime_trade_score": worst_score,
        "regime_stability_score": stability_score,
        "regime_breakdown": breakdown.sort_values("trade_regime_score", ascending=False).reset_index(drop=True),
        # Legacy aliases
        "best_regime_auc": float(breakdown["trade_ovr_auc"].max()),
        "worst_regime_auc": float(breakdown["trade_ovr_auc"].min()),
    }
