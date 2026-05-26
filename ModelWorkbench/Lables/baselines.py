from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def _model_factories(model_names: list[str] | tuple[str, ...]):
    factories = {}

    if "logistic" in model_names:
        factories["logistic"] = lambda: LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=300,
            tol=1e-3,
            random_state=42,
        )

    if "random_forest" in model_names:
        factories["random_forest"] = lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

    if "lightgbm" in model_names:
        try:
            from lightgbm import LGBMClassifier

            factories["lightgbm"] = lambda: LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
            )
        except Exception:
            pass

    if "xgboost" in model_names:
        try:
            from xgboost import XGBClassifier

            factories["xgboost"] = lambda: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
        except Exception:
            pass

    return factories


def _safe_binary_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_binary)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_binary, scores))
    except Exception:
        return float("nan")


def calculate_trade_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray | None = None,
    classes: np.ndarray | list[int] | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    preds = np.asarray(preds, dtype=int)

    precision = precision_score(y_true, preds, labels=[0, 2], average=None, zero_division=0)
    recall = recall_score(y_true, preds, labels=[0, 2], average=None, zero_division=0)
    f1 = f1_score(y_true, preds, labels=[0, 2], average=None, zero_division=0)

    sell_precision, buy_precision = float(precision[0]), float(precision[1])
    sell_recall, buy_recall = float(recall[0]), float(recall[1])
    sell_f1, buy_f1 = float(f1[0]), float(f1[1])

    predicted_trade_mask = preds != 1
    true_trade_mask = y_true != 1

    predicted_trade_rate = float(predicted_trade_mask.mean())
    true_trade_rate = float(true_trade_mask.mean())

    trade_precision = float(
        precision_score(true_trade_mask.astype(int), predicted_trade_mask.astype(int), zero_division=0)
    )
    trade_recall = float(
        recall_score(true_trade_mask.astype(int), predicted_trade_mask.astype(int), zero_division=0)
    )

    directional_mask = predicted_trade_mask & true_trade_mask
    directional_accuracy = (
        float((preds[directional_mask] == y_true[directional_mask]).mean())
        if directional_mask.any()
        else float("nan")
    )

    true_buy_mask = y_true == 2
    true_sell_mask = y_true == 0
    buy_as_sell_rate = float((preds[true_buy_mask] == 0).mean()) if true_buy_mask.any() else float("nan")
    sell_as_buy_rate = float((preds[true_sell_mask] == 2).mean()) if true_sell_mask.any() else float("nan")

    false_trade_rate = float((predicted_trade_mask & ~true_trade_mask).mean())
    trade_ovr_auc = float("nan")
    buy_auc = float("nan")
    sell_auc = float("nan")

    if proba is not None and classes is not None:
        class_to_col = {int(cls): idx for idx, cls in enumerate(classes)}
        auc_values = []
        if 0 in class_to_col:
            sell_auc = _safe_binary_auc((y_true == 0).astype(int), proba[:, class_to_col[0]])
            auc_values.append(sell_auc)
        if 2 in class_to_col:
            buy_auc = _safe_binary_auc((y_true == 2).astype(int), proba[:, class_to_col[2]])
            auc_values.append(buy_auc)
        valid_aucs = [v for v in auc_values if not math.isnan(v)]
        trade_ovr_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    trade_macro_precision = float(np.mean([sell_precision, buy_precision]))
    trade_macro_recall = float(np.mean([sell_recall, buy_recall]))
    trade_macro_f1 = float(np.mean([sell_f1, buy_f1]))

    return {
        "sell_precision": sell_precision,
        "sell_recall": sell_recall,
        "sell_f1": sell_f1,
        "buy_precision": buy_precision,
        "buy_recall": buy_recall,
        "buy_f1": buy_f1,
        "trade_macro_precision": trade_macro_precision,
        "trade_macro_recall": trade_macro_recall,
        "trade_macro_f1": trade_macro_f1,
        "trade_precision": trade_precision,
        "trade_recall": trade_recall,
        "trade_ovr_auc": trade_ovr_auc,
        "buy_ovr_auc": buy_auc,
        "sell_ovr_auc": sell_auc,
        "predicted_trade_rate": predicted_trade_rate,
        "true_trade_rate": true_trade_rate,
        "false_trade_rate": false_trade_rate,
        "directional_accuracy_on_trades": directional_accuracy,
        "buy_as_sell_rate": buy_as_sell_rate,
        "sell_as_buy_rate": sell_as_buy_rate,
    }


def evaluate_baseline_models(
    X: np.ndarray,
    y: np.ndarray,
    model_names: list[str] | tuple[str, ...] = ("logistic", "random_forest", "lightgbm"),
    n_splits: int = 3,
    max_train_rows: int = 100_000,
    max_test_rows: int = 50_000,
) -> dict:
    if len(X) < max(200, n_splits + 10):
        raise ValueError("Not enough rows to run baseline models.")

    factories = _model_factories(model_names)
    if not factories:
        raise ValueError("No baseline models are available in the current environment.")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    metrics: dict[str, list[float]] = {}

    for train_idx, test_idx in splitter.split(X):
        if max_train_rows and len(train_idx) > max_train_rows:
            train_idx = train_idx[-max_train_rows:]
        if max_test_rows and len(test_idx) > max_test_rows:
            test_idx = test_idx[:max_test_rows]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        for model_name, factory in factories.items():
            model = factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            fold_metrics = calculate_trade_metrics(
                y_test,
                preds,
                proba=proba,
                classes=getattr(model, "classes_", None),
            )

            for key, value in fold_metrics.items():
                metrics.setdefault(f"{model_name}_{key}", []).append(float(value))

    summary = {
        key: float(np.nanmean(values)) if values else float("nan")
        for key, values in metrics.items()
    }

    trade_precision_values = [summary.get(f"{name}_trade_precision", float("nan")) for name in factories]
    trade_macro_f1_values = [summary.get(f"{name}_trade_macro_f1", float("nan")) for name in factories]
    directional_values = [summary.get(f"{name}_directional_accuracy_on_trades", float("nan")) for name in factories]
    trade_auc_values = [summary.get(f"{name}_trade_ovr_auc", float("nan")) for name in factories]

    summary["mean_trade_precision"] = (
        float(np.nanmean(trade_precision_values))
        if any(not math.isnan(v) for v in trade_precision_values)
        else float("nan")
    )
    summary["mean_trade_macro_f1"] = (
        float(np.nanmean(trade_macro_f1_values))
        if any(not math.isnan(v) for v in trade_macro_f1_values)
        else float("nan")
    )
    summary["mean_directional_accuracy_on_trades"] = (
        float(np.nanmean(directional_values))
        if any(not math.isnan(v) for v in directional_values)
        else float("nan")
    )
    summary["mean_trade_ovr_auc"] = (
        float(np.nanmean(trade_auc_values))
        if any(not math.isnan(v) for v in trade_auc_values)
        else float("nan")
    )
    summary["baseline_score"] = float(
        np.nanmean(
            [
                summary["mean_trade_precision"],
                summary["mean_trade_macro_f1"],
                summary["mean_directional_accuracy_on_trades"],
            ]
        )
    )

    # Legacy aliases kept so notebook code can evolve without breaking callers.
    summary["mean_auc"] = summary["mean_trade_ovr_auc"]
    summary["mean_macro_f1"] = summary["mean_trade_macro_f1"]
    summary["mean_balanced_accuracy"] = summary["mean_directional_accuracy_on_trades"]
    return summary
