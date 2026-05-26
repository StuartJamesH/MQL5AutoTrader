from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_scale(value: float, low: float, high: float) -> float:
    if value is None or np.isnan(value):
        return float("nan")
    if high <= low:
        return float(value)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _nanmean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_scores(results: dict, weights: dict[str, float] | None = None) -> dict:
    baseline = results.get("baseline", {})
    mutual_information = results.get("mutual_information", {})
    separability = results.get("separability", {})
    regime = results.get("regime_analysis", {})

    support_score = _nanmean(
        [
            _nanmean(
                [
                    _clip_scale(mutual_information.get("mean_mi", np.nan), 0.00, 0.03),
                    _clip_scale(mutual_information.get("max_mi", np.nan), 0.01, 0.10),
                ]
            ),
            _nanmean(
                [
                    _clip_scale(separability.get("silhouette_score", np.nan), -0.05, 0.30),
                    _clip_scale(separability.get("fisher_ratio", np.nan), 0.25, 2.50),
                    _clip_scale(separability.get("class_distance", np.nan), 0.25, 2.50),
                ]
            ),
        ]
    )

    component_scores = {
        "trade_precision": _clip_scale(baseline.get("mean_trade_precision", np.nan), 0.10, 0.75),
        "trade_macro_f1": _clip_scale(baseline.get("mean_trade_macro_f1", np.nan), 0.05, 0.60),
        "directional_accuracy": _clip_scale(
            baseline.get("mean_directional_accuracy_on_trades", np.nan), 0.35, 0.85
        ),
        "regime_stability": _nanmean(
            [
                _clip_scale(regime.get("best_regime_trade_score", np.nan), 0.10, 0.75),
                _clip_scale(regime.get("worst_regime_trade_score", np.nan), 0.05, 0.65),
                _clip_scale(regime.get("regime_stability_score", np.nan), 0.10, 0.60),
            ]
        ),
        "support": support_score,
    }

    weights = weights or {
        "trade_precision": 0.40,
        "trade_macro_f1": 0.25,
        "directional_accuracy": 0.20,
        "regime_stability": 0.10,
        "support": 0.05,
    }

    learnability_score = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        component = component_scores.get(key, np.nan)
        if np.isnan(component):
            continue
        learnability_score += weight * component
        total_weight += weight

    if total_weight == 0:
        learnability_score = float("nan")
    else:
        learnability_score = float(learnability_score / total_weight)

    return {
        "baseline_score": _nanmean(
            [
                component_scores["trade_precision"],
                component_scores["trade_macro_f1"],
                component_scores["directional_accuracy"],
            ]
        ),
        "trade_precision_score": component_scores["trade_precision"],
        "trade_macro_f1_score": component_scores["trade_macro_f1"],
        "directional_accuracy_score": component_scores["directional_accuracy"],
        "support_score": component_scores["support"],
        "mi_score": _nanmean(
            [
                _clip_scale(mutual_information.get("mean_mi", np.nan), 0.00, 0.03),
                _clip_scale(mutual_information.get("max_mi", np.nan), 0.01, 0.10),
            ]
        ),
        "neighbor_score": float("nan"),
        "separability_score": _nanmean(
            [
                _clip_scale(separability.get("silhouette_score", np.nan), -0.05, 0.30),
                _clip_scale(separability.get("fisher_ratio", np.nan), 0.25, 2.50),
                _clip_scale(separability.get("class_distance", np.nan), 0.25, 2.50),
            ]
        ),
        "regime_score": component_scores["regime_stability"],
        "learnability_score": learnability_score,
    }


def rank_label_profiles(profile_results: dict[str, dict], weights: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for profile_name, result in profile_results.items():
        aggregate = result.get("aggregate")
        if aggregate is None:
            aggregate = aggregate_scores(result, weights=weights)
        baseline = result.get("baseline", {})
        label_summary = result.get("label_summary", {})
        rows.append(
            {
                "label_profile": profile_name,
                "learnability_score": aggregate.get("learnability_score", np.nan),
                "baseline_score": aggregate.get("baseline_score", np.nan),
                "trade_precision_score": aggregate.get("trade_precision_score", np.nan),
                "trade_macro_f1_score": aggregate.get("trade_macro_f1_score", np.nan),
                "directional_accuracy_score": aggregate.get("directional_accuracy_score", np.nan),
                "mi_score": aggregate.get("mi_score", np.nan),
                "support_score": aggregate.get("support_score", np.nan),
                "separability_score": aggregate.get("separability_score", np.nan),
                "regime_score": aggregate.get("regime_score", np.nan),
                "trade_precision": baseline.get("mean_trade_precision", np.nan),
                "trade_macro_f1": baseline.get("mean_trade_macro_f1", np.nan),
                "trade_ovr_auc": baseline.get("mean_trade_ovr_auc", np.nan),
                "directional_accuracy_on_trades": baseline.get("mean_directional_accuracy_on_trades", np.nan),
                "predicted_trade_rate": baseline.get("logistic_predicted_trade_rate", np.nan),
                "false_trade_rate": baseline.get("logistic_false_trade_rate", np.nan),
                "sell_precision": baseline.get("logistic_sell_precision", np.nan),
                "sell_recall": baseline.get("logistic_sell_recall", np.nan),
                "buy_precision": baseline.get("logistic_buy_precision", np.nan),
                "buy_recall": baseline.get("logistic_buy_recall", np.nan),
                "neighbor_consistency": result.get("nearest_neighbors", {}).get("neighbor_agreement", np.nan),
                "label_density": label_summary.get("label_density", np.nan),
                "tp_rate": label_summary.get("tp_rate", np.nan),
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values(
        ["learnability_score", "trade_precision", "trade_macro_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
