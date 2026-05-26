from __future__ import annotations

from pathlib import Path

from .data import resolve_dataset_path

DEFAULT_SCORE_WEIGHTS = {
    "trade_precision": 0.40,
    "trade_macro_f1": 0.25,
    "directional_accuracy": 0.20,
    "regime_stability": 0.10,
    "support": 0.05,
}


def build_experiment_config(
    symbol: str = "US500",
    label_profiles: list[str] | None = None,
    dataset_path: str | None = None,
    feature_builder: str = "auto",
    n_rows: int | None = 250_000,
    include_mtf: bool = False,
    temporal_folds: int = 3,
    baseline_models: tuple[str, ...] = ("logistic", "random_forest", "lightgbm"),
    baseline_train_rows: int = 100_000,
    baseline_test_rows: int = 50_000,
    mi_rows: int = 150_000,
    projection_rows: int = 25_000,
    neighbor_train_rows: int = 75_000,
    neighbor_eval_rows: int = 15_000,
    neighbor_k: int = 15,
    regime_test_rows: int = 50_000,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    test_frac: float = 0.15,
    rollover_window: tuple[str, str] = ("21:30", "22:00"),
    save_outputs: bool = False,
    output_dir: str = "Lables\\outputs",
    score_weights: dict[str, float] | None = None,
) -> dict:
    if label_profiles is None:
        label_profiles = [f"{symbol}_1m_dev"]

    cfg = {
        "symbol": symbol,
        "label_profiles": list(label_profiles),
        "dataset_path": resolve_dataset_path(symbol, dataset_path),
        "feature_builder": feature_builder,
        "n_rows": n_rows,
        "include_mtf": include_mtf,
        "temporal_folds": temporal_folds,
        "baseline_models": list(baseline_models),
        "baseline_train_rows": baseline_train_rows,
        "baseline_test_rows": baseline_test_rows,
        "mi_rows": mi_rows,
        "projection_rows": projection_rows,
        "neighbor_train_rows": neighbor_train_rows,
        "neighbor_eval_rows": neighbor_eval_rows,
        "neighbor_k": neighbor_k,
        "regime_test_rows": regime_test_rows,
        "train_frac": train_frac,
        "validation_frac": validation_frac,
        "test_frac": test_frac,
        "rollover_window": rollover_window,
        "save_outputs": save_outputs,
        "output_dir": str(Path(output_dir)),
        "score_weights": dict(score_weights or DEFAULT_SCORE_WEIGHTS),
    }
    return cfg
