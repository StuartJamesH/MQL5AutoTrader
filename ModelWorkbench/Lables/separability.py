from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def _temporal_subset(length: int, max_samples: int | None) -> np.ndarray:
    if max_samples is None or length <= max_samples:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, num=max_samples, dtype=int)


def _pooled_distance(X: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.sum() < 10 or mask_b.sum() < 10:
        return float("nan")
    mu_a = X[mask_a].mean(axis=0)
    mu_b = X[mask_b].mean(axis=0)
    std_a = X[mask_a].std(axis=0) + 1e-9
    std_b = X[mask_b].std(axis=0) + 1e-9
    pooled = 0.5 * (std_a + std_b)
    return float(np.mean(np.abs(mu_a - mu_b) / pooled))


def evaluate_class_separability(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 25_000,
) -> dict:
    idx = _temporal_subset(len(X), max_samples)
    X_s = X[idx]
    y_s = y[idx]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_s)
    projection_frame = pd.DataFrame(
        {
            "component_1": coords[:, 0],
            "component_2": coords[:, 1],
            "target": y_s,
            "method": "PCA",
        }
    )

    sil = float("nan")
    if len(np.unique(y_s)) > 1 and len(X_s) > len(np.unique(y_s)):
        sil = float(silhouette_score(X_s, y_s))

    fisher_sell_flat = _pooled_distance(X_s, y_s == 0, y_s == 1)
    fisher_buy_flat = _pooled_distance(X_s, y_s == 2, y_s == 1)
    fisher_ratio = float(np.nanmean([fisher_sell_flat, fisher_buy_flat]))

    class_distance = float(np.nanmean([fisher_sell_flat, fisher_buy_flat]))

    return {
        "silhouette_score": sil,
        "fisher_ratio": fisher_ratio,
        "class_distance": class_distance,
        "projection_frame": projection_frame,
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
