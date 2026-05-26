from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def _temporal_subset(length: int, max_samples: int | None) -> np.ndarray:
    if max_samples is None or length <= max_samples:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, num=max_samples, dtype=int)


def evaluate_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_samples: int = 150_000,
    n_chunks: int = 4,
    top_n: int = 15,
) -> dict:
    idx = _temporal_subset(len(X), max_samples)
    X_s = X[idx]
    y_s = y[idx]

    mi = mutual_info_classif(X_s, y_s, random_state=42)
    top_order = np.argsort(mi)[::-1][:top_n]
    top_features = [(feature_names[i], float(mi[i])) for i in top_order]

    chunk_scores = []
    for chunk in np.array_split(np.arange(len(X_s)), max(1, n_chunks)):
        if len(chunk) < 100 or len(np.unique(y_s[chunk])) < 2:
            continue
        chunk_scores.append(mutual_info_classif(X_s[chunk], y_s[chunk], random_state=42))

    stable_features: list[str] = []
    if chunk_scores:
        chunk_matrix = np.vstack(chunk_scores)
        positive_rate = (chunk_matrix > 0.001).mean(axis=0)
        stable_idx = np.where(positive_rate >= 0.60)[0]
        stable_order = stable_idx[np.argsort(mi[stable_idx])[::-1][:top_n]]
        stable_features = [feature_names[i] for i in stable_order]

    return {
        "mean_mi": float(np.mean(mi)),
        "max_mi": float(np.max(mi)),
        "top_features": top_features,
        "stable_features": stable_features,
        "mi_vector": mi.tolist(),
    }
