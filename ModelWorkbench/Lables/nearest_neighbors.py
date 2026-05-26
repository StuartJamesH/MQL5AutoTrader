from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def evaluate_neighbor_consistency(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 15,
    train_frac: float = 0.70,
    max_train_rows: int = 75_000,
    max_eval_rows: int = 15_000,
) -> dict:
    split = int(len(X) * train_frac)
    if split < n_neighbors + 10 or len(X) - split < 10:
        raise ValueError("Not enough rows to run nearest-neighbor consistency analysis.")

    X_train = X[:split]
    y_train = y[:split]
    X_eval = X[split:]
    y_eval = y[split:]

    if max_train_rows and len(X_train) > max_train_rows:
        X_train = X_train[-max_train_rows:]
        y_train = y_train[-max_train_rows:]
    if max_eval_rows and len(X_eval) > max_eval_rows:
        X_eval = X_eval[-max_eval_rows:]
        y_eval = y_eval[-max_eval_rows:]

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_train)
    _, indices = knn.kneighbors(X_eval)
    neighbor_labels = y_train[indices]

    modes = []
    agreements = []
    purities = []
    entropies = []

    class_count = max(len(np.unique(y_train)), 1)
    entropy_norm = np.log(class_count) if class_count > 1 else 1.0

    for label, labels in zip(y_eval, neighbor_labels):
        counts = np.bincount(labels, minlength=3).astype(float)
        probs = counts / counts.sum()
        modes.append(int(np.argmax(counts)))
        agreements.append(float(probs[int(label)]))
        purities.append(float(probs.max()))

        nz = probs[probs > 0]
        entropy = -float(np.sum(nz * np.log(nz))) / entropy_norm if entropy_norm > 0 else 0.0
        entropies.append(entropy)

    return {
        "neighbor_agreement": float(np.mean(np.array(modes) == y_eval)),
        "neighbor_label_probability": float(np.mean(agreements)),
        "local_entropy": float(np.mean(entropies)),
        "cluster_purity": float(np.mean(purities)),
    }
