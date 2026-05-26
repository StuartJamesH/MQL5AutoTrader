from __future__ import annotations

import numpy as np
import pandas as pd

from Learn.preprocess import preprocess_ohlcv

from .data import get_feature_builder


def build_research_features(
    df: pd.DataFrame,
    targets: pd.Series | np.ndarray,
    symbol: str,
    feature_builder: str = "auto",
    regime_params: dict | None = None,
    include_mtf: bool = False,
    max_rows: int | None = None,
    onehot_prefixes: tuple[str, ...] = ("OH_",),
    price_prefixes: tuple[str, ...] = ("PR_",),
):
    df_feat = df.copy().reset_index(drop=True)
    target_values = np.asarray(targets, dtype=int)

    if max_rows is not None and len(df_feat) > int(max_rows):
        df_feat = df_feat.tail(int(max_rows)).reset_index(drop=True)
        target_values = target_values[-len(df_feat):]

    df_feat["target"] = target_values

    feature_fn = get_feature_builder(symbol, feature_builder=feature_builder)
    try:
        df_feat = feature_fn(df_feat, include_mtf=include_mtf, regime_params=regime_params)
    except TypeError:
        df_feat = feature_fn(df_feat, regime_params=regime_params)

    X, y, _, feature_names, _, proc_df = preprocess_ohlcv(
        df_feat,
        target_col="target",
        onehot_prefixes=list(onehot_prefixes),
        price_prefixes=list(price_prefixes),
        return_df=True,
    )

    proc_df = proc_df.reset_index(drop=True)
    proc_df["target"] = np.asarray(y, dtype=int)
    return np.asarray(X), np.asarray(y, dtype=int), list(feature_names), proc_df
