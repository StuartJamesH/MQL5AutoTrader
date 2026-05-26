from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Learn.features import (
    _add_features_EURUSD,
    _add_features_SpotCrude,
    _add_features_US2000,
    _add_features_US500,
    _add_features_XAUUSD,
    _add_features_light,
    add_all_features,
)

MODEL_WORKBENCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_WORKBENCH_DIR.parent
LABEL_PARAMS_FILE = MODEL_WORKBENCH_DIR / "params" / "label_params.json"

SYMBOL_DATASET_MAP = {
    "US500": REPO_ROOT / "data" / "US500_M1_520weeks.csv",
    "EURUSD": REPO_ROOT / "data" / "EURUSD_M1_520weeks.csv",
    "XAUUSD": REPO_ROOT / "data" / "XAUUSD_M1_520weeks.csv",
    "US2000": REPO_ROOT / "data" / "US2000_M1_520weeks.csv",
    "NAS100": REPO_ROOT / "data" / "NAS100_M1_520weeks.csv",
    "SpotCrude": REPO_ROOT / "data" / "SpotCrude_M1_520weeks.csv",
}

SYMBOL_FEATURES_MAP = {
    "US500": _add_features_US500,
    "EURUSD": _add_features_EURUSD,
    "XAUUSD": _add_features_XAUUSD,
    "US2000": _add_features_US2000,
    "NAS100": _add_features_US500,
    "SpotCrude": _add_features_SpotCrude,
}

_COLUMN_RENAMES = {
    "timestamp": "Time",
    "time": "Time",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def resolve_dataset_path(symbol: str, dataset_path: str | None = None) -> str:
    if dataset_path:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = (MODEL_WORKBENCH_DIR / path).resolve()
        return str(path)

    if symbol not in SYMBOL_DATASET_MAP:
        raise ValueError(f"Unsupported symbol {symbol!r}. Choose from {sorted(SYMBOL_DATASET_MAP)}")
    return str(SYMBOL_DATASET_MAP[symbol])


def load_ohlcv_dataset(path_or_symbol: str, n_rows: int | None = None) -> pd.DataFrame:
    path = Path(path_or_symbol)
    if not path.exists() and path_or_symbol in SYMBOL_DATASET_MAP:
        path = SYMBOL_DATASET_MAP[path_or_symbol]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    rename_map = {col: _COLUMN_RENAMES[col.lower()] for col in df.columns if col.lower() in _COLUMN_RENAMES}
    df = df.rename(columns=rename_map)

    required = ["Time", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns {missing}. Found: {list(df.columns)}")

    df = df[required].copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").drop_duplicates(subset="Time").reset_index(drop=True)

    if n_rows is not None:
        df = df.tail(int(n_rows)).reset_index(drop=True)

    return df


def load_label_profiles(profile_names: list[str]) -> dict[str, dict]:
    data = json.loads(LABEL_PARAMS_FILE.read_text(encoding="utf-8"))
    profiles: dict[str, dict] = {}
    for profile_name in profile_names:
        if profile_name not in data:
            available = [key for key in data if not key.startswith("_")]
            raise ValueError(f"Label profile {profile_name!r} not found. Available: {available}")
        entry = data[profile_name]
        profiles[profile_name] = {
            "profile_name": profile_name,
            "comment": entry.get("_comment", ""),
            "regime_params": dict(entry["regime_params"]),
            "label_params": dict(entry["label_params"]),
        }
    return profiles


def get_feature_builder(symbol: str, feature_builder: str = "auto"):
    if feature_builder == "auto":
        if symbol not in SYMBOL_FEATURES_MAP:
            raise ValueError(f"Unsupported symbol {symbol!r}. Choose from {sorted(SYMBOL_FEATURES_MAP)}")
        return SYMBOL_FEATURES_MAP[symbol]
    if feature_builder == "add_all_features":
        return add_all_features
    if feature_builder == "_add_features_light":
        return _add_features_light
    if feature_builder.startswith("_add_features_"):
        for fn in list(SYMBOL_FEATURES_MAP.values()) + [_add_features_light]:
            if getattr(fn, "__name__", "") == feature_builder:
                return fn
    raise ValueError(
        f"Unsupported feature builder {feature_builder!r}. Use 'auto', 'add_all_features', '_add_features_light', or a known symbol builder."
    )
