"""
label_quality_probe.py
======================
Fast label-quality and learnability analysis for causal_triple_barrier_hilow_trend_labeler.

Runs three tiers of analysis with no neural network training required:

  Tier 1  — Label statistics only       (~15-60s per profile)
  Tier 2  — + Feature separability      (~2-4 min per profile)
  Tier 3  — + Logistic regression probe (~4-8 min per profile)  [default]

Accepts one or more label profiles and prints a side-by-side comparison table.

Usage examples:
    python label_quality_probe.py --symbol US500 --label-profiles US500_1m_dev US500_1m_dev_r2
    python label_quality_probe.py --symbol EURUSD --label-profiles EURUSD_1m_dev EURUSD_1m_r13 --tier 1
    python label_quality_probe.py --symbol US500 --label-profiles US500_1m_dev_r3 --tier 3 --n-rows 500000
    python label_quality_probe.py --symbol US500 --label-profiles US500_1m_dev --save

Output:
    Console comparison table.
    With --save: JSON written to ModelWorkbench/params/label_quality_<symbol>_<timestamp>.json
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from Learn.labels import (
    causal_triple_barrier_hilow_trend_labeler,
    causal_market_regime,
)
from Learn.features import (
    _add_features_EURUSD,
    _add_features_US500,
    _add_features_US2000,
    _add_features_XAUUSD,
    _add_features_SpotCrude,
)
from Learn.preprocess import preprocess_ohlcv
from talib import ATR

# ─── Symbol maps ─────────────────────────────────────────────────────────────

SYMBOL_DATASET_MAP: dict[str, str] = {
    "US500":     "../data/US500_M1_520weeks.csv",
    "EURUSD":    "../data/EURUSD_M1_520weeks.csv",
    "XAUUSD":    "../data/XAUUSD_M1_520weeks.csv",
    "US2000":    "../data/US2000_M1_520weeks.csv",
    "NAS100":    "../data/NAS100_M1_520weeks.csv",
    "SpotCrude": "../data/SpotCrude_M1_520weeks.csv",
}

SYMBOL_FEATURES_MAP = {
    "US500":     _add_features_US500,
    "EURUSD":    _add_features_EURUSD,
    "XAUUSD":    _add_features_XAUUSD,
    "US2000":    _add_features_US2000,
    "NAS100":    _add_features_US500,
    "SpotCrude": _add_features_SpotCrude,
}

_PARAMS_DIR = Path(__file__).parent / "params"
_LABEL_PARAMS_FILE = _PARAMS_DIR / "label_params.json"

# ─── Param loading ────────────────────────────────────────────────────────────

def _load_label_profile(profile: str) -> tuple[dict, dict]:
    data = json.loads(_LABEL_PARAMS_FILE.read_text(encoding="utf-8"))
    if profile not in data:
        available = [k for k in data if not k.startswith("_")]
        raise ValueError(f"Profile {profile!r} not found. Available: {available}")
    entry = data[profile]
    return entry["regime_params"], entry["label_params"]


def _load_ohlcv(symbol: str, n_rows: int | None) -> pd.DataFrame:
    path = SYMBOL_DATASET_MAP[symbol]
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    if n_rows:
        df = df.tail(n_rows).reset_index(drop=True)
    return df


# ─── Label conversion (mirrors apply_multiclass_labels in train_prod_model_cli.py) ───

_ROLLOVER = ("21:30", "22:00")

def _events_to_bar_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Convert events DataFrame to bar-level target series (0=SELL, 1=FLAT, 2=BUY)."""
    target = pd.Series(1, index=df.index, dtype=int)
    winning = events[events["label"] == 1]
    buy_idx  = winning[winning["side"] == 1].index
    sell_idx = winning[winning["side"] == -1].index
    target.loc[target.index.isin(buy_idx)]  = 2
    target.loc[target.index.isin(sell_idx)] = 0

    # Rollover mask
    t_start = pd.to_datetime(_ROLLOVER[0]).time()
    t_end   = pd.to_datetime(_ROLLOVER[1]).time()
    rollover = (
        (df["Time"].dt.time >= t_start) &
        (df["Time"].dt.time <  t_end)
    )
    target[rollover] = 1
    return target


# ─── Tier 1: Label Statistics ─────────────────────────────────────────────────

def tier1_label_stats(
    df: pd.DataFrame,
    regime_params: dict,
    label_params: dict,
) -> dict:
    """
    Pure label statistics. No feature engineering.

    Cascade decomposition:
        All bars → regime filter → z-score filter → candidate signals
        Candidate signals → triple barrier → TP hit (SELL/BUY) or FLAT

    Key metrics:
        regime_coverage   — fraction of bars in trending regime
        candidate_density — fraction of bars that pass regime + z-score filter
        label_density     — fraction of bars ultimately labeled SELL or BUY
        tp_rate           — fraction of candidate signals that hit TP
        sl_rate           — fraction that hit SL
        timeout_rate      — fraction that expire at max_horizon
        implied_edge      — per-candidate expected ATR at current sl/tp rate
        break_even_prec   — model precision needed for positive expected PnL
        near_miss_density — non-TP candidate signals / total_bars (hardest FLATs)
    """
    t_start = time.time()
    total_bars = len(df)

    # Regime coverage
    regime = causal_market_regime(df, **regime_params)
    regime_cov   = float((regime != 0).mean())
    uptrend_pct  = float((regime == 1).mean())
    downtrend_pct = float((regime == -1).mean())

    # Full events (includes SL hits and timeouts, not filtered to TP-only)
    # Mirror the CLI trainer: inject regime_params into the kwarg dict (line 773 of train_prod_model_cli.py)
    full_label_params = {**label_params, "regime_params": regime_params}
    events = causal_triple_barrier_hilow_trend_labeler(df, **full_label_params)
    n_events = len(events)

    if n_events == 0:
        return {
            "total_bars": total_bars,
            "regime_coverage": 0.0,
            "uptrend_pct": 0.0,
            "downtrend_pct": 0.0,
            "candidate_density": 0.0,
            "label_density": 0.0,
            "n_sell_labels": 0,
            "n_buy_labels": 0,
            "tp_rate": 0.0,
            "sl_rate": 0.0,
            "timeout_rate": 0.0,
            "near_miss_density": 0.0,
            "implied_edge": float("nan"),
            "break_even_prec": float("nan"),
            "sell_buy_ratio": float("nan"),
            "avg_bars_to_resolution": float("nan"),
            "tier1_seconds": round(time.time() - t_start, 1),
        }

    tp_events  = events[events["label"] == 1]
    sl_events  = events[events["label"] == -1]
    to_events  = events[events["label"] == 0]

    tp_rate      = len(tp_events) / n_events
    sl_rate      = len(sl_events) / n_events
    timeout_rate = len(to_events) / n_events

    buy_tp  = tp_events[tp_events["side"] == 1]
    sell_tp = tp_events[tp_events["side"] == -1]

    label_density    = len(tp_events) / total_bars
    candidate_density = n_events / total_bars
    near_miss_density = (n_events - len(tp_events)) / total_bars

    tp_mult = label_params.get("tp_mult", 2.5)
    sl_mult = label_params.get("sl_mult", 2.5)

    # Expected ATR per candidate signal (will often be negative — expected for a filtered labeller)
    implied_edge = tp_rate * tp_mult - sl_rate * sl_mult

    # Break-even precision: precision p where p×tp_mult == (1-p)×(sl_rate×sl_mult)
    # Assumes false positives share the same sl_rate as the average candidate signal.
    # p = (sl_rate × sl_mult) / (tp_mult + sl_rate × sl_mult)
    sl_expected = sl_rate * sl_mult
    break_even_prec = sl_expected / (tp_mult + sl_expected) if (tp_mult + sl_expected) > 0 else float("nan")

    avg_horizon = float((events["t_end"] - events.index.to_series()).mean())

    sell_buy_ratio = float(len(sell_tp) / len(buy_tp)) if len(buy_tp) > 0 else float("nan")

    return {
        "total_bars":           total_bars,
        "regime_coverage":      round(regime_cov, 4),
        "uptrend_pct":          round(uptrend_pct, 4),
        "downtrend_pct":        round(downtrend_pct, 4),
        "candidate_density":    round(candidate_density, 4),
        "label_density":        round(label_density, 4),
        "n_sell_labels":        int(len(sell_tp)),
        "n_buy_labels":         int(len(buy_tp)),
        "tp_rate":              round(tp_rate, 4),
        "sl_rate":              round(sl_rate, 4),
        "timeout_rate":         round(timeout_rate, 4),
        "near_miss_density":    round(near_miss_density, 4),
        "implied_edge":         round(implied_edge, 4),
        "break_even_prec":      round(break_even_prec, 4),
        "sell_buy_ratio":       round(sell_buy_ratio, 4) if not np.isnan(sell_buy_ratio) else float("nan"),
        "avg_bars_to_resolution": round(avg_horizon, 1),
        "tier1_seconds":        round(time.time() - t_start, 1),
    }


# ─── Tier 2: Feature Separability ────────────────────────────────────────────

def tier2_feature_separability(
    df: pd.DataFrame,
    bar_labels: pd.Series,
    add_features_fn,
    mi_sample_rows: int = 150_000,
) -> dict:
    """
    Quantify how well the feature set separates the three classes.

    Metrics:
        feature_count         — number of features after preprocess
        mi_max                — highest mutual information score across all features
        mi_mean_top5          — mean MI of the 5 most informative features
        top_feature           — name of the most informative feature
        f_max                 — highest ANOVA F-statistic across all features
        class_dist_sell_flat  — Fisher normalised distance: SELL vs FLAT class centroids
        class_dist_buy_flat   — Fisher normalised distance: BUY vs FLAT class centroids
    """
    from sklearn.feature_selection import mutual_info_classif, f_classif

    t_start = time.time()

    df_feat = df.copy()
    df_feat["target"] = bar_labels.values

    try:
        df_feat = add_features_fn(df_feat)
    except Exception as e:
        return {"tier2_error": str(e), "tier2_seconds": round(time.time() - t_start, 1)}

    # Preprocess (scale continuous features; pass through binary/one-hot)
    try:
        X, y, _, feature_names, _, _ = preprocess_ohlcv(
            df_feat,
            target_col="target",
            onehot_prefixes=["OH_"],
            price_prefixes=["PR_"],
            return_df=True,
        )
    except Exception as e:
        return {"tier2_error": str(e), "tier2_seconds": round(time.time() - t_start, 1)}

    feature_count = X.shape[1]
    y = np.array(y, dtype=int)

    # Sample for MI/F computation (stable at 150k; faster than full 3M)
    n = len(X)
    if n > mi_sample_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, mi_sample_rows, replace=False)
        idx.sort()
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y

    # Mutual information
    mi = mutual_info_classif(X_s, y_s, random_state=42)
    mi_max       = float(np.max(mi))
    mi_mean_top5 = float(np.sort(mi)[-5:].mean())
    top_feature  = feature_names[int(np.argmax(mi))] if feature_names else f"feat_{int(np.argmax(mi))}"

    # ANOVA F-statistic
    f_vals, _ = f_classif(X_s, y_s)
    f_max = float(np.nanmax(f_vals))

    # Normalised class centroid distance (Fisher-style)
    # distance = ||mu_A - mu_B|| / (0.5*(std_A + std_B)) averaged over features
    mask_flat = y == 1
    mask_sell = y == 0
    mask_buy  = y == 2

    def _centroid_dist(mask_a, mask_b):
        if mask_a.sum() < 10 or mask_b.sum() < 10:
            return float("nan")
        mu_a = X[mask_a].mean(axis=0)
        mu_b = X[mask_b].mean(axis=0)
        std_a = X[mask_a].std(axis=0) + 1e-9
        std_b = X[mask_b].std(axis=0) + 1e-9
        pooled_std = 0.5 * (std_a + std_b)
        return float(np.mean(np.abs(mu_a - mu_b) / pooled_std))

    class_dist_sell_flat = _centroid_dist(mask_sell, mask_flat)
    class_dist_buy_flat  = _centroid_dist(mask_buy,  mask_flat)

    return {
        "feature_count":          feature_count,
        "mi_max":                 round(mi_max, 4),
        "mi_mean_top5":           round(mi_mean_top5, 4),
        "top_feature":            top_feature,
        "f_max":                  round(f_max, 2),
        "class_dist_sell_flat":   round(class_dist_sell_flat, 4),
        "class_dist_buy_flat":    round(class_dist_buy_flat, 4),
        "tier2_seconds":          round(time.time() - t_start, 1),
    }


# ─── Tier 3: Logistic Regression Probe ───────────────────────────────────────

def tier3_lr_probe(
    df: pd.DataFrame,
    bar_labels: pd.Series,
    add_features_fn,
    train_frac: float = 0.70,
) -> dict:
    """
    Per-bar logistic regression probe using a temporal train/test split.

    Uses per-bar features (no sequences). Logistic regression is a linear
    model — if it can separate classes at all, the labels have exploitable
    structure. Results are a LOWER BOUND on what a non-linear model (TCN/LSTM)
    can achieve.

    Metrics:
        lr_balanced_acc   — balanced accuracy on test split (random=0.33 for 3 classes)
        lr_macro_f1       — macro-averaged F1
        lr_sell_f1        — per-class F1 for SELL
        lr_flat_f1        — per-class F1 for FLAT
        lr_buy_f1         — per-class F1 for BUY
        lr_sell_prec      — precision for SELL predictions
        lr_buy_prec       — precision for BUY predictions
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score

    t_start = time.time()

    df_feat = df.copy()
    df_feat["target"] = bar_labels.values

    try:
        df_feat = add_features_fn(df_feat)
    except Exception as e:
        return {"tier3_error": str(e), "tier3_seconds": round(time.time() - t_start, 1)}

    try:
        X, y, _, _, _, _ = preprocess_ohlcv(
            df_feat,
            target_col="target",
            onehot_prefixes=["OH_"],
            price_prefixes=["PR_"],
            return_df=True,
        )
    except Exception as e:
        return {"tier3_error": str(e), "tier3_seconds": round(time.time() - t_start, 1)}

    y = np.array(y, dtype=int)
    n = len(X)
    split = int(n * train_frac)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Subsample training set for speed (100k is plenty for a linear model)
    if len(X_train) > 100_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), 100_000, replace=False)
        idx.sort()
        X_train, y_train = X_train[idx], y_train[idx]

    # Cap test set at 200k for speed
    if len(X_test) > 200_000:
        X_test = X_test[-200_000:]
        y_test = y_test[-200_000:]

    try:
        lr = LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=300,
            tol=1e-3,
            C=1.0,
            multi_class="multinomial",
            n_jobs=-1,
            random_state=42,
        )
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        bal_acc    = float(balanced_accuracy_score(y_test, y_pred))
        macro_f1   = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        per_f1     = f1_score(y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        per_prec   = precision_score(y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0)

    except Exception as e:
        return {"tier3_error": str(e), "tier3_seconds": round(time.time() - t_start, 1)}

    return {
        "lr_balanced_acc":  round(bal_acc, 4),
        "lr_macro_f1":      round(macro_f1, 4),
        "lr_sell_f1":       round(float(per_f1[0]), 4),
        "lr_flat_f1":       round(float(per_f1[1]), 4),
        "lr_buy_f1":        round(float(per_f1[2]), 4),
        "lr_sell_prec":     round(float(per_prec[0]), 4),
        "lr_buy_prec":      round(float(per_prec[2]), 4),
        "tier3_seconds":    round(time.time() - t_start, 1),
    }


# ─── Display helpers ──────────────────────────────────────────────────────────

def _fmt_pct(v, decimals=1) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v * 100:.{decimals}f}%"

def _fmt_f(v, decimals=3) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v:.{decimals}f}"

def _fmt_int(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{int(v):,}"

def _col(v: str, width: int) -> str:
    return v.rjust(width)


def print_comparison_table(
    profiles: list[str],
    results: dict[str, dict],
    tier: int,
):
    col_w  = 18
    label_w = 32

    sep = "─" * (label_w + col_w * len(profiles) + 3)
    eq  = "═" * len(sep)

    def hdr(title):
        print(f"\n  {title}")
        print("  " + sep)

    header_cols = "".join(_col(p[:col_w], col_w) for p in profiles)

    def rowp(label, fmt_fn):
        cells = []
        for p in profiles:
            v = results[p].get(label, float("nan"))
            if isinstance(v, str):
                cells.append(_col(v, col_w))
            else:
                try:
                    cells.append(_col(fmt_fn(v), col_w))
                except Exception:
                    cells.append(_col("err", col_w))
        print(f"  {label.ljust(label_w - 2)}" + "".join(cells))

    print(f"\n{eq}")
    print(f"  Label Quality Probe — {profiles[0].split('_')[0]}  |  {len(profiles)} profile(s)")
    print(f"{eq}")

    # ── Tier 1 ──────────────────────────────────────────────────────────────
    hdr("TIER 1: Label Statistics")
    print(f"  {'Metric'.ljust(label_w - 2)}" + header_cols)
    print("  " + sep)

    rowp("total_bars",             _fmt_int)
    rowp("regime_coverage",        _fmt_pct)
    rowp("uptrend_pct",            _fmt_pct)
    rowp("downtrend_pct",          _fmt_pct)
    rowp("candidate_density",      _fmt_pct)
    rowp("label_density",          _fmt_pct)
    rowp("n_sell_labels",          _fmt_int)
    rowp("n_buy_labels",           _fmt_int)
    rowp("sell_buy_ratio",         lambda v: _fmt_f(v, 3))
    rowp("tp_rate",                _fmt_pct)
    rowp("sl_rate",                _fmt_pct)
    rowp("timeout_rate",           _fmt_pct)
    rowp("near_miss_density",      _fmt_pct)
    rowp("implied_edge",           lambda v: _fmt_f(v, 3) + " ATR")
    rowp("break_even_prec",        _fmt_pct)
    rowp("avg_bars_to_resolution", lambda v: _fmt_f(v, 1) + " bars")
    rowp("tier1_seconds",          lambda v: _fmt_f(v, 1) + "s")

    if tier >= 2:
        hdr("TIER 2: Feature Separability")
        print(f"  {'Metric'.ljust(label_w - 2)}" + header_cols)
        print("  " + sep)
        rowp("feature_count",          _fmt_int)
        rowp("mi_max",                 lambda v: _fmt_f(v, 4))
        rowp("mi_mean_top5",           lambda v: _fmt_f(v, 4))
        rowp("f_max",                  lambda v: _fmt_f(v, 1))
        rowp("class_dist_sell_flat",   lambda v: _fmt_f(v, 4))
        rowp("class_dist_buy_flat",    lambda v: _fmt_f(v, 4))
        rowp("tier2_seconds",          lambda v: _fmt_f(v, 1) + "s")

        # Print top feature (string) separately
        cells = "".join(_col(results[p].get("top_feature", "n/a")[:col_w], col_w) for p in profiles)
        print(f"  {'top_feature'.ljust(label_w - 2)}" + cells)

    if tier >= 3:
        hdr("TIER 3: Logistic Regression Probe (per-bar, 70/30 time split)")
        print(f"  {'Metric'.ljust(label_w - 2)}" + header_cols)
        print("  " + sep)
        rowp("lr_balanced_acc",   _fmt_pct)
        rowp("lr_macro_f1",       lambda v: _fmt_f(v, 4))
        rowp("lr_sell_f1",        lambda v: _fmt_f(v, 4))
        rowp("lr_flat_f1",        lambda v: _fmt_f(v, 4))
        rowp("lr_buy_f1",         lambda v: _fmt_f(v, 4))
        rowp("lr_sell_prec",      lambda v: _fmt_f(v, 4))
        rowp("lr_buy_prec",       lambda v: _fmt_f(v, 4))
        rowp("tier3_seconds",     lambda v: _fmt_f(v, 1) + "s")

    print(f"\n{eq}")

    # ── Interpretation notes ───────────────────────────────────────────────
    print("\n  INTERPRETATION GUIDE")
    print("  " + sep)
    print("  label_density      5–12% is healthy. < 4% = label-bound (model may under-fit minority class).")
    print("  sell_buy_ratio     Target ≈ 1.0. Divergence > 0.15 suggests asymmetric regime coverage.")
    print("  tp_rate            37–42% is typical for trend-following configs. < 30% = noisy labelling.")
    print("  near_miss_density  High values (> 8%) = many hard FLAT examples; increases task difficulty.")
    print("  break_even_prec    Estimated precision needed for +EV trading at this sl_rate/tp_mult.")
    print("  implied_edge       Negative is normal (labeller selects TP hits only; raw edge is weak).")
    if tier >= 2:
        print("  mi_max             > 0.03 = meaningful signal in at least one feature.")
        print("  class_dist_*       > 0.10 = classes are meaningfully separated in feature space.")
    if tier >= 3:
        print("  lr_balanced_acc    Random = 33.3%. > 40% = learnable signal present.")
        print("  lr_sell/buy_prec   Compare to break_even_prec. Must exceed it for +EV live trades.")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Label quality probe — compare label profiles before training."
    )
    parser.add_argument("--symbol",         required=True, choices=list(SYMBOL_DATASET_MAP.keys()))
    parser.add_argument("--label-profiles", required=True, nargs="+", metavar="PROFILE")
    parser.add_argument("--tier",           type=int, default=3, choices=[1, 2, 3],
                        help="Analysis depth: 1=label stats, 2=+features, 3=+LR probe (default)")
    parser.add_argument("--n-rows",         type=int, default=None,
                        help="Use only the last N rows of the dataset (None = full dataset)")
    parser.add_argument("--save",           action="store_true",
                        help="Save results as JSON to ModelWorkbench/params/")
    args = parser.parse_args()

    symbol   = args.symbol
    profiles = args.label_profiles
    tier     = args.tier

    add_features_fn = SYMBOL_FEATURES_MAP[symbol]

    print(f"\nLoading {symbol} dataset…")
    df = _load_ohlcv(symbol, args.n_rows)
    print(f"  {len(df):,} bars  |  {df['Time'].min().date()} → {df['Time'].max().date()}")

    all_results: dict[str, dict] = {}

    for profile in profiles:
        print(f"\n{'─'*60}")
        print(f"  Profile: {profile}")
        regime_params, label_params = _load_label_profile(profile)

        # ── Tier 1 ──
        print("  [1/3] Computing label statistics…", end=" ", flush=True)
        t1 = tier1_label_stats(df, regime_params, label_params)
        print(f"done ({t1['tier1_seconds']}s)")
        result = dict(t1)

        # ── Tier 2 ──
        if tier >= 2:
            print("  [2/3] Computing feature separability…", end=" ", flush=True)
            full_lp = {**label_params, "regime_params": regime_params}
            bar_labels = _events_to_bar_labels(
                df,
                causal_triple_barrier_hilow_trend_labeler(df, **full_lp),
            )
            t2 = tier2_feature_separability(df, bar_labels, add_features_fn)
            print(f"done ({t2.get('tier2_seconds', '?')}s)")
            result.update(t2)
        else:
            bar_labels = None

        # ── Tier 3 ──
        if tier >= 3:
            print("  [3/3] Running logistic regression probe…", end=" ", flush=True)
            if bar_labels is None:
                full_lp = {**label_params, "regime_params": regime_params}
                bar_labels = _events_to_bar_labels(
                    df,
                    causal_triple_barrier_hilow_trend_labeler(df, **full_lp),
                )
            t3 = tier3_lr_probe(df, bar_labels, add_features_fn)
            print(f"done ({t3.get('tier3_seconds', '?')}s)")
            result.update(t3)

        all_results[profile] = result

    # ── Print comparison table ─────────────────────────────────────────────
    print_comparison_table(profiles, all_results, tier)

    # ── Optionally save JSON ───────────────────────────────────────────────
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _PARAMS_DIR / f"label_quality_{symbol}_{ts}.json"
        payload = {
            "generated": ts,
            "symbol": symbol,
            "tier": tier,
            "n_rows": args.n_rows,
            "profiles": all_results,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
