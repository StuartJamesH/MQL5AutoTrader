"""
eval_model_packs.py
===================
Batch-evaluate all model packs in Engine/Model Packs against a recent window
of data.  Produces a ranked DataFrame of pnl / gated_pnl / recent_gated_pnl
and saves results to a CSV.

Run from ModelWorkbench:
    ..\.venv\Scripts\python.exe eval_model_packs.py
"""

import math
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

# ── User config ───────────────────────────────────────────────────────────────
SYMBOL       = "US500"
TP_MULT      = 1.5       # take-profit multiplier applied to ALL model outcomes
SL_MULT      = 1.0       # stop-loss multiplier applied to ALL model outcomes
START_DATE   = "2026-05-10"   # evaluate bars from this date; set to None to use N_BARS
N_BARS       = None   # bars to load when START_DATE is None
RECENCY_BARS = 2880    # tail window for recent_gated_pnl diagnostic
BATCH_SIZE   = 512
# ─────────────────────────────────────────────────────────────────────────────

# Resolve paths relative to this file so the script can be launched from anywhere
_HERE            = Path(__file__).parent
_REPO_ROOT       = _HERE.parent
MODEL_PACKS_DIR  = Path(r"D:\.archive") # _REPO_ROOT / "Engine" / "Model Packs"
OUTPUT_CSV       = _HERE / "eval_model_packs_results.csv"

sys.path.insert(0, str(_HERE))

from Learn.labels import calculate_trade_outcomes_all_candles
from Learn.Loaders import SequenceDataset
from Learn.preprocess import preprocess_ohlcv
from train_prod_model_cli import (
    SYMBOL_DATASET_MAP,
    _build_val_gate_arrays,
    _compute_gated_pnl,
    _session_seq_indices,
    apply_multiclass_labels,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw_data(symbol: str, start_date: str | None, n_bars: int) -> pd.DataFrame:
    ds_path = _REPO_ROOT / SYMBOL_DATASET_MAP[symbol]
    df = pd.read_csv(ds_path)
    df = df.sort_values("Time").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = pd.to_datetime(df["Time"])

    if start_date is not None:
        cutoff = pd.Timestamp(start_date)
        # Match timezone of the data so tz-aware/naive comparison doesn't raise
        if df["Time"].dt.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(df["Time"].dt.tz)
        df = df[df["Time"] >= cutoff].reset_index(drop=True)
        print(f"  Data filtered from {start_date}: {len(df):,} bars")
    else:
        df = df.tail(n_bars).reset_index(drop=True)
        print(f"  Data tail {n_bars:,} bars loaded")

    return df


# ── Per-pack inference ─────────────────────────────────────────────────────────

def _run_inference(
    model: torch.nn.Module,
    X: np.ndarray,
    y_int: list,
    outcomes_2d: np.ndarray,
    seq_len: int,
    seq_idx: list,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = SequenceDataset(
        X, y_int, seq_len=seq_len,
        df_idx=list(range(len(X))),
        custom_targets=None,
        trade_outcomes=outcomes_2d,
        seq_idx_filter=seq_idx,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_targets, all_outcomes = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, ob in loader:
            xb = xb.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(yb.numpy().tolist())
            all_outcomes.extend(ob.numpy().tolist())

    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_outcomes),
    )


def _compute_raw_pnl(
    preds: np.ndarray, outcomes: np.ndarray
) -> tuple[float, float, float, int, int]:
    sell_mask   = preds == 0
    buy_mask    = preds == 2
    profit_sell = float(outcomes[sell_mask, 0].sum()) if sell_mask.any() else 0.0
    profit_buy  = float(outcomes[buy_mask,  1].sum()) if buy_mask.any()  else 0.0
    return (
        profit_sell + profit_buy,
        profit_sell,
        profit_buy,
        int(sell_mask.sum()),
        int(buy_mask.sum()),
    )


def evaluate_pack(
    pack_path: Path,
    df_raw: pd.DataFrame,
    device: torch.device,
) -> dict | None:
    """Load one model pack, run inference on df_raw, and return a metrics dict."""
    print(f"\n[{pack_path.name}]")

    with open(pack_path, "rb") as fh:
        pack = pickle.load(fh)

    features_fn      = pack["feature_function"]
    label_params     = pack["label_params"]
    regime_params    = pack["regime_params"]
    outcome_params_p = pack["outcome_params"]
    rollover         = pack["rollover_window"]
    seq_len          = pack["input_shape"][0]
    scaler           = pack["scaler"]
    model_cls        = pack["model_class"]
    model_params     = pack["model_params"]
    model_info       = pack["model_info"]
    preprocess_args  = pack["preprocess_args"]

    # Override TP/SL with globally declared values; retain pack's ATR window
    outcome_params = {
        "atr_window": outcome_params_p["atr_window"],
        "tp_mult":    TP_MULT,
        "sl_mult":    SL_MULT,
    }

    # ── Labels + outcomes ──────────────────────────────────────────────────
    df = apply_multiclass_labels(df_raw.copy(), label_params, rollover)
    outcomes_df = calculate_trade_outcomes_all_candles(df, **outcome_params)
    df["sell_y"] = outcomes_df["sell_outcome"].fillna(0.0)
    df["buy_y"]  = outcomes_df["buy_outcome"].fillna(0.0)

    # ── Feature engineering ────────────────────────────────────────────────
    try:
        df_feat = features_fn(df, include_mtf=True, regime_params=regime_params)
    except TypeError:
        df_feat = features_fn(df, regime_params=regime_params)

    # ── Preprocess (use stored scaler — no re-fitting) ─────────────────────
    X, y, _, _, _, proc_df = preprocess_ohlcv(
        df_feat.copy(),
        **preprocess_args,
        scaler=scaler,
        return_df=True,
    )
    X = np.ascontiguousarray(X)
    y_int = [int(v) for v in y]

    outcomes_2d = np.stack(
        [proc_df["sell_y"].values, proc_df["buy_y"].values], axis=1
    ).astype(float)

    # ── Sequence indices ────────────────────────────────────────────────────
    val_times = proc_df["Time"].to_numpy()
    seq_idx   = _session_seq_indices(val_times, seq_len, trading_hours=None)

    # ── Gate arrays ─────────────────────────────────────────────────────────
    # row_offset corrects for NaN rows dropped by preprocess_ohlcv
    row_offset  = len(df_raw) - len(X)
    gate_arrays = _build_val_gate_arrays(
        df_raw, regime_params, seq_len,
        val_seq_idx=seq_idx,
        row_offset=row_offset,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = model_cls(**model_params).to(device)
    model.load_state_dict({k: v.to(device) for k, v in pack["model"].items()})

    print(
        f"  seq_len={seq_len}  X={X.shape}  n_seqs={len(seq_idx)}  "
        f"row_offset={row_offset}"
    )

    # ── Inference ───────────────────────────────────────────────────────────
    preds, targets, outcomes = _run_inference(
        model, X, y_int, outcomes_2d, seq_len, seq_idx, device
    )

    # ── Raw PnL ─────────────────────────────────────────────────────────────
    pnl, pnl_sell, pnl_buy, n_sell, n_buy = _compute_raw_pnl(preds, outcomes)

    # ── Gated PnL ───────────────────────────────────────────────────────────
    g_pnl, g_ppt, g_n_sell, g_n_buy, g_pnl_sell, g_pnl_buy = _compute_gated_pnl(
        preds, outcomes, gate_arrays
    )

    # ── Recent gated PnL (tail window) ──────────────────────────────────────
    rec_n      = min(RECENCY_BARS, len(preds))
    rec_gate   = {k: v[-rec_n:] for k, v in gate_arrays.items()}
    r_pnl, r_ppt, r_n_sell, r_n_buy, _, _ = _compute_gated_pnl(
        preds[-rec_n:], outcomes[-rec_n:], rec_gate
    )

    # ── Classification metrics ───────────────────────────────────────────────
    acc = float(accuracy_score(targets, preds))
    p_per, r_per, f_per, _ = precision_recall_fscore_support(
        targets, preds, labels=[0, 1, 2], average=None, zero_division=0
    )

    checkpoint_type = model_info.get("checkpoint_type", "standard")

    print(
        f"  pnl={pnl:.1f}  gated_pnl={g_pnl:.1f}  "
        f"recent_gated_pnl={r_pnl:.1f}  acc={acc:.3f}"
    )

    return {
        "model_pack":          pack_path.name,
        "checkpoint_type":     checkpoint_type,
        "symbol":              SYMBOL,
        "tp_mult":             TP_MULT,
        "sl_mult":             SL_MULT,
        "date_trained":        model_info.get("date_trained", ""),
        "model_type":          model_info.get("model_type", ""),
        "model_version":       model_info.get("model_version", ""),
        "label_profile":       model_info.get("label_profile", ""),
        "model_profile":       model_info.get("model_profile", ""),
        "loss_profile":        model_info.get("loss_profile", ""),
        "seq_len":             seq_len,
        "n_preds":             len(preds),
        "n_preds_sell":        n_sell,
        "n_preds_buy":         n_buy,
        # Raw PnL
        "pnl":                 pnl,
        "pnl_sell":            pnl_sell,
        "pnl_buy":             pnl_buy,
        "ppt":                 (pnl / (n_sell + n_buy)) if (n_sell + n_buy) > 0 else float("nan"),
        # Gated PnL (regime + breakout + Donchian gates)
        "gated_pnl":           g_pnl,
        "gated_pnl_sell":      g_pnl_sell,
        "gated_pnl_buy":       g_pnl_buy,
        "gated_ppt":           g_ppt,
        "n_gated_sell":        g_n_sell,
        "n_gated_buy":         g_n_buy,
        # Recent gated PnL (last RECENCY_BARS sequences)
        "recent_gated_pnl":    r_pnl,
        "recent_gated_ppt":    r_ppt,
        "n_recent_gated_sell": r_n_sell,
        "n_recent_gated_buy":  r_n_buy,
        # Classification metrics
        "accuracy":            acc,
        "prec_sell":           float(p_per[0]),
        "prec_buy":            float(p_per[2]),
        "rec_sell":            float(r_per[0]),
        "rec_buy":             float(r_per[2]),
        "f1_sell":             float(f_per[0]),
        "f1_buy":              float(f_per[2]),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: load data
    print(f"\nLoading {SYMBOL} data...")
    df_raw = load_raw_data(SYMBOL, START_DATE, N_BARS)
    print(
        f"  Date range: {df_raw['Time'].iloc[0]}  →  {df_raw['Time'].iloc[-1]}"
    )

    # Discover all model packs (including gated checkpoints, excluding subdirs
    # that use a different naming convention)
    pack_paths = sorted(MODEL_PACKS_DIR.rglob("*_model.pkl"))
    print(f"\nFound {len(pack_paths)} model pack(s) in {MODEL_PACKS_DIR}")

    results = []
    for pack_path in pack_paths:
        try:
            row = evaluate_pack(pack_path, df_raw, device)
            if row is not None:
                results.append(row)
        except Exception:
            print(f"  ERROR evaluating {pack_path.name}:")
            traceback.print_exc()

    if not results:
        print("\nNo results — check errors above.")
        return

    df_results = pd.DataFrame(results)

    # Sort by gated_pnl descending so the best model appears first
    df_results = df_results.sort_values("gated_pnl", ascending=False).reset_index(drop=True)

    # Save
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")

    # Display summary table
    display_cols = [
        "model_version", "checkpoint_type",
        "pnl", "gated_pnl", "recent_gated_pnl",
        "gated_ppt", "n_gated_sell", "n_gated_buy",
        "prec_sell", "prec_buy", "accuracy",
        "label_profile", "loss_profile",
    ]
    available = [c for c in display_cols if c in df_results.columns]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n── Results (sorted by gated_pnl) ──")
    print(df_results[available].to_string(index=True))


if __name__ == "__main__":
    main()
