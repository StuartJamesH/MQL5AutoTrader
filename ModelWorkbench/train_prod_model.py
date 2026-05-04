import inspect
import json
import logging
import math
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from Learn.features import _add_features_XAUUSD, _add_features_EURUSD, _add_features_US500
from Learn.labels import causal_triple_barrier_hilow_trend_labeler, calculate_trade_outcomes_all_candles
from Learn.preprocess import preprocess_ohlcv
from Learn.Loaders import SequenceDataset
from Learn.Models import LSTMAttentionSEClassifier, TCNAttentionSEClassifier
from Learn.Loss import TradeProfitabilityLoss


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DS_NAME = "data/US500_M1_520weeks.csv"
N_ROWS = None
FEATURES = _add_features_US500

# Use a fixed recent tail for validation; train on everything before it.
VAL_BARS = 50_000
MIN_TRAIN_ROWS = 20_000

SEQ_LEN = 256
BATCH_SIZE = 512
NUM_EPOCHS = 30
PATIENCE = 7        # early-stop after this many epochs without val-loss improvement
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-6

ROLLOVER_WINDOW = ("21:30", "22:00")
TRADING_HOURS = None
COMMISSION = 0.00

MODEL_ARCH    = "LSTM"   # "LSTM" | "TCN"
MODEL_VERSION = "prod"
RESUME_MODEL_PACK: str | None = None  # Set to a .pkl model pack path to resume training from that checkpoint
OUTPUT_DIR = Path("Engine/Model Packs")
LOG_FILE = Path("Engine/train_multiclass_prod.log")
CLOUD_LOG = True  # <-- set False to disable mirroring to CLOUD_LOG_DIR in .env

regime_params = {
      "ma_period": 60,
      "slope_smoothness": 50,
      "regime_min_duration": 0,
      "atr_window": 60,
      "atr_lookback": 720,
      "atr_percentile": 0.0,
      "slope_threshold": 5e-6 # 0.06 Gold
    }

label_params = {
      "z_window": 14,
      "z_thresh": 1,
      "z_limit": 5,
      "atr_window": 14,
      "tp_mult": 2.5,
      "sl_mult": 2.5,
      "max_horizon": 90,
      "trend_pullback_thresh": 1.2,
      "regime_params": regime_params,
      "skip_range": True
    }

outcome_params = {
    k: v for k, v in label_params.items() if k in ["atr_window", "tp_mult", "sl_mult"]
}

lstm_model_params = {
    "input_dim": None,
    'hidden_dim':         256,
    'num_layers':         3,
    'num_classes':        3,
    'bidirectional':      True,
    'dropout':            0.15,  # was 0.20 — lower inter-layer suppression compensates for tight head
    'dropout_out':        0.50,  # was 0.38 — aggressive; only fires on strong consistent activations
    'attn_heads':         8,
    'attn_dropout':       0.05,  # was 0.08
    'se_context_window':  32,   # new: SE context window (in timesteps) for local feature recalibration
    'use_learned_query':  True,
    'bias_init':          None,
}

# TCN: kernel_size=3, num_layers=6 → receptive field ≈ 253 bars (matches SEQ_LEN=256)
tcn_model_params = {
    "input_dim": None,
    "hidden_channels": 256,
    "num_layers": 6,
    "kernel_size": 3,
    "num_classes": 3,
    "dropout": 0.20,
    "dropout_out": 0.40,
    "attn_heads": 8,
    "attn_dropout": 0.10,
    "use_learned_query": True,
    "se_context_window": 32,
    "bias_init": None,
}

loss_params_template = {
    'alpha':             None,
    'gamma':             2.5,
    'trade_classes':     (0, 2),
    'pr_weight':         12.0,   # primary precision lever
    'recall_floor':      0.20,   # hinge activates below this recall per class
    'rec_floor_weight':  60.0,   # quadratic hinge strength
    'direction_penalty': 1.5,    # SELL↔BUY confusion cost
    'eps':               1e-6,
}


def setup_logging(cloud_log: bool = True) -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_multiclass_prod")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if cloud_log:
        load_dotenv()
        cloud_dir = os.getenv("CLOUD_LOG_DIR")
        if cloud_dir:
            try:
                os.makedirs(cloud_dir, exist_ok=True)
                cloud_path = os.path.join(cloud_dir, LOG_FILE.name)
                cloud_handler = logging.FileHandler(cloud_path, encoding="utf-8")
                cloud_handler.setFormatter(formatter)
                logger.addHandler(cloud_handler)
            except OSError as exc:
                logger.warning(
                    "Could not set up cloud log at '%s': %s — logging locally only.",
                    cloud_dir, exc,
                )
        else:
            logger.warning(
                "CLOUD_LOG=True but CLOUD_LOG_DIR is not set in .env — logging locally only."
            )

    return logger


def load_ohlcv(ds_name: str, n_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(ds_name)
    if n_rows is not None:
        df = df.tail(n_rows)
    return df.sort_values("Time").reset_index(drop=True)


def split_train_val_tail(df: pd.DataFrame, val_bars: int, min_train_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_bars <= 0:
        raise ValueError(f"VAL_BARS must be > 0, got {val_bars}")
    if val_bars >= len(df):
        raise ValueError(f"VAL_BARS ({val_bars}) must be smaller than dataset size ({len(df)})")

    df_train = df.iloc[:-val_bars, :].copy().reset_index(drop=True)
    df_val = df.iloc[-val_bars:, :].copy().reset_index(drop=True)

    if len(df_train) < min_train_rows:
        raise ValueError(
            f"Training rows too small after split: {len(df_train)} < MIN_TRAIN_ROWS ({min_train_rows}). "
            "Reduce VAL_BARS or MIN_TRAIN_ROWS."
        )

    return df_train, df_val


def apply_multiclass_labels(df: pd.DataFrame, label_params_: dict | None = None, rollover_window_: tuple | None = None) -> pd.DataFrame:
    lp = label_params_ if label_params_ is not None else label_params
    rw = rollover_window_ if rollover_window_ is not None else ROLLOVER_WINDOW
    d = df.copy()
    signals = causal_triple_barrier_hilow_trend_labeler(d, **lp).rename(columns={"side": "target"})
    winning = signals[signals["label"] == 1]

    d["target"] = 1
    d.loc[winning[winning["target"] == 1].index, "target"] = 2
    d.loc[winning[winning["target"] == -1].index, "target"] = 0

    if not pd.api.types.is_datetime64_any_dtype(d["Time"]):
        d["Time"] = pd.to_datetime(d["Time"])

    rollover = (
        (d["Time"].dt.time >= pd.to_datetime(rw[0]).time())
        & (d["Time"].dt.time < pd.to_datetime(rw[1]).time())
    )
    d.loc[rollover, "target"] = 1
    return d


def add_outcomes(df: pd.DataFrame, outcome_params_: dict | None = None) -> pd.DataFrame:
    # Outcomes are binary: 1 = TP hit, -1 = SL hit, NaN = unresolved (fillna'd to 0).
    # TP outcomes are doubled to reflect the asymmetric reward of letting winners run.
    op = outcome_params_ if outcome_params_ is not None else outcome_params
    d = df.copy()
    outcomes = calculate_trade_outcomes_all_candles(d, **op)

    for col in ["buy_outcome", "sell_outcome"]:
        outcomes[col] = outcomes[col].where(outcomes[col] <= 0, outcomes[col] * 2)

    d["sell_y"] = outcomes["sell_outcome"].fillna(0.0)
    d["buy_y"]  = outcomes["buy_outcome"].fillna(0.0)
    return d


def _session_seq_indices(times: np.ndarray, seq_len: int, trading_hours: tuple[str, str] | None) -> list[int]:
    if trading_hours is None:
        return list(range(len(times) - seq_len))

    t_start = pd.to_datetime(trading_hours[0]).time()
    t_end = pd.to_datetime(trading_hours[1]).time()

    return [
        i for i in range(len(times) - seq_len)
        if t_start <= pd.Timestamp(times[i + seq_len - 1]).time() < t_end
    ]


def build_dataloaders(df_train: pd.DataFrame, df_val: pd.DataFrame, logger: logging.Logger, resume_scaler=None, seq_len: int | None = None):
    _seq_len = seq_len if seq_len is not None else SEQ_LEN
    preprocess_ohlcv_args = {
        "target_col": "target",
        "outcomes_col": None,
        "shift": 0,
        "onehot_prefixes": ["OH_"],
        "price_prefixes": ["PR_"],
    }

    X_train, y_train, scaler, features, _, proc_df_train = preprocess_ohlcv(
        df_train.copy(), **preprocess_ohlcv_args, scaler=resume_scaler, return_df=True
    )
    X_val, y_val, _, _, _, proc_df_val = preprocess_ohlcv(
        df_val.copy(), **preprocess_ohlcv_args, scaler=scaler, return_df=True
    )

    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val)

    y_train = [int(x) for x in y_train]
    y_val = [int(x) for x in y_val]

    outcomes_train_2d = np.stack([proc_df_train["sell_y"].values, proc_df_train["buy_y"].values], axis=1).astype(float)
    outcomes_val_2d = np.stack([proc_df_val["sell_y"].values, proc_df_val["buy_y"].values], axis=1).astype(float)

    train_times = proc_df_train["Time"].to_numpy()
    val_times = proc_df_val["Time"].to_numpy()

    train_seq_idx = _session_seq_indices(train_times, _seq_len, TRADING_HOURS)
    val_seq_idx = _session_seq_indices(val_times, _seq_len, TRADING_HOURS)

    train_ds = SequenceDataset(
        X_train,
        y_train,
        seq_len=_seq_len,
        df_idx=list(range(len(X_train))),
        custom_targets=None,
        trade_outcomes=outcomes_train_2d,
        seq_idx_filter=train_seq_idx,
    )
    val_ds = SequenceDataset(
        X_val,
        y_val,
        seq_len=_seq_len,
        df_idx=list(range(len(X_val))),
        custom_targets=None,
        trade_outcomes=outcomes_val_2d,
        seq_idx_filter=val_seq_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    logger.info(
        "Preprocess complete | X_train=%s X_val=%s | seqs train=%d val=%d",
        X_train.shape,
        X_val.shape,
        len(train_ds),
        len(val_ds),
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "features": features,
        "scaler": scaler,
        "preprocess_ohlcv_args": preprocess_ohlcv_args,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds, all_targets, all_outcomes = [], [], []

    with torch.no_grad():
        for xb, yb, outcome_b in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outcome_b = outcome_b.to(device) if isinstance(outcome_b, torch.Tensor) else outcome_b

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb, outcome_b)

            bs = yb.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(yb.cpu().numpy().tolist())
            all_outcomes.extend(outcome_b.cpu().numpy().tolist())

    val_loss = total_loss / max(1, total)
    all_preds_np = np.asarray(all_preds)
    all_targets_np = np.asarray(all_targets)
    all_outcomes_np = np.asarray(all_outcomes)

    p_per, r_per, f_per, _ = precision_recall_fscore_support(
        all_targets_np, all_preds_np, labels=[0, 1, 2], average=None, zero_division=0
    )

    sell_mask = all_preds_np == 0
    buy_mask = all_preds_np == 2

    sell_net = all_outcomes_np[:, 0] - COMMISSION * np.abs(all_outcomes_np[:, 0])
    buy_net = all_outcomes_np[:, 1] - COMMISSION * np.abs(all_outcomes_np[:, 1])
    profit_sell = float(sell_net[sell_mask].sum()) if sell_mask.any() else 0.0
    profit_buy = float(buy_net[buy_mask].sum()) if buy_mask.any() else 0.0
    profit = profit_sell + profit_buy

    return {
        "val_loss": val_loss,
        "acc": float(accuracy_score(all_targets_np, all_preds_np)),
        "f1_sell": float(f_per[0]),
        "prec_sell": float(p_per[0]),
        "rec_sell": float(r_per[0]),
        "f1_buy": float(f_per[2]),
        "prec_buy": float(p_per[2]),
        "rec_buy": float(r_per[2]),
        "profit": float(profit),
        "profit_sell": float(profit_sell),
        "profit_buy": float(profit_buy),
        "targets": all_targets_np,
        "preds": all_preds_np,
    }


def save_plots(model_name: str, history: dict, eval_pack: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(history["val_losses_all"])
    axes[0, 0].axvline(history["best_epoch"], color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Val Loss")

    axes[0, 1].plot(history["f1_sell"], label="F1 SELL", color="tab:blue")
    axes[0, 1].plot(history["f1_buy"], label="F1 BUY", color="tab:green")
    axes[0, 1].plot(history["rec_sell"], label="Rec SELL", color="tab:blue", linestyle="--", alpha=0.6)
    axes[0, 1].plot(history["rec_buy"], label="Rec BUY", color="tab:green", linestyle="--", alpha=0.6)
    ax2 = axes[0, 1].twinx()
    ax2.plot(history["prec_sell"], label="Prec SELL", color="tab:orange")
    ax2.plot(history["prec_buy"], label="Prec BUY", color="tab:red")
    axes[0, 1].set_title("F1 & Recall (left) | Precision (right)")
    axes[0, 1].legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    ConfusionMatrixDisplay(
        confusion_matrix(eval_pack["targets"], eval_pack["preds"]),
        display_labels=["SELL", "FLAT", "BUY"],
    ).plot(cmap="Blues", values_format="d", ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix (best checkpoint)")

    axes[1, 1].plot(history["pnl"], color="tab:blue", linewidth=2, label="Total")
    axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.3)
    axes[1, 1].set_title(f"Val Profit after {COMMISSION:.0%} commission")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{model_name}_plots.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logger = setup_logging(cloud_log=CLOUD_LOG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Resume from model pack ---
    resume_pack = None
    resume_model_pack_path = None
    if RESUME_MODEL_PACK:
        resume_model_pack_path = Path(RESUME_MODEL_PACK)
        logger.info("Loading resume pack: %s", resume_model_pack_path)
        with open(resume_model_pack_path, "rb") as _f:
            resume_pack = pickle.load(_f)
        logger.info(
            "Resume pack loaded | model_type=%s seq_len=%d best_prev_epoch=%d",
            resume_pack["model_info"].get("model_type", "unknown"),
            resume_pack["input_shape"][0],
            resume_pack["model_info"].get("best_epoch", -1),
        )

    df_all = load_ohlcv(DS_NAME, N_ROWS)
    if not pd.api.types.is_datetime64_any_dtype(df_all["Time"]):
        df_all["Time"] = pd.to_datetime(df_all["Time"])

    df_train_raw, df_val_raw = split_train_val_tail(df_all, VAL_BARS, MIN_TRAIN_ROWS)
    logger.info(
        "Tail split complete | total=%d train=%d val=%d | train_end=%s val_start=%s",
        len(df_all),
        len(df_train_raw),
        len(df_val_raw),
        df_train_raw["Time"].iloc[-1],
        df_val_raw["Time"].iloc[0],
    )

    _label_params    = resume_pack["label_params"]    if resume_pack else label_params
    _regime_params   = resume_pack["regime_params"]   if resume_pack else regime_params
    _outcome_params  = resume_pack["outcome_params"]  if resume_pack else outcome_params
    _rollover_window = resume_pack["rollover_window"] if resume_pack else ROLLOVER_WINDOW
    _features_fn     = resume_pack["feature_function"] if resume_pack else FEATURES
    _seq_len         = resume_pack["input_shape"][0]  if resume_pack else SEQ_LEN

    df_train = add_outcomes(apply_multiclass_labels(df_train_raw, _label_params, _rollover_window), _outcome_params)
    df_val   = add_outcomes(apply_multiclass_labels(df_val_raw,   _label_params, _rollover_window), _outcome_params)

    logger.info(
        "Label dist train=%s | val=%s",
        df_train["target"].value_counts(normalize=True).sort_index().to_dict(),
        df_val["target"].value_counts(normalize=True).sort_index().to_dict(),
    )

    # include_mtf=True enables higher-timeframe features where supported.
    # Wrap in try/except for backward compatibility when resuming from old packs
    # whose stored feature function may not accept the include_mtf kwarg.
    try:
        df_train = _features_fn(df_train, include_mtf=True, regime_params=_regime_params)
        df_val   = _features_fn(df_val,   include_mtf=True, regime_params=_regime_params)
    except TypeError:
        df_train = _features_fn(df_train, regime_params=_regime_params)
        df_val   = _features_fn(df_val,   regime_params=_regime_params)

    _resume_scaler = resume_pack["scaler"] if resume_pack else None
    data_pack = build_dataloaders(df_train, df_val, logger, resume_scaler=_resume_scaler, seq_len=_seq_len)

    y_train_arr = np.array(data_pack["y_train"])
    p_sell = float((y_train_arr == 0).mean())
    p_flat = float((y_train_arr == 1).mean())
    p_buy = float((y_train_arr == 2).mean())
    bias_init = [math.log(p_sell + 1e-8), math.log(p_flat + 1e-8), math.log(p_buy + 1e-8)]

    if resume_pack:
        model_cls = resume_pack["model_class"]
        model_params_local = dict(resume_pack["model_params"])
        model_params_local["bias_init"] = bias_init
        model_type = resume_pack["model_info"].get("model_type", "LSTM_Multiclass")
    else:
        _MODEL_REGISTRY = {
            "LSTM": (LSTMAttentionSEClassifier, lstm_model_params),
            "TCN":  (TCNAttentionSEClassifier,  tcn_model_params),
        }
        if MODEL_ARCH not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown MODEL_ARCH {MODEL_ARCH!r}. Choose 'LSTM' or 'TCN'.")
        model_cls, base_params = _MODEL_REGISTRY[MODEL_ARCH]
        model_type = f"{MODEL_ARCH}_Multiclass"

        model_params_local = dict(base_params)
        model_params_local["input_dim"] = int(data_pack["X_train"].shape[1])
        model_params_local["bias_init"] = bias_init

    model = model_cls(**model_params_local).to(device)
    if resume_pack:
        model.load_state_dict({k: v.to(device) for k, v in resume_pack["model"].items()})
        logger.info("Loaded model weights from resume pack.")

    counts = np.maximum(np.bincount(y_train_arr, minlength=3), 1)
    raw_weights = (len(y_train_arr) / (len(counts) * counts)).astype(float)
    weights = np.power(raw_weights, 0.6)
    # Cap FLAT weight so it cannot exceed the mean of the two trade-class weights.
    # pr_weight in TradeProfitabilityLoss already drives trade selectivity; allowing
    # a large FLAT weight in focal CE compounds that and suppresses minority gradients.
    weights[1] = min(weights[1], (weights[0] + weights[2]) / 2.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    loss_params = dict(loss_params_template)
    loss_params["alpha"] = class_weights

    criterion = TradeProfitabilityLoss(**loss_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    warmup_steps = 400
    total_steps = max(1, len(data_pack["train_loader"]) * NUM_EPOCHS)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "train_losses": [],
        "val_losses_all": [],
        "f1_sell": [],
        "prec_sell": [],
        "rec_sell": [],
        "f1_buy": [],
        "prec_buy": [],
        "rec_buy": [],
        "pnl": [],
        "best_val_loss": float("inf"),
        "best_epoch": -1,
    }
    best_model_state = None
    epochs_no_improve = 0

    logger.info("Training start | epochs=%d | batches train=%d val=%d", NUM_EPOCHS, len(data_pack["train_loader"]), len(data_pack["val_loader"]))

    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss_sum = 0.0
            n_train = 0

            for xb, yb, outcome_b in tqdm(data_pack["train_loader"], desc=f"Epoch {epoch}"):
                xb = xb.to(device)
                yb = yb.to(device)
                outcome_b = outcome_b.to(device) if isinstance(outcome_b, torch.Tensor) else outcome_b

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(xb), yb, outcome_b)

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                # NaN/inf guard — AMP (bfloat16) can produce inf gradients on rare sequences;
                # without this, clip_grad_norm_ returns inf → clip_coef=0 → all grads zeroed.
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                scheduler.step()

                bs = yb.size(0)
                train_loss_sum += float(loss.item()) * bs
                n_train += bs

            train_loss_epoch = train_loss_sum / max(1, n_train)
            history["train_losses"].append(float(train_loss_epoch))

            eval_pack = evaluate(model, data_pack["val_loader"], criterion, device)
            history["val_losses_all"].append(eval_pack["val_loss"])
            history["f1_sell"].append(eval_pack["f1_sell"])
            history["prec_sell"].append(eval_pack["prec_sell"])
            history["rec_sell"].append(eval_pack["rec_sell"])
            history["f1_buy"].append(eval_pack["f1_buy"])
            history["prec_buy"].append(eval_pack["prec_buy"])
            history["rec_buy"].append(eval_pack["rec_buy"])
            history["pnl"].append(eval_pack["profit"])

            if eval_pack["val_loss"] < history["best_val_loss"]:
                history["best_val_loss"] = eval_pack["val_loss"]
                history["best_epoch"] = epoch
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_tag = " <-- best"
                epochs_no_improve = 0
            else:
                best_tag = ""
                epochs_no_improve += 1

            logger.info(
                "Epoch %d | train_loss=%.6f val_loss=%.6f acc=%.4f profit=%.2f (SELL %.2f | BUY %.2f)%s",
                epoch,
                train_loss_epoch,
                eval_pack["val_loss"],
                eval_pack["acc"],
                eval_pack["profit"],
                eval_pack["profit_sell"],
                eval_pack["profit_buy"],
                best_tag,
            )

            if epochs_no_improve >= PATIENCE:
                logger.info(
                    "Early stopping: no val-loss improvement for %d epochs (patience=%d). "
                    "Best epoch was %d.",
                    epochs_no_improve, PATIENCE, history["best_epoch"],
                )
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user — saving best checkpoint so far (epoch %d).", history["best_epoch"])

    if best_model_state is None:
        logger.warning("No completed epoch checkpoint available — saving current model state.")
    else:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    final_eval = evaluate(model, data_pack["val_loader"], criterion, device)
    logger.info(
        "Best checkpoint restored | best_epoch=%d best_val_loss=%.6f final_eval_val_loss=%.6f",
        history["best_epoch"],
        history["best_val_loss"],
        final_eval["val_loss"],
    )

    run_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    today = run_ts[:8]  # kept for human-readable date_trained field
    ds_title = Path(DS_NAME).name.split(".")[0]
    model_name = "_".join([ds_title, model_type, f"{_seq_len}seq", run_ts, MODEL_VERSION])

    if history["val_losses_all"]:
        save_plots(model_name, history, final_eval)
    else:
        logger.warning("No epoch history to plot — skipping plot generation.")

    model_info = {
        "dataset_name": ds_title,
        "dataset_dir": DS_NAME,
        "date_trained": today,
        "model_type": model_type,
        "model_version": MODEL_VERSION,
        "task": "multiclass",
        "class_map": {0: "SELL", 1: "FLAT", 2: "BUY"},
        "seq_len": _seq_len,
        "class_weights": weights.tolist(),
        "n_epochs": NUM_EPOCHS,
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
    }

    loss_params_serializable = {**loss_params, "alpha": loss_params["alpha"].cpu().tolist()}

    model_pack_path = resume_model_pack_path if resume_pack else OUTPUT_DIR / f"{model_name}_model.pkl"
    with open(model_pack_path, "wb") as f:
        pickle.dump(
            {
                "model": model.state_dict(),
                "model_class": model_cls,
                "model_class_source": inspect.getsource(model_cls),
                "model_params": model_params_local,
                "model_info": model_info,
                "features": data_pack["features"],
                "feature_count": data_pack["X_train"].shape[1],
                "feature_function": _features_fn,
                "feature_function_source": inspect.getsource(_features_fn),
                "preprocess_function": preprocess_ohlcv,
                "preprocess_function_source": inspect.getsource(preprocess_ohlcv),
                "preprocess_args": data_pack["preprocess_ohlcv_args"],
                "scaler": data_pack["scaler"],
                "label_function": causal_triple_barrier_hilow_trend_labeler,
                "label_function_source": inspect.getsource(causal_triple_barrier_hilow_trend_labeler),
                "label_params": _label_params,
                "regime_params": _regime_params,
                "outcome_params": _outcome_params,
                "rollover_window": _rollover_window,
                "input_shape": (_seq_len, data_pack["X_train"].shape[1]),
                "loss_params": loss_params_serializable,
                "loss_function": TradeProfitabilityLoss,
                "loss_function_source": inspect.getsource(TradeProfitabilityLoss),
                "data_split": {
                    "split_method": "tail_val_bars",
                    "val_bars": VAL_BARS,
                    "train_rows": len(data_pack["X_train"]),
                    "val_rows": len(data_pack["X_val"]),
                    "train_end_time": str(df_train_raw["Time"].iloc[-1]),
                    "val_start_time": str(df_val_raw["Time"].iloc[0]),
                },
                "torch_version": torch.__version__,
                "val_metrics": {
                    "final_f1_sell": history["f1_sell"][-1],
                    "final_f1_buy": history["f1_buy"][-1],
                    "final_precision_sell": history["prec_sell"][-1],
                    "final_precision_buy": history["prec_buy"][-1],
                    "final_recall_sell": history["rec_sell"][-1],
                    "final_recall_buy": history["rec_buy"][-1],
                    "final_profit": history["pnl"][-1],
                    "best_f1_sell": max(history["f1_sell"]),
                    "best_f1_buy": max(history["f1_buy"]),
                    "best_precision_sell": max(history["prec_sell"]),
                    "best_precision_buy": max(history["prec_buy"]),
                    "f1_sell_curve": history["f1_sell"],
                    "f1_buy_curve": history["f1_buy"],
                    "precision_sell_curve": history["prec_sell"],
                    "precision_buy_curve": history["prec_buy"],
                    "recall_sell_curve": history["rec_sell"],
                    "recall_buy_curve": history["rec_buy"],
                    "pnl_curve": history["pnl"],
                },
            },
            f,
        )

    summary = {
        "model_name": model_name,
        "model_pack": str(model_pack_path),
        "plot_path": str(OUTPUT_DIR / f"{model_name}_plots.png"),
        "config": {
            "dataset": DS_NAME,
            "val_bars": VAL_BARS,
            "seq_len": _seq_len,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "rollover_window": ROLLOVER_WINDOW,
            "trading_hours": TRADING_HOURS,
            "commission": COMMISSION,
            "model_version": MODEL_VERSION,
        },
        "split": {
            "total_rows": len(df_all),
            "train_rows": len(df_train_raw),
            "val_rows": len(df_val_raw),
            "train_end_time": str(df_train_raw["Time"].iloc[-1]),
            "val_start_time": str(df_val_raw["Time"].iloc[0]),
        },
        "best": {
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
        },
        "curves": {
            "train_loss": history["train_losses"],
            "val_loss": history["val_losses_all"],
            "f1_sell": history["f1_sell"],
            "f1_buy": history["f1_buy"],
            "precision_sell": history["prec_sell"],
            "precision_buy": history["prec_buy"],
            "recall_sell": history["rec_sell"],
            "recall_buy": history["rec_buy"],
            "pnl": history["pnl"],
        },
    }

    summary_path = (
        Path(str(resume_model_pack_path).replace("_model.pkl", "_summary.json"))
        if resume_pack else OUTPUT_DIR / f"{model_name}_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved model pack: %s", model_pack_path)
    logger.info("Saved summary JSON: %s", summary_path)


if __name__ == "__main__":
    main()
