"""
TCN Multiclass Sweep Trainer
============================
Trains multiple TCNAttentionSEClassifier configurations sequentially on a single dataset,
saving a model pack to Engine/Model Packs/ after each run.

Usage (from ModelWorkbench/ directory):
    python train_tcn_sweep.py

To add or remove configs, edit SWEEP_CONFIGS below.
All other shared parameters (dataset, labels, loss, optimiser) are in the CONFIG section.
"""

import json
import math
import os
import pickle
import inspect
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from Learn.features import _add_features_US2000
from Learn.labels import (
    causal_triple_barrier_hilow_trend_labeler,
    calculate_trade_outcomes_all_candles,
)
from Learn.Loaders import SequenceDataset
from Learn.Loss import TradeProfitabilityLoss
from Learn.Models import TCNAttentionSEClassifier
from Learn.preprocess import preprocess_ohlcv

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — shared across all sweep runs
# ══════════════════════════════════════════════════════════════════════════════

# ── Dataset ───────────────────────────────────────────────────────────────────
DS_NAME    = '../data/US2000_M1_520weeks.csv'
TEST_START = '2026-04-01'
N_ROWS     = 1_000_000
FEATURES   = _add_features_US2000

# ── Sequence & batch ──────────────────────────────────────────────────────────
SEQ_LEN    = 256
BATCH_SIZE = 512

# ── Optimiser ─────────────────────────────────────────────────────────────────
NUM_EPOCHS   = 10
PATIENCE     = 5
BASE_LR      = 1.5e-4
WEIGHT_DECAY = 3e-4

# ── Preprocessing ─────────────────────────────────────────────────────────────
HOLDOUT_SIZE    = 0.20
ROLLOVER_WINDOW = ('21:30', '22:00')
TRADING_HOURS   = None

# ── Fees ──────────────────────────────────────────────────────────────────────
COMMISSION = 0.00

# ── Regime detection ──────────────────────────────────────────────────────────
regime_params = {
    'ma_period':           60,
    'slope_smoothness':    50,
    'regime_min_duration': 0,
    'atr_window':          60,
    'atr_lookback':        720,
    'atr_percentile':      0.0,
    'slope_threshold':     5e-6,
}

# ── Triple-barrier labelling ──────────────────────────────────────────────────
label_params = {
    'z_window':              14,
    'z_thresh':              1,
    'z_limit':               5,
    'atr_window':            14,
    'tp_mult':               2.5,
    'sl_mult':               2.0,
    'max_horizon':           90,
    'trend_pullback_thresh': 1.0,
    'regime_params':         regime_params,
    'skip_range':            True,
}

# ── Class weights & loss ──────────────────────────────────────────────────────
CLASS_WEIGHTS_RAW = np.array([1.0, 3.5, 1.0], dtype=float)  # [SELL, FLAT, BUY]

LOSS_PARAMS = {
    # alpha (class_weights tensor) is injected at runtime
    'gamma':             2.5,
    'trade_classes':     (0, 2),
    'pr_weight':         15.0,
    'recall_floor':      0.25,
    'rec_floor_weight':  80.0,
    'direction_penalty': 1.5,
    'eps':               1e-6,
}

# ══════════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGS
# Each entry must have:
#   version_tag        — appended to the saved model pack filename
#   trade_bias_offset  — log-prior penalty applied to SELL/BUY bias initialisation
#   model_params       — TCNAttentionSEClassifier kwargs (input_dim & bias_init injected automatically)
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_CONFIGS = [
    # ── v4 — v3 refined: higher bias offset to target precision ≥ 0.35 ───────
    # v3 was the sweep winner (+241 PnL at best epoch, lowest val loss 0.02574).
    # Raising trade_bias_offset 0.10→0.15 suppresses marginal trade signals,
    # pushing precision toward 0.35 while preserving the effective regularisation.
    {
        'version_tag':        'v4',
        'trade_bias_offset':  0.15,
        'model_params': {
            'hidden_channels':   128,
            'num_layers':        6,
            'kernel_size':       3,     # RF ≈ 127 bars
            'num_classes':       3,
            'dropout':           0.18,
            'dropout_out':       0.48,  # up from 0.45 — more output-head regularisation
            'attn_heads':        8,     # 128/8=16 dims/head
            'attn_dropout':      0.05,
            'use_learned_query': True,
        },
    },
    # ── v5 — v3 with full-seq receptive field (7 layers, RF=255) ─────────────
    # kernel=3, num_layers=7 gives RF=(3-1)×(2^7-1)+1=255, matching seq_len.
    # Tests whether covering the full 256-bar context improves signal quality.
    {
        'version_tag':        'v5',
        'trade_bias_offset':  0.10,
        'model_params': {
            'hidden_channels':   128,
            'num_layers':        7,     # RF ≈ 255 bars — covers full seq_len
            'kernel_size':       3,
            'num_classes':       3,
            'dropout':           0.18,
            'dropout_out':       0.45,
            'attn_heads':        8,
            'attn_dropout':      0.05,
            'use_learned_query': True,
        },
    },
    # ── v6 — intermediate width (160ch) with v3 regularisation ───────────────
    # v1 (192ch, 1.46M) showed no advantage over v3 (128ch, 666k). 160 channels
    # tests whether a moderate capacity increase adds value before committing
    # to 192ch, using the dropout profile proven effective in v3.
    {
        'version_tag':        'v6',
        'trade_bias_offset':  0.10,
        'model_params': {
            'hidden_channels':   160,   # between v3 (128) and v1 (192)
            'num_layers':        6,
            'kernel_size':       3,     # RF ≈ 127 bars
            'num_classes':       3,
            'dropout':           0.20,
            'dropout_out':       0.45,
            'attn_heads':        8,     # 160/8=20 dims/head
            'attn_dropout':      0.05,
            'use_learned_query': True,
        },
    },
    # ── v7 — high bias offset targeting the dominant failure mode ─────────────
    # avg_pnl_per_buy was negative in every epoch for every model in the sweep.
    # trade_bias_offset=0.25 heavily penalises SELL/BUY at initialisation,
    # forcing high-confidence-only predictions. Accepts lower recall for precision.
    {
        'version_tag':        'v7',
        'trade_bias_offset':  0.25,
        'model_params': {
            'hidden_channels':   128,
            'num_layers':        6,
            'kernel_size':       3,
            'num_classes':       3,
            'dropout':           0.18,
            'dropout_out':       0.45,
            'attn_heads':        8,
            'attn_dropout':      0.05,
            'use_learned_query': True,
        },
    },
    # ── v8 — 4 attention heads (untested dimension in this dataset) ───────────
    # All prior configs used 8 heads (128/8=16 dims/head). 4 heads gives
    # 128/4=32 dims/head — wider per-head representation that may better capture
    # longer-range momentum structure in US2000.
    {
        'version_tag':        'v8',
        'trade_bias_offset':  0.10,
        'model_params': {
            'hidden_channels':   128,
            'num_layers':        6,
            'kernel_size':       3,
            'num_classes':       3,
            'dropout':           0.18,
            'dropout_out':       0.45,
            'attn_heads':        4,     # 128/4=32 dims/head — untested in this sweep
            'attn_dropout':      0.05,
            'use_learned_query': True,
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE — runs once, shared across all sweep runs
# ══════════════════════════════════════════════════════════════════════════════

def build_data_pipeline(device):
    print('=' * 70)
    print('BUILDING DATA PIPELINE (runs once)')
    print('=' * 70)

    # Load
    df = pd.read_csv(DS_NAME)
    if N_ROWS is not None:
        df = df.tail(N_ROWS)
    df = df.sort_values('Time').reset_index(drop=True)

    test_start = df[df['Time'] > TEST_START].index[0]
    df_test = df.iloc[test_start:].copy().reset_index(drop=True)
    df      = df[df['Time'] < TEST_START].reset_index(drop=True)
    print(f'Train: {len(df):,} rows | Val: {len(df_test):,} rows')

    # Labels
    df_val = df_test.copy()
    for name, d in [('df', df), ('df_val', df_val)]:
        signals = causal_triple_barrier_hilow_trend_labeler(d, **label_params).rename(columns={'side': 'target'})
        winning = signals[signals['label'] == 1]
        d['target'] = 1
        d.loc[winning[winning['target'] ==  1].index, 'target'] = 2
        d.loc[winning[winning['target'] == -1].index, 'target'] = 0
        if not pd.api.types.is_datetime64_any_dtype(d['Time']):
            d['Time'] = pd.to_datetime(d['Time'])
        rollover = (
            (d['Time'].dt.time >= pd.to_datetime(ROLLOVER_WINDOW[0]).time()) &
            (d['Time'].dt.time <  pd.to_datetime(ROLLOVER_WINDOW[1]).time())
        )
        d.loc[rollover, 'target'] = 1
        print(f'{name}: {rollover.sum()} rollover rows zeroed | target dist:')
        print(d['target'].value_counts().sort_index().rename({0: 'SELL', 1: 'FLAT', 2: 'BUY'}), '\n')

    # Trade outcomes — binary only: 1 = TP hit, -1 = SL hit, NaN = unresolved (fillna'd to 0)
    outcome_params = {k: v for k, v in label_params.items()
                      if k in ['atr_window', 'tp_mult', 'sl_mult']}

    for name, d in [('df', df), ('df_val', df_val)]:
        outcomes = calculate_trade_outcomes_all_candles(d, **outcome_params)
        d['sell_y'] = outcomes['sell_outcome'].fillna(0.0)
        d['buy_y']  = outcomes['buy_outcome'].fillna(0.0)
        print(f'{name} outcomes | '
              f'Buy  TP:{(outcomes["buy_outcome"]  == 1).sum()} '
              f'SL:{(outcomes["buy_outcome"]  == -1).sum()} '
              f'Unresolved:{outcomes["buy_outcome"].isna().sum()} | '
              f'Sell TP:{(outcomes["sell_outcome"] == 1).sum()} '
              f'SL:{(outcomes["sell_outcome"] == -1).sum()} '
              f'Unresolved:{outcomes["sell_outcome"].isna().sum()}')

    # Features
    df     = FEATURES(df,     regime_params=regime_params)
    df_val = FEATURES(df_val, regime_params=regime_params)

    # Preprocess
    preprocess_ohlcv_args = {
        'target_col':      'target',
        'outcomes_col':    None,
        'shift':           0,
        'onehot_prefixes': ['OH_'],
        'price_prefixes':  ['PR_'],
    }
    split_idx = int(len(df) * (1 - HOLDOUT_SIZE))

    X_train, y_train, scaler, features, _, proc_df_train = preprocess_ohlcv(
        df.iloc[:split_idx].copy(), **preprocess_ohlcv_args, scaler=None, return_df=True)
    X_test, y_test, _, _, _, proc_df_test = preprocess_ohlcv(
        df.iloc[split_idx:].copy(), **preprocess_ohlcv_args, scaler=scaler, return_df=True)

    X_train = np.ascontiguousarray(X_train)
    X_test  = np.ascontiguousarray(X_test)
    y_train = [int(x) for x in y_train]
    y_test  = [int(x) for x in y_test]

    outcomes_train_2d = np.stack([proc_df_train['sell_y'].values,
                                   proc_df_train['buy_y'].values], axis=1).astype(float)
    outcomes_test_2d  = np.stack([proc_df_test['sell_y'].values,
                                   proc_df_test['buy_y'].values], axis=1).astype(float)

    # Session gating
    def _session_seq_indices(times, seq_len, trading_hours):
        if trading_hours is None:
            return list(range(len(times) - seq_len))
        t_start = pd.to_datetime(trading_hours[0]).time()
        t_end   = pd.to_datetime(trading_hours[1]).time()
        return [
            i for i in range(len(times) - seq_len)
            if t_start <= pd.Timestamp(times[i + seq_len - 1]).time() < t_end
        ]

    train_seq_idx = _session_seq_indices(proc_df_train['Time'].to_numpy(), SEQ_LEN, TRADING_HOURS)
    test_seq_idx  = _session_seq_indices(proc_df_test['Time'].to_numpy(),  SEQ_LEN, TRADING_HOURS)
    print(f'\nTrain: {X_train.shape} | Test: {X_test.shape}')
    print(f'Session-gated seqs -- Train: {len(train_seq_idx):,} | Test: {len(test_seq_idx):,}')

    # DataLoaders
    train_ds = SequenceDataset(X_train, y_train, seq_len=SEQ_LEN,
                               df_idx=list(range(len(X_train))),
                               custom_targets=None, trade_outcomes=outcomes_train_2d,
                               seq_idx_filter=train_seq_idx)
    val_ds   = SequenceDataset(X_test, y_test, seq_len=SEQ_LEN,
                               df_idx=list(range(len(X_test))),
                               custom_targets=None, trade_outcomes=outcomes_test_2d,
                               seq_idx_filter=test_seq_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=0)

    print(f'DataLoaders ready -- train batches: {len(train_loader):,} | val batches: {len(val_loader):,}\n')

    # Class weights tensor (shared, device-agnostic — cloned per run)
    weights_norm = CLASS_WEIGHTS_RAW / CLASS_WEIGHTS_RAW.mean()
    class_weights_cpu = torch.tensor(weights_norm, dtype=torch.float32)

    return dict(
        X_train=X_train, y_train=y_train,
        scaler=scaler, features=features,
        preprocess_ohlcv_args=preprocess_ohlcv_args,
        outcomes_train_2d=outcomes_train_2d,
        outcomes_test_2d=outcomes_test_2d,
        outcome_params=outcome_params,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights_cpu=class_weights_cpu,
        weights_norm=weights_norm,
        proc_df_train=proc_df_train,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION — one config
# ══════════════════════════════════════════════════════════════════════════════

def train_one_config(cfg, data, device):
    version_tag       = cfg['version_tag']
    trade_bias_offset = cfg.get('trade_bias_offset', 0.10)
    arch_params       = cfg['model_params']

    print(f'\n{"=" * 70}')
    print(f'TRAINING CONFIG: {version_tag}')
    print(f'  arch_params:       {arch_params}')
    print(f'  trade_bias_offset: {trade_bias_offset}')
    print(f'{"=" * 70}\n')

    X_train           = data['X_train']
    y_train           = data['y_train']
    train_loader      = data['train_loader']
    val_loader        = data['val_loader']
    class_weights_cpu = data['class_weights_cpu']
    weights_norm      = data['weights_norm']

    # bias_init
    y_train_arr = np.array(y_train)
    p_sell = (y_train_arr == 0).mean()
    p_flat = (y_train_arr == 1).mean()
    p_buy  = (y_train_arr == 2).mean()
    bias_init = [
        math.log(p_sell + 1e-8) - trade_bias_offset,
        math.log(p_flat + 1e-8),
        math.log(p_buy  + 1e-8) - trade_bias_offset,
    ]
    print(f'Class rates -- SELL: {p_sell:.4f}  FLAT: {p_flat:.4f}  BUY: {p_buy:.4f}')
    print(f'bias_init: {[round(b, 3) for b in bias_init]}')

    # Model
    model_params = {
        'input_dim': X_train.shape[1],
        'bias_init': bias_init,
        **arch_params,
    }
    model = TCNAttentionSEClassifier(**model_params).to(device)
    MODEL_TYPE = 'TCN_Multiclass'
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Loss & optimiser
    class_weights = class_weights_cpu.clone().to(device)
    loss_params = {**LOSS_PARAMS, 'alpha': class_weights}
    criterion   = TradeProfitabilityLoss(**loss_params)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    use_amp     = device.type == 'cuda'
    scaler_amp  = torch.amp.GradScaler('cuda', enabled=use_amp)

    warmup_steps = 400
    total_steps  = len(train_loader) * NUM_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Metric history
    f1_sell, prec_sell, rec_sell = [], [], []
    f1_buy,  prec_buy,  rec_buy  = [], [], []
    pnl = []
    train_loss_curve, val_loss_curve, acc_curve = [], [], []
    n_sell_curve, n_flat_curve, n_buy_curve     = [], [], []
    avg_pnl_sell_curve, avg_pnl_buy_curve       = [], []
    best_epoch_metrics = {}

    best_val_loss    = float('inf')
    best_model_state = None
    best_epoch       = -1
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb, outcome_b in tqdm(train_loader, desc=f'[{version_tag}] Epoch {epoch}'):
            xb, yb = xb.to(device), yb.to(device)
            outcome_b = outcome_b.to(device) if isinstance(outcome_b, torch.Tensor) else outcome_b
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(xb), yb, outcome_b)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_targets, all_outcomes = [], [], []
        with torch.no_grad():
            for xb, yb, outcome_b in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outcome_b = outcome_b.to(device) if isinstance(outcome_b, torch.Tensor) else outcome_b
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits    = model(xb)
                    val_loss += criterion(logits, yb, outcome_b).item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())
                all_outcomes.extend(
                    outcome_b.cpu().numpy() if isinstance(outcome_b, torch.Tensor) else outcome_b)

        # Metrics
        all_preds_np    = np.array(all_preds)
        all_outcomes_np = np.array(all_outcomes)
        sell_mask = (all_preds_np == 0)
        buy_mask  = (all_preds_np == 2)
        sell_net    = all_outcomes_np[:, 0] - COMMISSION * np.abs(all_outcomes_np[:, 0])
        buy_net     = all_outcomes_np[:, 1] - COMMISSION * np.abs(all_outcomes_np[:, 1])
        profit_sell = float(sell_net[sell_mask].sum()) if sell_mask.any() else 0.0
        profit_buy  = float(buy_net[buy_mask].sum())   if buy_mask.any()  else 0.0
        profit      = profit_sell + profit_buy

        val_loss_value = val_loss / len(all_targets)
        acc = accuracy_score(all_targets, all_preds)

        if val_loss_value < best_val_loss:
            best_val_loss    = val_loss_value
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch       = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        p_per, r_per, f_per, _ = precision_recall_fscore_support(
            all_targets, all_preds, labels=[0, 1, 2], average=None, zero_division=0)
        f1_sell.append(f_per[0]);  prec_sell.append(p_per[0]);  rec_sell.append(r_per[0])
        f1_buy.append(f_per[2]);   prec_buy.append(p_per[2]);   rec_buy.append(r_per[2])
        pnl.append(profit)
        train_loss_curve.append(train_loss / len(train_loader))
        val_loss_curve.append(val_loss_value)
        acc_curve.append(acc)
        n_sell_curve.append(int(sell_mask.sum()))
        n_flat_curve.append(int((all_preds_np == 1).sum()))
        n_buy_curve.append(int(buy_mask.sum()))
        avg_pnl_sell_curve.append(float(sell_net[sell_mask].mean()) if sell_mask.any() else 0.0)
        avg_pnl_buy_curve.append(float(buy_net[buy_mask].mean())   if buy_mask.any()  else 0.0)

        if epoch == best_epoch:
            best_epoch_metrics = {
                'accuracy':         acc,
                'f1_sell':          float(f_per[0]),   'f1_buy':          float(f_per[2]),
                'precision_sell':   float(p_per[0]),   'precision_buy':   float(p_per[2]),
                'recall_sell':      float(r_per[0]),   'recall_buy':      float(r_per[2]),
                'pnl':              profit,
                'pnl_sell':         profit_sell,       'pnl_buy':         profit_buy,
                'n_sell_trades':    int(sell_mask.sum()),
                'n_buy_trades':     int(buy_mask.sum()),
                'avg_pnl_per_sell': float(sell_net[sell_mask].mean()) if sell_mask.any() else 0.0,
                'avg_pnl_per_buy':  float(buy_net[buy_mask].mean())   if buy_mask.any()  else 0.0,
            }

        best_str = '  <- best' if epoch == best_epoch else f'  (best: {best_val_loss:.6f} @ epoch {best_epoch})'
        print(f'[{version_tag}] Epoch {epoch:>2} | Acc: {acc:.4f} | '
              f'Profit: {profit:.0f} [SELL: {profit_sell:.0f}  BUY: {profit_buy:.0f}] | '
              f'Val Loss: {val_loss_value:.6f}{best_str}')
        print(classification_report(all_targets, all_preds,
                                     target_names=['SELL', 'FLAT', 'BUY'], zero_division=0))

        if PATIENCE and epochs_no_improve >= PATIENCE:
            print(f'[{version_tag}] Early stopping at epoch {epoch} '
                  f'(no improvement for {PATIENCE} epochs)')
            break

    print(f'\n[{version_tag}] Training complete. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}')

    # Save model pack
    today      = pd.Timestamp.now().strftime('%Y%m%d')
    ds_title   = DS_NAME.split('/')[-1].split('.')[0]
    model_name = '_'.join([ds_title, MODEL_TYPE, str(SEQ_LEN) + 'seq', today, version_tag])

    model_info = {
        'dataset_name':  ds_title,
        'dataset_dir':   DS_NAME,
        'date_trained':  today,
        'model_type':    MODEL_TYPE,
        'task':          'multiclass',
        'class_map':     {0: 'SELL', 1: 'FLAT', 2: 'BUY'},
        'seq_len':       SEQ_LEN,
        'class_weights': weights_norm.tolist(),
        'n_epochs':      NUM_EPOCHS,
    }

    loss_params_serializable = {**loss_params, 'alpha': loss_params['alpha'].cpu().tolist()}

    out_path = os.path.join('..', 'Engine', 'Model Packs', model_name + '_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({
            # Model
            'model':              best_model_state,
            'model_class':        TCNAttentionSEClassifier,
            'model_class_source': inspect.getsource(TCNAttentionSEClassifier),
            'model_params':       model_params,
            'model_info':         model_info,
            # Features & Preprocessing
            'features':                   data['features'],
            'feature_count':              X_train.shape[1],
            'feature_function':           FEATURES,
            'feature_function_source':    inspect.getsource(FEATURES),
            'preprocess_function':        preprocess_ohlcv,
            'preprocess_function_source': inspect.getsource(preprocess_ohlcv),
            'preprocess_args':            data['preprocess_ohlcv_args'],
            'scaler':                     data['scaler'],
            # Labeling & Outcomes
            'label_function':          causal_triple_barrier_hilow_trend_labeler,
            'label_function_source':   inspect.getsource(causal_triple_barrier_hilow_trend_labeler),
            'label_params':            label_params,
            'regime_params':           regime_params,
            'outcome_params':          data['outcome_params'],
            'rollover_window':         ROLLOVER_WINDOW,
            'trading_hours':           TRADING_HOURS,
            # Training Config
            'input_shape':   (SEQ_LEN, X_train.shape[1]),
            'loss_params':   loss_params_serializable,
            'loss_function': TradeProfitabilityLoss,
            'loss_function_source': inspect.getsource(TradeProfitabilityLoss),
            'data_split': {
                'test_start_date': TEST_START,
                'train_rows':      len(X_train),
                'val_rows':        len(data['outcomes_test_2d']),
                'holdout_size':    HOLDOUT_SIZE,
            },
            'torch_version': torch.__version__,
            # Validation Metrics
            'val_metrics': {
                'final_f1_sell':        f1_sell[-1],   'final_f1_buy':        f1_buy[-1],
                'final_precision_sell': prec_sell[-1], 'final_precision_buy': prec_buy[-1],
                'final_recall_sell':    rec_sell[-1],  'final_recall_buy':    rec_buy[-1],
                'final_profit':         pnl[-1],
                'best_f1_sell':         max(f1_sell),  'best_f1_buy':         max(f1_buy),
                'best_precision_sell':  max(prec_sell),'best_precision_buy':  max(prec_buy),
                'f1_sell_curve':        f1_sell,        'f1_buy_curve':        f1_buy,
                'precision_sell_curve': prec_sell,      'precision_buy_curve': prec_buy,
                'recall_sell_curve':    rec_sell,        'recall_buy_curve':   rec_buy,
                'pnl_curve':            pnl,
            },
        }, f)

    print(f'[{version_tag}] Saved: {out_path}')

    summary = {
        'version_tag':     version_tag,
        'model_name':      model_name,
        'model_pack':      out_path,
        'param_count':     sum(p.numel() for p in model.parameters()),
        'training_config': {
            'dataset':        DS_NAME,
            'seq_len':        SEQ_LEN,
            'batch_size':     BATCH_SIZE,
            'epochs':         NUM_EPOCHS,
            'base_lr':        BASE_LR,
            'weight_decay':   WEIGHT_DECAY,
            'holdout_size':   HOLDOUT_SIZE,
            'commission':     COMMISSION,
            'rollover_window': list(ROLLOVER_WINDOW),
            'trading_hours':  TRADING_HOURS,
        },
        'architecture':    arch_params,
        'bias_init':       bias_init,
        'class_weights':   weights_norm.tolist(),
        'label_params':    {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in label_params.items()},
        'loss_params':     {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in LOSS_PARAMS.items()},
        'data_split': {
            'test_start_date': TEST_START,
            'train_rows':      int(len(X_train)),
            'val_rows':        int(len(data['outcomes_test_2d'])),
            'n_train_seqs':    len(data['train_loader'].dataset),
            'n_val_seqs':      len(data['val_loader'].dataset),
        },
        'best': {
            'epoch':    best_epoch,
            'val_loss': best_val_loss,
            'metrics':  best_epoch_metrics,
        },
        'final_epoch_metrics': {
            'accuracy':       acc_curve[-1],
            'f1_sell':        f1_sell[-1],    'f1_buy':        f1_buy[-1],
            'precision_sell': prec_sell[-1],  'precision_buy': prec_buy[-1],
            'recall_sell':    rec_sell[-1],   'recall_buy':    rec_buy[-1],
            'pnl':            pnl[-1],
            'n_sell_trades':  n_sell_curve[-1],
            'n_buy_trades':   n_buy_curve[-1],
        },
        'curves': {
            'train_loss':       train_loss_curve,
            'val_loss':         val_loss_curve,
            'accuracy':         acc_curve,
            'f1_sell':          f1_sell,        'f1_buy':        f1_buy,
            'precision_sell':   prec_sell,      'precision_buy': prec_buy,
            'recall_sell':      rec_sell,       'recall_buy':    rec_buy,
            'pnl':              pnl,
            'pred_dist': {
                'SELL': n_sell_curve,
                'FLAT': n_flat_curve,
                'BUY':  n_buy_curve,
            },
            'n_sell_trades':    n_sell_curve,
            'n_buy_trades':     n_buy_curve,
            'avg_pnl_per_sell': avg_pnl_sell_curve,
            'avg_pnl_per_buy':  avg_pnl_buy_curve,
        },
    }

    summary_path = out_path.replace('_model.pkl', '_sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[{version_tag}] Summary: {summary_path}\n')

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f'Device: {device}')
    print(f'Sweep: {len(SWEEP_CONFIGS)} configs — {[c["version_tag"] for c in SWEEP_CONFIGS]}\n')

    data = build_data_pipeline(device)

    results = []
    for i, cfg in enumerate(SWEEP_CONFIGS):
        print(f'\n[{i + 1}/{len(SWEEP_CONFIGS)}] Starting config: {cfg["version_tag"]}')
        try:
            summary = train_one_config(cfg, data, device)
            results.append({'version_tag': cfg['version_tag'], 'status': 'OK', 'summary': summary})
        except Exception as e:
            print(f'\n[ERROR] Config {cfg["version_tag"]} failed: {e}')
            traceback.print_exc()
            results.append({'version_tag': cfg['version_tag'], 'status': f'FAILED: {e}'})

    today    = pd.Timestamp.now().strftime('%Y%m%d')
    ds_title = DS_NAME.split('/')[-1].split('.')[0]
    sweep_comparison = {
        'sweep_date':    pd.Timestamp.now().isoformat(),
        'dataset':       DS_NAME,
        'shared_config': {
            'seq_len':       SEQ_LEN,
            'batch_size':    BATCH_SIZE,
            'epochs':        NUM_EPOCHS,
            'base_lr':       BASE_LR,
            'weight_decay':  WEIGHT_DECAY,
            'label_params':  {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in label_params.items()},
            'loss_params':   {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in LOSS_PARAMS.items()},
            'class_weights': CLASS_WEIGHTS_RAW.tolist(),
        },
        'configs': [
            {'version_tag': c['version_tag'],
             'trade_bias_offset': c.get('trade_bias_offset', 0.10),
             **c['model_params']}
            for c in SWEEP_CONFIGS
        ],
        'results': [
            {'version_tag': r['version_tag'],
             'status':      r['status'],
             'summary':     r.get('summary')}
            for r in results
        ],
    }
    sweep_path = os.path.join('..', 'Engine', 'Model Packs',
                              f'{ds_title}_TCN_sweep_{today}_comparison.json')
    with open(sweep_path, 'w') as f:
        json.dump(sweep_comparison, f, indent=2)
    print(f'\nSweep comparison saved: {sweep_path}')

    print('\n' + '=' * 70)
    print('SWEEP COMPLETE')
    print('=' * 70)
    for r in results:
        print(f'  {r["version_tag"]:12s}  {r["status"]}')
