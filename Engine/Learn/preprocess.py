import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from talib import EMA

from collections import Counter

def preprocess_ohlcv(
        df, 
        target_col=None,
        outcomes_col=None,
        shift=0, 
        onehot_prefixes=['OH_'], 
        price_prefixes=['PR_'], 
        vol_window=20, 
        scaler=None, 
        return_df=False):
    """
    Preprocess OHLCV dataframe for ML models (classification/regression).
    
    Features included:
      - log returns
      - relative OHLC (normalized by Close)
      - volatility-scaled returns
      - engineered features (continuous scaled, one-hot untouched)
    """
    
    df = df.copy()

    # Initial drop of NaNs to avoid scaling issues from partially-computed features
    df = df.dropna().reset_index(drop=True)

    # Outcomes (if provided) — we'll realign after target shift and final dropna
    outcomes = df[outcomes_col] if outcomes_col in df.columns and outcomes_col is not None else None

    # Detect One-Hot by prefix and price-based features by prefix
    if onehot_prefixes is None:
        onehot_prefixes = []
    onehot_features = [c for c in df.columns if any(c.startswith(p) for p in onehot_prefixes)]

    if price_prefixes is None:
        price_prefixes = []
    price_features = [c for c in df.columns if any(c.startswith(p) for p in price_prefixes)]

    # Normalize price-based features by Close (relative context)
    for c in price_features:
        if 'Close' in df.columns:
            df[c] = (df[c] - df['Close']) / df['Close']

    # Base features that we want scaled (except constant C_rel)
    base_features_scale = ['log_return','O_rel','H_rel','L_rel']

    # Dynamically detect binary/ternary flags that should NOT be scaled
    # Treat columns with only {0,1} or {-1,0,1} as categorical pass-through
    def _is_binary_or_ternary(col):
        s = df[col]
        # Skip non-numeric columns (e.g., lists like outcomes); only numeric can be binary/ternary
        if not pd.api.types.is_numeric_dtype(s):
            return False
        vals = s.dropna().unique()
        if len(vals) == 0:
            return False
        # If values are unhashable (e.g., arrays), treat as non-binary
        try:
            set_vals = set(vals.tolist())
        except TypeError:
            return False
        return set_vals.issubset({0, 1}) or set_vals.issubset({-1, 0, 1})

    binary_like_features = [c for c in df.columns if _is_binary_or_ternary(c)]

    # Known normalized/bounded features to pass through without additional scaling
    normalized_prefixes = ['z_', 'price_loc_']
    normalized_suffixes = ['_norm']
    normalized_names = set([
        'RSI','MFI','ADX','WilliamsR','StochK','StochD',
        'AroonUp','AroonDown','AroonOsc','DI_plus_14','DI_minus_14','DI_diff_14',
        'bb_pos','volume_z','ret_vol_scaled','trend_alignment','mtf_avg_adx'
    ])

    def _is_normalized_feature(name):
        if name in normalized_names:
            return True
        if any(name.startswith(p) for p in normalized_prefixes):
            return True
        if any(name.endswith(suf) for suf in normalized_suffixes):
            return True
        return False

    normalized_passthrough = [c for c in df.columns if _is_normalized_feature(c)]

    # Exclude non-feature columns and constant columns from scaling group
    exclude = set(['Time','Open','High','Low','Close','Volume','Pivot','target','vol','outcomes','C_rel', 'sell_y', 'buy_y']) # 

    # Also exclude any provided label/outcome columns from ALL feature groups
    label_exclude = set(['Pivot','target', 'sell_y', 'buy_y'])
    if target_col is not None:
        label_exclude.add(target_col)
    if outcomes_col is not None:
        label_exclude.add(outcomes_col)

    # Continuous features considered for scaling: everything not excluded, not one-hot prefix, not binary-like, not normalized pass-through, and not in base_features_scale
    cont_candidates = [c for c in df.columns if c not in exclude and c not in label_exclude]
    cont_scale_features = [
        c for c in cont_candidates
        if c not in onehot_features
        and c not in binary_like_features
        and c not in normalized_passthrough
        and c not in base_features_scale
    ]

    # Build final column groups
    scale_cols = [c for c in (base_features_scale + cont_scale_features) if c not in label_exclude]
    # Combine explicit one-hot prefix features with dynamically detected binary-like features
    onehot_all = sorted((set(onehot_features) | set(binary_like_features)) - label_exclude)
    passthrough_cols = [c for c in normalized_passthrough if c not in label_exclude]

    # Fit/transform scaler on scale_cols only
    if scale_cols:
        if scaler is None:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(df[scale_cols].values)
        else:
            X_scaled = scaler.transform(df[scale_cols].values)
    else:
        X_scaled = np.empty((len(df), 0))

    X_parts = [X_scaled]
    feature_cols = []
    feature_cols.extend(scale_cols)

    # Add normalized/bounded passthrough features
    if passthrough_cols:
        X_parts.append(df[passthrough_cols].values)
        feature_cols.extend(passthrough_cols)

    # Add one-hot/binary-like features without scaling
    if onehot_all:
        X_parts.append(df[onehot_all].values)
        feature_cols.extend(onehot_all)

    # Concatenate all parts
    X = np.hstack([p for p in X_parts if p.shape[1] > 0])

    # Targets (next candle Pivot)
    if target_col is not None:
        df['target'] = df[target_col].shift(shift)
    else:
        y = None

    # Final alignment after shifting target and recomputing valid rows
    df = df.dropna().reset_index(drop=True)

    # Align X with the shifted target: drop the first `shift` rows from X
    if shift > 0 and len(X) >= shift:
        X = X[shift:,:]

    # Build y
    if target_col is not None:
        y = df['target'].values
    else:
        y = None

    # Realign outcomes (if provided) to match df's current index
    aligned_outcomes = None
    if outcomes is not None:
        # Recompute from the current df state to ensure perfect alignment
        aligned_outcomes = df[outcomes_col].values if outcomes_col in df.columns else None

    if return_df:
        proc_df = df.copy()
        proc_df = proc_df.reset_index(drop=True)
        return X, y, scaler, feature_cols, aligned_outcomes, proc_df

    return X, y, scaler, feature_cols

def _preprocess_ohlcv(df, target_col=None, shift=0, onehot_prefixes=['OH_'], price_prefixes=['PR_'], vol_window=20, scaler=None, return_df=False):
    """
    Preprocess OHLCV dataframe for ML models (classification/regression).
    
    Features included:
      - log returns
      - relative OHLC (normalized by Close)
      - volatility-scaled returns
      - engineered features (continuous scaled, one-hot untouched)
    """
    
    df = df.copy()

    ema = EMA(df['Close'], timeperiod=20)
    
    # --- 1. Compute log returns ---
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # --- 2. Relative OHLC features (normalize by Close) ---
    df['O_rel'] = (df['Open'] - df['Close']) / df['Close']
    df['H_rel'] = (df['High'] - df['Close']) / df['Close']
    df['L_rel'] = (df['Low'] - df['Close']) / df['Close']
    df['C_rel'] = 0.0  # always baseline
    
    # --- 3. Volatility scaling (rolling std of returns) ---
    df['vol'] = df['log_return'].rolling(vol_window).std()
    df['ret_vol_scaled'] = df['log_return'] / df['vol']

    df['O_ema'] = df['Open'] - ema
    df['H_ema'] = df['High'] - ema
    df['L_ema'] = df['Low'] - ema
    df['C_ema'] = df['Close'] - ema

    # --- 4. Drop NaNs from rolling ---
    df = df.dropna().reset_index(drop=True)
    
    # --- 5. Split engineered features ---
    if onehot_prefixes is None:
        onehot_prefixes = []  # user-defined list of col prefixes

    if price_prefixes is None:
        price_prefixes = []
    
    # detect one-hot features
    onehot_features = [c for c in df.columns 
                       if any(c.startswith(p) for p in onehot_prefixes)]
    
    # detect price-based features
    price_features = [c for c in df.columns 
                      if any(c.startswith(p) for p in price_prefixes)]
    
    # Convert to relative changes with respect to Close
    for c in price_features:
        df[c] = (df[c] - df['Close']) / df['Close']  # normalize by Close
    
    # detect continuous engineered features (everything else except OHLCV + labels)
    exclude = set(['Time','Open','High','Low','Close','Volume','Pivot','target','vol'])
    base_features = ['log_return','O_rel','H_rel','L_rel','ret_vol_scaled']
    cont_features = [c for c in df.columns if c not in exclude 
                     and c not in onehot_features
                    #  and c not in price_features
                     and c not in base_features]
    
    # --- 6. Scale continuous features ---
    # minmax_scaler = MinMaxScaler()
    # X_cont_minmax = minmax_scaler.fit_transform(df[base_features + cont_features].values)
    if scaler is None:
        scaler = StandardScaler()
        # X_cont = scaler.fit_transform(X_cont_minmax)
        X_cont = scaler.fit_transform(df[base_features + cont_features].values)
    else:
        X_cont = scaler.transform(df[base_features + cont_features].values)
    
    # --- 7. Concatenate with one-hot features (no scaling) ---
    if onehot_features:
        X_onehot = df[onehot_features].values
        X = np.hstack([X_cont, X_onehot])
    else:
        X = X_cont
    
    # --- 8. Targets (next candle Pivot) ---
    if target_col is not None:
        df['target'] = df[target_col].shift(shift)
    else:
        y = None
    
    df = df.dropna().reset_index(drop=True)
    X = X[shift:,:]  # align with shifted target

    if target_col is not None:
        y = df['target'].values
    else:
        y = None
    
    
    feature_cols = (base_features + cont_features + onehot_features)

    if return_df:
        # Build the processed dataframe aligned with X and y so callers can map back to prices
        proc_df = df.copy()
        # proc_df at this point has been shifted and dropna() applied in the function
        # ensure index is reset and aligned with X rows
        proc_df = proc_df.reset_index(drop=True)
        return X, y, scaler, feature_cols, proc_df

    return X, y, scaler, feature_cols

def oversample_sequences(X_seq, y_seq, multiplier=1.0, custom_targets=None):
    """
    Oversample sequences to balance classes.
    - multiplier: float, 1.0 = balance to max class, <1.0 = less oversampling
    - custom_targets: dict {class_label: target_count}, overrides multiplier
    """
    from collections import Counter
    counts = Counter(y_seq)
    max_count = max(counts.values())
    X_balanced, y_balanced = [], []
    for cls in counts:
        idxs = np.where(y_seq == cls)[0]
        if custom_targets and cls in custom_targets:
            n_target = custom_targets[cls]
        else:
            n_target = int(max_count * multiplier)
        n_to_add = n_target - len(idxs)
        if n_to_add > 0:
            add_idxs = np.random.choice(idxs, n_to_add, replace=True)
            X_balanced.append(X_seq[add_idxs])
            y_balanced.append(y_seq[add_idxs])
        X_balanced.append(X_seq[idxs])
        y_balanced.append(y_seq[idxs])
    X_bal = np.concatenate(X_balanced)
    y_bal = np.concatenate(y_balanced)
    p = np.random.permutation(len(y_bal))
    return X_bal[p], y_bal[p]

def filter_signals_profit(ohlc, pivot_col='Pivot', target_col='target', signal_lookback=1):

    df = ohlc.copy()
    df['exit_type'] = 'none'
    df['exit_idx'] = pd.NA
    df['exit_price'] = pd.NA
    df['exit_time'] = pd.NaT
    df['exit_steps'] = pd.NA

    pivot_indices = df.index[df[target_col] != 0].tolist()

    for p in pivot_indices:
        s = df.at[p, 'target']
        tp = df.at[p, 'tp']
        sl = df.at[p, 'sl']

        for _, i in df.iloc[p:,:].head(60).iterrows():
            
            if s == 1:
                if i['High'] >= sl:
                    df.at[p, 'exit_type'] = 'sl'
                    df.at[p, 'exit_idx'] = _
                    df.at[p, 'exit_price'] = sl
                    df.at[p, 'exit_time'] = i['Time']
                    df.at[p, 'exit_steps'] = _ - p
                    break
                elif i['Low'] <= tp:
                    df.at[p, 'exit_type'] = 'tp'
                    df.at[p, 'exit_idx'] = _
                    df.at[p, 'exit_price'] = tp
                    df.at[p, 'exit_time'] = i['Time']
                    df.at[p, 'exit_steps'] = _ - p
                    break
            elif s == -1:
                if i['Low'] <= sl:
                    df.at[p, 'exit_type'] = 'sl'
                    df.at[p, 'exit_idx'] = _
                    df.at[p, 'exit_price'] = sl
                    df.at[p, 'exit_time'] = i['Time']
                    df.at[p, 'exit_steps'] = _ - p
                    break
                elif i['High'] >= tp:
                    df.at[p, 'exit_type'] = 'tp'
                    df.at[p, 'exit_idx'] = _
                    df.at[p, 'exit_price'] = tp
                    df.at[p, 'exit_time'] = i['Time']
                    df.at[p, 'exit_steps'] = _ - p
                    break

    df['filtered_target'] = [x if y == 'tp' else 0 for x, y in zip(df[target_col], df['exit_type'])]
    df[pivot_col] = df['filtered_target'].shift(-signal_lookback)
    ohlc[pivot_col] = df[pivot_col]
    return df

def filter_signals_ema(ohlc, pivot_col='Pivot', target_col='target', signal_lookback=1, ema1=8, ema2=30):

    df = ohlc.copy()
    df['ema1'] = EMA(df['Close'], timeperiod=ema1)
    df['ema2'] = EMA(df['Close'], timeperiod=ema2)

    pivot_indices = df.index[df[pivot_col] != 0].tolist()

    for p in pivot_indices:
        row = df.iloc[p]
        t = row[pivot_col]
        e1 = row['ema1']
        e2 = row['ema2']

        if t == 1 and e1 < e2:
            df.at[p, pivot_col] = 0
        elif t == -1 and e1 > e2:
            df.at[p, pivot_col] = 0
    
    return df