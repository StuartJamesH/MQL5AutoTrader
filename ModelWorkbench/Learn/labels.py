import numpy as np
import pandas as pd
from numba import njit
from talib import ATR, EMA

# DEPRECATED
# def atr_filter(df, atr_window=28, atr_threshold=40.0, cooldown=5):
#     """
#     Returns a boolean Series where True indicates rows with ATR above the threshold.
#     """
#     df = df.copy()
#     df['atr'] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
#     df['atr_bool'] = [1 if x > atr_threshold else 0 for x in df['atr']]
#     df['atr_filter'] = df['atr_bool'].rolling(window=cooldown).max().fillna(0).astype(int)
#     return df['atr_filter']


# DEPRECATED
# def triple_barrier_hilow_labeler(
#     df,
#     z_window=30,
#     z_thresh=2.5,
#     z_limit=3.0,
#     atr_window=14,
#     tp_mult=1.5,
#     sl_mult=1.0,
#     max_horizon=20
# ):
#     """
#     df: pandas DataFrame with columns ['open','high','low','close']
#     z_window: window for rolling mean/std for z-score
#     z_thresh: threshold for mean-reversion signals
#     atr_window: ATR window
#     tp_mult/sl_mult: volatility-scaled barrier strength
#     max_horizon: vertical barrier (bars)
#     """
# 
#     # ----------------------------------------------------------------------
#     # 1. Compute z-score for mean reversion signals
#     # ----------------------------------------------------------------------
#     df = df.copy()
# 
#     df["mean"] = df["Close"].rolling(z_window).mean()
#     df["std"]  = df["Close"].rolling(z_window).std()
#     df["z"] = (df["Close"] - df["mean"]) / df["std"]
# 
#     # signal: +1 = long, -1 = short
#     signals = pd.Series(index=df.index, dtype=float)
#     signals[(df["z"] < -z_thresh) & (df["z"] > -z_limit)] = +1   # oversold → long
#     signals[(df["z"] > +z_thresh) & (df["z"] < +z_limit)] = -1   # overbought → short
#     signals = signals.dropna()
# 
#     # ----------------------------------------------------------------------
#     # 2. Compute ATR for volatility-scaled barriers
#     # ----------------------------------------------------------------------
#     df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
# 
#     # ----------------------------------------------------------------------
#     # 3. Triple-barrier event labeling
#     # ----------------------------------------------------------------------
#     events = []
# 
#     for t0, side in signals.items():
#         if t0 >= len(df) - 1:
#             continue  # avoid last-row issues
# 
#         # Use candle High/Low as the entry price for hilow labeler:
#         #  - for long signals (+1) use the candle Low as a conservative entry
#         #  - for short signals (-1) use the candle High as a conservative entry
#         if side == +1:
#             entry_price = df.loc[t0, "Low"]
#         else:
#             entry_price = df.loc[t0, "High"]
#         atr = df.loc[t0, "atr"]
# 
#         if np.isnan(atr) or atr == 0:
#             continue  # skip events without stable volatility estimates
# 
#         # compute profit and stop barriers
#         if side == +1:  # long
#             tp = entry_price + tp_mult * atr
#             sl = entry_price - sl_mult * atr
#         else:           # short
#             tp = entry_price - tp_mult * atr
#             sl = entry_price + sl_mult * atr
# 
#         t_end = min(t0 + max_horizon, df.index[-1])
#         label = 0       # default = vertical barrier (neutral)
#         end_time = t_end
# 
#         # scan forward through the window
#         for t in range(t0 + 1, t_end + 1):
#             high = df.loc[t, "High"]
#             low  = df.loc[t, "Low"]
# 
#             if side == +1:  # long
#                 if high >= tp:
#                     label = +1
#                     end_time = t
#                     break
#                 if low <= sl:
#                     label = -1
#                     end_time = t
#                     break
#             else:           # short
#                 if low <= tp:
#                     label = +1
#                     end_time = t
#                     break
#                 if high >= sl:
#                     label = -1
#                     end_time = t
#                     break
# 
#         events.append({
#             "t0": t0,
#             "side": side,
#             "z": df.loc[t0, "z"],
#             "tp": tp,
#             "sl": sl,
#             "t_end": end_time,
#             "label": label
#         })
# 
#     return pd.DataFrame(events).set_index("t0")

# DEPRECATED
# def triple_barrier_labeler(
#     df,
#     z_window=30,
#     z_thresh=2.5,
#     z_limit=3.0,
#     atr_window=14,
#     tp_mult=1.5,
#     sl_mult=1.0,
#     max_horizon=20
# ):
#     """
#     df: pandas DataFrame with columns ['open','high','low','close']
#     z_window: window for rolling mean/std for z-score
#     z_thresh: threshold for mean-reversion signals
#     atr_window: ATR window
#     tp_mult/sl_mult: volatility-scaled barrier strength
#     max_horizon: vertical barrier (bars)
#     """
# 
#     # ----------------------------------------------------------------------
#     # 1. Compute z-score for mean reversion signals
#     # ----------------------------------------------------------------------
#     df = df.copy()
# 
#     df["mean"] = df["Close"].rolling(z_window).mean()
#     df["std"]  = df["Close"].rolling(z_window).std()
#     df["z"] = (df["Close"] - df["mean"]) / df["std"]
# 
#     # signal: +1 = long, -1 = short
#     signals = pd.Series(index=df.index, dtype=float)
#     signals[(df["z"] < -z_thresh) & (df["z"] > -z_limit)] = +1   # oversold → long
#     signals[(df["z"] > +z_thresh) & (df["z"] < +z_limit)] = -1   # overbought → short
#     signals = signals.dropna()
# 
#     # ----------------------------------------------------------------------
#     # 2. Compute ATR for volatility-scaled barriers
#     # ----------------------------------------------------------------------
#     df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
# 
#     # ----------------------------------------------------------------------
#     # 3. Triple-barrier event labeling
#     # ----------------------------------------------------------------------
#     events = []
# 
#     for t0, side in signals.items():
#         if t0 >= len(df) - 1:
#             continue  # avoid last-row issues
# 
#         entry_price = df.loc[t0, "Close"]
#         atr = df.loc[t0, "atr"]
# 
#         if np.isnan(atr) or atr == 0:
#             continue  # skip events without stable volatility estimates
# 
#         # compute profit and stop barriers
#         if side == +1:  # long
#             tp = entry_price + tp_mult * atr
#             sl = entry_price - sl_mult * atr
#         else:           # short
#             tp = entry_price - tp_mult * atr
#             sl = entry_price + sl_mult * atr
# 
#         t_end = min(t0 + max_horizon, df.index[-1])
#         label = 0       # default = vertical barrier (neutral)
#         end_time = t_end
# 
#         # scan forward through the window
#         for t in range(t0 + 1, t_end + 1):
#             high = df.loc[t, "High"]
#             low  = df.loc[t, "Low"]
# 
#             if side == +1:  # long
#                 if high >= tp:
#                     label = +1
#                     end_time = t
#                     break
#                 if low <= sl:
#                     label = -1
#                     end_time = t
#                     break
#             else:           # short
#                 if low <= tp:
#                     label = +1
#                     end_time = t
#                     break
#                 if high >= sl:
#                     label = -1
#                     end_time = t
#                     break
# 
#         events.append({
#             "t0": t0,
#             "side": side,
#             "z": df.loc[t0, "z"],
#             "tp": tp,
#             "sl": sl,
#             "t_end": end_time,
#             "label": label
#         })
# 
#     return pd.DataFrame(events).set_index("t0")

# DEPRECATED
# def triple_barrier_outcome(
#     df,
#     z_window=30,
#     z_thresh=2.5, # Keep so we can use the same **params dict
#     atr_window=14,
#     tp_mult=1.5,
#     sl_mult=1.0,
#     max_horizon=20
# ):
#     """
#     df: pandas DataFrame with columns ['open','high','low','close']
#     z_window: window for rolling mean/std for z-score
#     z_thresh: threshold for mean-reversion signals
#     atr_window: ATR window
#     tp_mult/sl_mult: volatility-scaled barrier strength
#     max_horizon: vertical barrier (bars)
#     """
# 
#     # ----------------------------------------------------------------------
#     # 1. Compute z-score for mean reversion signals
#     # ----------------------------------------------------------------------
#     df = df.copy()
# 
#     df["mean"] = df["Close"].rolling(z_window).mean()
#     df["std"]  = df["Close"].rolling(z_window).std()
#     df["z"] = (df["Close"] - df["mean"]) / df["std"]
# 
#     # # init signals
#     signals = pd.Series(index=df.index, dtype=float)
# 
#     # ----------------------------------------------------------------------
#     # 2. Compute ATR for volatility-scaled barriers
#     # ----------------------------------------------------------------------
#     df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
# 
#     # ----------------------------------------------------------------------
#     # 3. Create barriers
#     # ----------------------------------------------------------------------
#     events = []
# 
#     for _side in [1, -1]:
#         signals[:] = _side
#         for t0, side in signals.items():
#             if t0 >= len(df) - 1:
#                 continue  # avoid last-row issues
# 
#             entry_price = df.loc[t0, "Close"]
#             atr = df.loc[t0, "atr"]
# 
#             if np.isnan(atr) or atr == 0:
#                 continue  # skip events without stable volatility estimates
# 
#             # compute profit and stop barriers
#             if side == +1:  # long
#                 tp = entry_price + tp_mult * atr
#                 sl = entry_price - sl_mult * atr
#             else:           # short
#                 tp = entry_price - tp_mult * atr
#                 sl = entry_price + sl_mult * atr
# 
#             t_end = min(t0 + max_horizon, df.index[-1])
#             exit_price = df.loc[t_end, "Close"]       # default = vertical barrier (neutral)
#             end_time = t_end
# 
#             # scan forward through the window
#             for t in range(t0 + 1, t_end + 1):
#                 high = df.loc[t, "High"]
#                 low  = df.loc[t, "Low"]
# 
#                 if side == +1:  # long
#                     if high >= tp:
#                         exit_price = tp
#                         # profit = exit_price - entry_price
#                         end_time = t
#                         break
#                     if low <= sl:
#                         exit_price = sl
#                         # profit = exit_price - entry_price
#                         end_time = t
#                         break
#                 else:           # short
#                     if low <= tp:
#                         exit_price = tp
#                         # profit = entry_price - exit_price
#                         end_time = t
#                         break
#                     if high >= sl:
#                         exit_price = sl
#                         # profit = entry_price - exit_price
#                         end_time = t
#                         break
# 
#             events.append({
#                 "t0": t0,
#                 "side": side,
#                 "t_end": end_time,
#                 "entry_price": entry_price,
#                 "exit_price": exit_price,
#             })
# 
#     return pd.DataFrame(events).set_index("t0")

# DEPRECATED
# def get_market_regime(df, ma_period=80, slope_smoothness=15, range_proportion=40, regime_min_duration=30):
#     """
#     Calculates market regime based on Centered Moving Average slope.
#     Returns a Series with values: 1 (Uptrend), 0 (Range), -1 (Downtrend)
#     """
#     df = df.copy()
#     
#     # 1. Centered Moving Average (Look-ahead)
#     df['Centered_MA'] = df['Close'].rolling(window=ma_period).mean().shift(-int(ma_period/2))
#     
#     # 2. Slope with Smoothing
#     df['Regime_Slope'] = df['Centered_MA'].diff().rolling(window=slope_smoothness).mean()
#     
#     # 3. Thresholding (Percentile)
#     slope_abs = df['Regime_Slope'].abs()
#     slope_threshold = np.percentile(slope_abs.dropna(), range_proportion)
#     
#     df['Regime'] = np.where(df['Regime_Slope'] > slope_threshold, 1,
#                    np.where(df['Regime_Slope'] < -slope_threshold, -1,
#                    0))
#     
#     # 4. Filter short-lived regimes
#     block_ids = (df['Regime'] != df['Regime'].shift()).cumsum()
#     block_sizes = df.groupby(block_ids)['Regime'].transform('count')
#     df.loc[block_sizes < regime_min_duration, 'Regime'] = np.nan
#     df['Regime'] = df['Regime'].ffill().bfill()
#     
#     return df['Regime']

def super_smoother(series, period):
    """
    Ehlers Super Smoother (2-pole IIR) - robust to NaNs.
    """
    import numpy as np

    if period <= 0:
        raise ValueError("period must be > 0")

    # Protect against NaNs by forward/back filling for calculation
    s = series.astype(float).ffill().bfill().to_numpy()

    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    filt = np.zeros_like(s, dtype=float)
    if len(s) == 0:
        return pd.Series(filt, index=series.index)

    filt[0] = s[0]
    if len(s) > 1:
        filt[1] = c1 * (s[1] + s[0]) / 2 + c2 * filt[0]

    for i in range(2, len(s)):
        filt[i] = (
            c1 * (s[i] + s[i - 1]) / 2
            + c2 * filt[i - 1]
            + c3 * filt[i - 2]
        )

    return pd.Series(filt, index=series.index)

def causal_market_regime(df, ma_period=21, slope_smoothness=1, regime_min_duration=1,
                         slope_threshold=0, atr_window=14, atr_lookback=100, atr_percentile=20,
                         slope_lookback=200, slope_percentile=30):
    """
    Calculates market regime using purely causal (trailing) indicators.
    Returns a Series with values: 1 (Uptrend), 0 (Range), -1 (Downtrend)

    Differences from get_market_regime:
      - Uses trailing SMA of Close (no centered shift)
      - Uses trailing smoothed slope via rolling mean
      - Uses rolling quantile of past slope magnitudes for thresholds
      - Enforces regime_min_duration causally (requires sustained evidence)
    """
    df = df.copy()

    # 1. Trailing Moving Average (causal)
    # ma = df['Close'].rolling(window=ma_period, min_periods=ma_period).mean()
    # ma = HMA(df['Close'], timeperiod=ma_period)
    ma = EMA(df['Close'], timeperiod=ma_period)

    # 2. Causal slope with smoothing (trailing) — normalized by MA level for price-scale invariance
    slope = ma.diff() / ma
    slope_sm = super_smoother(slope, period=slope_smoothness)

    # 3. Directional regime by slope magnitude vs threshold
    # Bars where |slope| < slope_threshold start as 0 (undecided); the forward-fill
    # in step 4 will carry the previous trend through these flat patches.
    regime = pd.Series(0, index=df.index, dtype=int)
    regime[slope_sm > 0] = 1
    regime[slope_sm < 0] = -1

    # 4. Forward-fill zero values with latest trend value (causal)
    vals = regime.values.copy()
    last_trend = 0
    for i in range(len(vals)):
        if vals[i] == 0:
            vals[i] = last_trend
        else:
            last_trend = vals[i]
    regime_ff = pd.Series(vals, index=regime.index, dtype=int)

    # 5. Causal enforcement of minimum duration: require sustained evidence
    # Compute run-length of current regime label (causal)
    vals2 = regime_ff.values
    runlen = np.zeros_like(vals2, dtype=int)
    for i in range(1, len(vals2)):
        if vals2[i] == vals2[i-1]:
            runlen[i] = runlen[i-1] + 1
        else:
            runlen[i] = 0

    regime_causal = regime_ff.copy()
    # Require at least regime_min_duration consecutive bars to declare a regime; otherwise mark as range (0)
    min_run = max(int(regime_min_duration) - 1, 0)
    short_mask = runlen < min_run
    regime_causal[short_mask] = 0

    # 6. ATR-based gating: low-volatility regime becomes range (0), causal
    atr = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
    q_atr = float(atr_percentile) / 100.0
    atr_threshold = atr.rolling(window=atr_lookback, min_periods=atr_lookback).quantile(q_atr)
    low_vol = (atr <= atr_threshold).fillna(False)
    regime_causal[low_vol] = 0

    # 7. Filter flat slope regimes to range (0)
    # Adaptive threshold: rolling percentile of past normalised slope magnitudes (causal)
    slope_adaptive_thresh = (
        slope_sm.abs()
        .rolling(window=slope_lookback, min_periods=slope_lookback)
        .quantile(slope_percentile / 100.0)
    )
    flat_slope = slope_sm.abs() < slope_adaptive_thresh.fillna(0)
    if slope_threshold > 0:
        flat_slope = flat_slope | (slope_sm.abs() < slope_threshold)
    regime_causal[flat_slope] = 0

    return regime_causal


# DEPRECATED
# def triple_barrier_hilow_trend_labeler(
#     df,
#     z_window=14,
#     z_thresh=1.0,
#     z_limit=2.5,
#     atr_window=14,
#     tp_mult=4.0,
#     sl_mult=2.0,
#     max_horizon=60,
#     trend_pullback_thresh=0.0, # Z-score threshold for entering with the trend (e.g. buy dip in uptrend)
#     regime_params = None,
#     skip_range = False
# ):
#     """
#     Enhanced labeler that uses Market Regime to filter signals.
#     - Regime 0 (Range): Uses standard Mean Reversion (Buy Oversold, Sell Overbought)
#     - Regime 1 (Uptrend): Only Buys, preferably on dips (Z < trend_pullback_thresh)
#     - Regime -1 (Downtrend): Only Sells, preferably on rallies (Z > -trend_pullback_thresh)
#     """
#     df = df.copy()
#     
#     # 1. Get Regime
#     if regime_params is None:
#         df['Regime'] = get_market_regime(df)
#     else:
#         df['Regime'] = get_market_regime(df, **regime_params)
#     
#     # 2. Compute Z-Score
#     df["mean"] = df["Close"].rolling(z_window).mean()
#     df["std"]  = df["Close"].rolling(z_window).std()
#     df["z"] = (df["Close"] - df["mean"]) / df["std"]
#     
#     # 3. Generate Signals based on Regime
#     signals = pd.Series(index=df.index, dtype=float)
#     
#     # --- Range Logic (Regime 0) ---
#     # Buy Oversold, Sell Overbought
#     if skip_range:
#         pass
#     else:
#         mask_range = df['Regime'] == 0
#         signals[mask_range & (df["z"] < -z_thresh) & (df["z"] > -z_limit)] = 1  # Long
#         signals[mask_range & (df["z"] > +z_thresh) & (df["z"] < +z_limit)] = -1 # Short
#     
#     # --- Uptrend Logic (Regime 1) ---
#     # Only Buy. Look for "Clean Entries" (Dips/Pullbacks)
#     # We use a relaxed Z-score (e.g. Z < 0) to find pullbacks within the trend
#     mask_uptrend = df['Regime'] == 1
#     signals[mask_uptrend & (df["z"] < -trend_pullback_thresh) & (df["z"] > -z_limit)] = 1 # Long on dip
#     
#     # --- Downtrend Logic (Regime -1) ---
#     # Only Sell. Look for Rallies
#     mask_downtrend = df['Regime'] == -1
#     signals[mask_downtrend & (df["z"] > trend_pullback_thresh) & (df["z"] < z_limit)] = -1 # Short on rally
#     
#     signals = signals.dropna()
# 
#     # 4. Triple Barrier Loop (Unchanged logic for barriers)
#     df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
#     events = []
# 
#     for t0, side in signals.items():
#         if t0 >= len(df) - 1: continue
# 
#         # Entry Price Logic (Consistent with Mean Reversion / Pullback style)
#         # Long -> Enter at candle's high
#         # Short -> Enter at candle's low
#         if side == +1:
#             entry_price = df.loc[t0, "High"]
#         else:
#             entry_price = df.loc[t0, "Low"]
#             
#         atr = df.loc[t0, "atr"]
#         if np.isnan(atr) or atr == 0: continue
# 
#         # Barriers
#         if side == +1:  # Long
#             tp = entry_price + tp_mult * atr
#             sl = entry_price - sl_mult * atr
#         else:           # Short
#             tp = entry_price - tp_mult * atr
#             sl = entry_price + sl_mult * atr
# 
#         t_end = min(t0 + max_horizon, df.index[-1])
#         label = 0
#         end_time = t_end
# 
#         for t in range(t0 + 1, t_end + 1):
#             high = df.loc[t, "High"]
#             low  = df.loc[t, "Low"]
# 
#             if side == +1:  # Long
#                 if high >= tp:
#                     label = 1
#                     end_time = t
#                     break
#                 if low <= sl:
#                     label = -1
#                     end_time = t
#                     break
#             else:           # Short
#                 if low <= tp:
#                     label = 1
#                     end_time = t
#                     break
#                 if high >= sl:
#                     label = -1
#                     end_time = t
#                     break
# 
#         events.append({
#             "t0": t0,
#             "side": side,
#             "z": df.loc[t0, "z"],
#             "regime": df.loc[t0, "Regime"], # Log regime for debugging
#             "tp": tp,
#             "sl": sl,
#             "t_end": end_time,
#             "label": label
#         })
# 
#     return pd.DataFrame(events).set_index("t0")

def causal_triple_barrier_hilow_trend_labeler(
    df,
    z_window=14,
    z_thresh=1.0,
    z_limit=2.5,
    atr_window=14,
    tp_mult=4.0,
    sl_mult=2.0,
    max_horizon=60,
    trend_pullback_thresh=0.0, # Z-score threshold for entering with the trend (e.g. buy dip in uptrend)
    regime_params = None,
    skip_range = False
):
    """
    Enhanced labeler that uses Market Regime to filter signals.
    - Regime 0 (Range): Uses standard Mean Reversion (Buy Oversold, Sell Overbought)
    - Regime 1 (Uptrend): Only Buys, preferably on dips (Z < trend_pullback_thresh)
    - Regime -1 (Downtrend): Only Sells, preferably on rallies (Z > -trend_pullback_thresh)
    """
    df = df.copy()
    
    # 1. Get Regime
    if regime_params is None:
        df['Regime'] = causal_market_regime(df)
    else:
        df['Regime'] = causal_market_regime(df, **regime_params)
    
    # 2. Compute Z-Score
    df["mean"] = df["Close"].rolling(z_window).mean()
    df["std"]  = df["Close"].rolling(z_window).std()
    df["z"] = (df["Close"] - df["mean"]) / df["std"]
    
    # 3. Generate Signals based on Regime
    signals = pd.Series(index=df.index, dtype=float)
    
    # --- Range Logic (Regime 0) ---
    # Buy Oversold, Sell Overbought
    if skip_range:
        pass
    else:
        mask_range = df['Regime'] == 0
        signals[mask_range & (df["z"] < -z_thresh) & (df["z"] > -z_limit)] = 1  # Long
        signals[mask_range & (df["z"] > +z_thresh) & (df["z"] < +z_limit)] = -1 # Short
    
    # --- Uptrend Logic (Regime 1) ---
    # Only Buy. Look for "Clean Entries" (Dips/Pullbacks)
    # We use a relaxed Z-score (e.g. Z < 0) to find pullbacks within the trend
    mask_uptrend = df['Regime'] == 1
    signals[mask_uptrend & (df["z"] < -trend_pullback_thresh) & (df["z"] > -z_limit)] = 1 # Long on dip
    
    # --- Downtrend Logic (Regime -1) ---
    # Only Sell. Look for Rallies
    mask_downtrend = df['Regime'] == -1
    signals[mask_downtrend & (df["z"] > trend_pullback_thresh) & (df["z"] < z_limit)] = -1 # Short on rally
    
    signals = signals.dropna()

    # 4. Triple Barrier Loop (Unchanged logic for barriers)
    df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
    events = []

    for t0, side in signals.items():
        if t0 >= len(df) - 1: continue

        # Entry Price Logic (Consistent with Mean Reversion / Pullback style)
        # Long -> Enter at candle's high
        # Short -> Enter at candle's low
        if side == +1:
            entry_price = df.loc[t0, "High"]
        else:
            entry_price = df.loc[t0, "Low"]
            
        atr = df.loc[t0, "atr"]
        if np.isnan(atr) or atr == 0: continue

        # Barriers
        if side == +1:  # Long
            tp = entry_price + tp_mult * atr
            sl = entry_price - sl_mult * atr
        else:           # Short
            tp = entry_price - tp_mult * atr
            sl = entry_price + sl_mult * atr

        t_end = min(t0 + max_horizon, df.index[-1])
        label = 0
        end_time = t_end

        for t in range(t0 + 1, t_end + 1):
            high = df.loc[t, "Close"]  # Use Close for more realistic exit timing
            low  = df.loc[t, "Close"]

            if side == +1:  # Long
                if high >= tp:
                    label = 1
                    end_time = t
                    break
                if low <= sl:
                    label = -1
                    end_time = t
                    break
            else:           # Short
                if low <= tp:
                    label = 1
                    end_time = t
                    break
                if high >= sl:
                    label = -1
                    end_time = t
                    break

        events.append({
            "t0": t0,
            "side": side,
            "z": df.loc[t0, "z"],
            "regime": df.loc[t0, "Regime"], # Log regime for debugging
            "tp": tp,
            "sl": sl,
            "t_end": end_time,
            "label": label
        })

    return pd.DataFrame(events).set_index("t0")

# DEPRECATED
@njit(cache=True)
def _compute_outcomes_nb(highs, lows, atrs, tp_mult, sl_mult):
    """
    Numba-compiled inner kernel for calculate_trade_outcomes_all_candles.
    Scans forward to end-of-data for every bar with no horizon cap, giving
    outcomes that match production behaviour (trades run until TP or SL).
    Returns four float64 arrays: buy_outcomes, sell_outcomes,
    buy_exit_prices, sell_exit_prices.  Unresolved bars stay NaN.
    """
    n = len(highs)
    buy_outcomes     = np.full(n, np.nan)
    sell_outcomes    = np.full(n, np.nan)
    buy_exit_prices  = np.full(n, np.nan)
    sell_exit_prices = np.full(n, np.nan)

    for t0 in range(n - 1):
        atr = atrs[t0]
        if np.isnan(atr) or atr == 0.0:
            continue

        entry_buy  = highs[t0]
        tp_buy     = entry_buy + tp_mult * atr
        sl_buy     = entry_buy - sl_mult * atr

        entry_sell = lows[t0]
        tp_sell    = entry_sell - tp_mult * atr
        sl_sell    = entry_sell + sl_mult * atr

        buy_resolved  = False
        sell_resolved = False

        for t1 in range(t0 + 1, n):
            h = highs[t1]
            l = lows[t1]

            if not buy_resolved:
                if h >= tp_buy:
                    buy_outcomes[t0]    = 1.0
                    buy_exit_prices[t0] = tp_buy
                    buy_resolved = True
                elif l <= sl_buy:
                    buy_outcomes[t0]    = -1.0
                    buy_exit_prices[t0] = sl_buy
                    buy_resolved = True

            if not sell_resolved:
                if l <= tp_sell:
                    sell_outcomes[t0]    = 1.0
                    sell_exit_prices[t0] = tp_sell
                    sell_resolved = True
                elif h >= sl_sell:
                    sell_outcomes[t0]    = -1.0
                    sell_exit_prices[t0] = sl_sell
                    sell_resolved = True

            if buy_resolved and sell_resolved:
                break

    return buy_outcomes, sell_outcomes, buy_exit_prices, sell_exit_prices


def calculate_trade_outcomes_all_candles(
    df,
    atr_window=14,
    tp_mult=4.0,
    sl_mult=2.0,
    max_horizon=None,  # ignored — kept for API compatibility; full horizon always used
):
    """
    Calculate trade outcomes for BOTH buy and sell at every candle.
    Returns DataFrame with columns: ['buy_outcome', 'sell_outcome', 'buy_exit_price', 'sell_exit_price']

    Outcome encoding:
      1: Take Profit hit
     -1: Stop Loss hit
    NaN: Neither TP nor SL reached before end of dataset (only affects the
         last few bars). Callers should fillna(0.0).

    Looks forward to the end of the dataset with no time cap, matching
    production behaviour where a trade runs until TP or SL is hit.
    Uses a Numba-compiled inner loop for O(n) average-case performance.
    """
    df = df.copy()
    df["atr"] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)

    highs = df['High'].values.astype(np.float64)
    lows  = df['Low'].values.astype(np.float64)
    atrs  = df['atr'].values.astype(np.float64)

    buy_out, sell_out, buy_exit, sell_exit = _compute_outcomes_nb(
        highs, lows, atrs, float(tp_mult), float(sl_mult)
    )

    return pd.DataFrame({
        'buy_outcome':     buy_out,
        'sell_outcome':    sell_out,
        'buy_exit_price':  buy_exit,
        'sell_exit_price': sell_exit,
    }, index=df.index)