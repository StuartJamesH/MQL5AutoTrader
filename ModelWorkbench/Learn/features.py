"""
Feature engineering for OHLCV price data.

Public API
----------
  add_all_features            — Full feature set (all indicators + MTF)
  add_selected_features       — Reduced feature set for faster training
  add_price_features          — Minimal OHLC-only feature set
  add_multitimeframe_features — Higher-timeframe indicator overlay
"""

import warnings
import numpy as np
import pandas as pd
import talib
from talib import ATR
from Learn.labels import causal_market_regime


def _tf_to_minutes(tf: str) -> float:
    """Convert common pandas resample strings to minutes.

    Supports: 'min', 'T', 'H', 'D' and numeric prefixes (e.g. '15min', '1H').
    Returns np.nan if unknown.
    """
    if tf is None:
        return np.nan
    s = str(tf).strip().lower()
    if not s:
        return np.nan

    # Extract leading integer if present
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    n = int(s[:i]) if i > 0 else 1
    unit = s[i:]

    # normalize common units
    if unit in ("t", "min", "mins", "minute", "minutes"):
        return float(n)
    if unit in ("h", "hour", "hours"):
        return float(n) * 60.0
    if unit in ("d", "day", "days"):
        return float(n) * 1440.0
    return np.nan


def _infer_base_minutes(df: pd.DataFrame) -> float:
    """Infer base sampling period from the Time column (median diff)."""
    if df is None or len(df) < 3 or 'Time' not in df.columns:
        return np.nan
    t = pd.to_datetime(df['Time'])
    dt = t.diff().dropna()
    if dt.empty:
        return np.nan
    return float(dt.median() / pd.Timedelta(minutes=1))


def _default_mtf_timeframes(df: pd.DataFrame) -> list:
    """Choose sensible higher timeframes based on base frequency."""
    base_min = _infer_base_minutes(df)
    # Fallback to the old default for unknown input.
    if not np.isfinite(base_min) or base_min <= 0:
        return ['5min', '15min', '30min']

    # Only include higher timeframes (strictly greater than base_min).
    if base_min <= 1.0:
        return ['5min', '15min', '30min']
    if base_min <= 5.0:
        return ['15min', '30min', '60min']
    if base_min <= 15.0:
        return ['30min', '60min', '240min']
    if base_min <= 60.0:
        return ['240min', '1440min']
    return ['1440min']

def donchian_trend(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """
    Donchian channel trend direction (+1 uptrend, -1 downtrend).
    Equivalent to the TradingView / PineScript Donchian Trend indicator.
    """

    high = df['High']
    low = df['Low']
    close = df['Close']

    # Donchian channel
    hh = high.rolling(length, min_periods=length).max()
    ll = low.rolling(length, min_periods=length).min()

    trend = np.zeros(len(df), dtype=int)

    for i in range(1, len(df)):
        if pd.isna(hh.iloc[i-1]) or pd.isna(ll.iloc[i-1]):
            trend[i] = trend[i-1]
        elif close.iloc[i] > hh.iloc[i-1]:
            trend[i] = 1
        elif close.iloc[i] < ll.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    return pd.Series(trend, index=df.index, name="donchian_trend")

def time_in_trend(trend_series: pd.Series) -> pd.Series:
    """Bar count since the last trend direction change (+1 / -1)."""
    time_in_trend = np.zeros(len(trend_series), dtype=int)

    for i in range(1, len(trend_series)):
        if trend_series.iloc[i] == trend_series.iloc[i-1]:
            time_in_trend[i] = time_in_trend[i-1] + 1
        else:
            time_in_trend[i] = 1

    return pd.Series(time_in_trend, index=trend_series.index, name="time_in_trend")

def checkhl(data_back, data_forward, hl):
    """Return 1 if the last element of data_back is a pivot high/low, 0 otherwise."""
    if hl == 'high' or hl == 'High':
        ref = data_back[len(data_back)-1]
        for i in range(len(data_back)-1):
            if ref < data_back[i]:
                return 0
        for i in range(len(data_forward)):
            if ref <= data_forward[i]:
                return 0
        return 1
    if hl == 'low' or hl == 'Low':
        ref = data_back[len(data_back)-1]
        for i in range(len(data_back)-1):
            if ref > data_back[i]:
                return 0
        for i in range(len(data_forward)):
            if ref >= data_forward[i]:
                return 0
        return 1


def pivot(osc, LBL, LBR, highlow):
    """Detect pivot highs/lows in osc using LBL left bars and LBR right bars."""
    left = []
    right = []
    pivots = []
    for i in range(len(osc)):
        pivots.append(0.0)
        if i < LBL + 1:
            left.append(osc[i])
        if i > LBL:
            right.append(osc[i])
        if i > LBL + LBR:
            left.append(right[0])
            left.pop(0)
            right.pop(0)
            if checkhl(left, right, highlow):
                pivots[i - LBR] = osc[i - LBR]
    return pivots

def WMA(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average — linearly increasing weights over period."""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def HMA(series: pd.Series, timeperiod: int) -> pd.Series:
    """Hull Moving Average — reduces lag by combining fast/slow WMAs."""
    wma_half = WMA(series, timeperiod // 2) * 2
    wma_full = WMA(series, timeperiod)
    return WMA(wma_half - wma_full, int(np.sqrt(timeperiod)))

def efficiency_ratio(series: pd.Series, window: int) -> pd.Series:
    """Kaufman Efficiency Ratio: trend strength vs noise (causal)."""
    if window <= 1:
        return pd.Series(np.full(len(series), np.nan), index=series.index)
    change = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum()
    return change / (volatility + 1e-9)

def rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Causal rolling VWAP over window."""
    v = volume.astype(float)
    num = (close * v).rolling(window).sum()
    den = v.rolling(window).sum()
    return num / (den + 1e-9)

def rolling_slope_logprice(series, window):
    """
    Compute OLS slope of log(price) over a rolling window.
    Returns array aligned to the right (slope at index t uses data t-window+1...t).
    """
    if window is None:
        return np.full(len(series), np.nan)
    n = int(window)
    if n < 2:
        return np.full(len(series), np.nan)
    if len(series) < n:
        return np.full(len(series), np.nan)

    logp = np.log(series.values)

    # constant sums for X = 0..n-1
    X = np.arange(n)
    X_mean = X.mean()
    denom = ((X - X_mean)**2).sum()

    # rolling sums of y and xy
    y = logp
    y_sum = pd.Series(y).rolling(window).sum().values
    xy = (np.lib.stride_tricks.sliding_window_view(y, window) * X).sum(axis=1)
    # xy has length len(y)-window+1, align it with index: start at window-1
    slopes = np.full(len(y), np.nan)
    yi_mean = y_sum / n
    # compute numerator using vectorized windowed operations:
    # numerator = sum((X-X_mean)*(y - y_mean)) = sum(X*y) - n*X_mean*y_mean
    sum_Xy = xy
    numerator = sum_Xy - n * X_mean * yi_mean[window-1:]
    slopes[window-1:] = numerator / denom
    return slopes

def atr_filter(df, atr_window=28, atr_threshold=40.0, cooldown=5):
    """
    Returns a boolean Series where True indicates rows with ATR above the threshold.
    """
    df = df.copy()
    df['atr'] = ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_window)
    df['atr_bool'] = [1 if x > atr_threshold else 0 for x in df['atr']]
    df['atr_filter'] = df['atr_bool'].rolling(window=cooldown).max().fillna(0).astype(int)
    return df['atr_filter']


def add_multitimeframe_features(
        df: pd.DataFrame,
        timeframes: list = ['5min', '15min', '30min', '60min'],
        causal: bool = True,
) -> pd.DataFrame:
    """
    Resample OHLCV data to each higher timeframe and append trend/momentum indicators.

    Parameters
    ----------
    df         : DataFrame with columns Time, Open, High, Low, Close, Volume
    timeframes : Pandas resample strings for each higher timeframe (e.g. '15min')
    causal     : If True, shift HTF features by 1 bar to prevent lookahead leakage
    """
    df = df.copy()
    df_original = df.copy()
    
    # Ensure Time is datetime and set as index for resampling
    if 'Time' in df.columns:
        df = df.set_index('Time')
    
    base_min = _infer_base_minutes(df_original)
    for tf in timeframes:
        # Safety: only compute true higher-timeframe features.
        tf_min = _tf_to_minutes(tf)
        if np.isfinite(base_min) and np.isfinite(tf_min) and tf_min <= base_min:
            continue
        # Resample OHLCV data
        # NOTE: We intentionally keep default label/closed semantics and enforce causality
        # by shifting HTF features (see below). This prevents partial higher-timeframe bar
        # leakage into earlier base-timeframe rows.
        df_tf = df[['Open', 'High', 'Low', 'Close', 'Volume']].resample(tf).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Trend direction (EMA crossover)
        ema_fast = talib.EMA(df_tf['Close'], timeperiod=8)
        ema_slow = talib.EMA(df_tf['Close'], timeperiod=21)
        df_tf[f'MTF_{tf}_trend'] = np.sign(ema_fast - ema_slow)  # +1 uptrend, -1 downtrend

        # Trend strength (ADX)
        df_tf[f'MTF_{tf}_adx'] = talib.ADX(df_tf['High'], df_tf['Low'], df_tf['Close'], timeperiod=14) / 100

        # Price momentum (ROC - Rate of Change)
        df_tf[f'MTF_{tf}_roc'] = talib.ROC(df_tf['Close'], timeperiod=10) / 100

        # RSI: overbought / oversold on the higher timeframe
        df_tf[f'MTF_{tf}_rsi'] = talib.RSI(df_tf['Close'], timeperiod=14) / 100

        # Distance from EMA (ATR-normalised)
        atr_tf = talib.ATR(df_tf['High'], df_tf['Low'], df_tf['Close'], timeperiod=14)
        df_tf[f'MTF_{tf}_ema_dist'] = (df_tf['Close'] - ema_slow) / (atr_tf + 1e-9)

        # Log-price slope
        # IMPORTANT: keep the slope window fixed; do not make it a function of
        # dataset length, otherwise bulk vs live computations diverge.
        df_tf[f'MTF_{tf}_slope'] = rolling_slope_logprice(df_tf['Close'], window=10)

        # Higher high / lower low detection
        df_tf[f'MTF_{tf}_hh'] = (df_tf['High'] >= df_tf['High'].rolling(5).max().shift(1)).astype(int)
        df_tf[f'MTF_{tf}_ll'] = (df_tf['Low'] <= df_tf['Low'].rolling(5).min().shift(1)).astype(int)

        # Donchian trend and time-in-trend
        df_tf[f'MTF_{tf}_donchian_trend'] = donchian_trend(df_tf, length=20)
        df_tf[f'MTF_{tf}_time_in_trend'] = time_in_trend(df_tf[f'MTF_{tf}_donchian_trend'])
        
        # Forward fill to align with original base timeframe.
        # IMPORTANT: If causal=True, shift HTF-derived features by 1 HTF bar so that
        # a base-timeframe row only sees the last *completed* HTF bar.
        mtf_cols = [c for c in df_tf.columns if c.startswith('MTF_')]
        df_tf_aligned = df_tf
        if causal:
            df_tf_aligned = df_tf.copy()
            df_tf_aligned[mtf_cols] = df_tf_aligned[mtf_cols].shift(1)
        
        # Merge back to original timeframe using forward fill
        for col in mtf_cols:
            # Reindex to original timeframe and forward fill
            df[col] = df_tf_aligned[col].reindex(df.index, method='ffill')
    
    # Reset index to get Time back as column
    df = df.reset_index()
    
    # Merge with original dataframe
    mtf_feature_cols = [c for c in df.columns if c.startswith('MTF_')]
    df_result = df_original.merge(df[['Time'] + mtf_feature_cols], on='Time', how='left')
    
    return df_result

def add_all_features(
        df: pd.DataFrame,
        lookback: int = 8,
        vol_window: int = 20,
        include_mtf: bool = True,
        regime_params: dict = None,
) -> pd.DataFrame:
    """
    Compute the full feature set used for model training.

    Appends log-return, volatility, momentum, ATR-normalised, MTF, indicator,
    and one-hot features to the input DataFrame in-place on a copy.
    """
    df = df.copy()
    
    # Add multi-timeframe features first (if requested)
    if include_mtf:
        df = add_multitimeframe_features(df, timeframes=_default_mtf_timeframes(df), causal=True)

    # Add time based features
    df['hour'] = df['Time'].dt.hour
    df['dayofweek'] = df['Time'].dt.dayofweek
    
    # Compute log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Price action / candle anatomy (helps minority-class precision by better
    # distinguishing clean impulse bars from chop)
    eps = 1e-9
    df['hl_range'] = (df['High'] - df['Low'])
    df['body'] = (df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body_to_range'] = df['body'].abs() / (df['hl_range'] + eps)
    df['close_loc'] = (df['Close'] - df['Low']) / (df['hl_range'] + eps)
    df['gap_open'] = (df['Open'] - df['Close'].shift(1))
    
    # Relative OHLC features (normalize by Close)
    df['O_rel'] = (df['Open'] - df['Close']) / df['Close']
    df['H_rel'] = (df['High'] - df['Close']) / df['Close']
    df['L_rel'] = (df['Low'] - df['Close']) / df['Close']
    df['C_rel'] = 0.0  # always baseline

    # Add causal regime data
    if regime_params is not None:
        df['Regime'] = causal_market_regime(df, **regime_params)
    
    # Volatility scaling (rolling std of returns)
    df['vol'] = df['log_return'].rolling(vol_window).std()
    df['ret_vol_scaled'] = df['log_return'] / df['vol']

    # Multi-horizon z-scores and location (context for 256-length sequences)
    eps = 1e-9
    for w in (64, 128, 256):
        roll_mean = df['Close'].rolling(w).mean()
        roll_std = df['Close'].rolling(w).std()
        df[f'z_{w}'] = (df['Close'] - roll_mean) / (roll_std + eps)
        roll_min = df['Close'].rolling(w).min()
        roll_max = df['Close'].rolling(w).max()
        df[f'price_loc_{w}'] = (df['Close'] - roll_min) / ((roll_max - roll_min) + eps)
        df[f'roll_range_{w}'] = (roll_max - roll_min)

    # Volatility regime features (minority trades often require non-chop)
    df['rv_10'] = df['log_return'].rolling(10).std()
    df['rv_60'] = df['log_return'].rolling(60).std()
    df['rv_ratio_10_60'] = df['rv_10'] / (df['rv_60'] + eps)

    # Parkinson volatility proxy (uses only current/past H/L)
    df['parkinson_20'] = (np.log((df['High'] + eps) / (df['Low'] + eps)) ** 2).rolling(20).mean()

    # Detrended residuals (remove local trend via EMA, causal)
    ema64 = talib.EMA(df['Close'], timeperiod=64)
    ema256 = talib.EMA(df['Close'], timeperiod=256)
    df['close_detrended_64'] = df['Close'] - ema64
    df['close_detrended_256'] = df['Close'] - ema256

    # Prices relative to EMA
    ema = talib.EMA(df['Close'], timeperiod=21)
    df['O_ema'] = df['Open'] - ema
    df['H_ema'] = df['High'] - ema
    df['L_ema'] = df['Low'] - ema
    df['C_ema'] = df['Close'] - ema

    # ATR & slope features
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # ATR-based regime scalars
    df['atr_pct'] = atr / (df['Close'] + eps)
    df['hl_range_atr'] = df['hl_range'] / (atr + eps)
    df['body_atr'] = df['body'] / (atr + eps)
    df['gap_open_atr'] = (df['Open'] - df['Close'].shift(1)) / (atr + eps)

    # Normalize detrended residuals by ATR (available now)
    df['close_detrended_64_atr'] = df['close_detrended_64'] / (atr + eps)
    df['close_detrended_256_atr'] = df['close_detrended_256'] / (atr + eps)

    # Efficiency ratio (trend purity) on multiple horizons
    df['efficiency_64'] = efficiency_ratio(df['Close'], 64)
    df['efficiency_128'] = efficiency_ratio(df['Close'], 128)
    df['efficiency_256'] = efficiency_ratio(df['Close'], 256)
    df['efficiency_30'] = efficiency_ratio(df['Close'], 30)

    df['slope_15'] = rolling_slope_logprice(df['Close'], window=15)
    df['slope_15_z'] = (df['slope_15'] - df['slope_15'].rolling(100).mean()) / df['slope_15'].rolling(100).std()
    df['slope_norm'] = df['slope_15'] / (atr + 1e-6)
    df['slope_15_norm'] = df['slope_norm']

    # Additional longer-horizon slopes to detect sustained moves
    df['slope_30'] = rolling_slope_logprice(df['Close'], window=30)
    df['slope_60'] = rolling_slope_logprice(df['Close'], window=60)
    df['slope_30_norm'] = df['slope_30'] / (atr + 1e-9)
    df['slope_60_norm'] = df['slope_60'] / (atr + 1e-9)
    df['signed_slope_30'] = df['slope_30_norm'] * df['efficiency_30']

    # Cumulative returns over multiple horizons normalized by ATR (trend magnitude)
    for h in (5, 15, 60):
        col = f'cumret_{h}'
        df[col] = df['Close'].pct_change().rolling(h).apply(lambda r: (1 + r).prod() - 1, raw=True)
        df[f'{col}_norm'] = df[col] / (atr + 1e-9)

    # Simple horizon returns (more direct than cumprod; model can choose)
    for h in (1, 5, 15, 60):
        df[f'ret_{h}'] = df['Close'].pct_change(h)
    df['momentum_vote'] = (
        np.sign(df['ret_5'].fillna(0)) +
        np.sign(df['ret_15'].fillna(0)) +
        np.sign(df['ret_60'].fillna(0))
    )

    # VWAP context
    vwap_64 = rolling_vwap(df['Close'], df['Volume'], 64)
    vwap_256 = rolling_vwap(df['Close'], df['Volume'], 256)
    df['vwap_diff_64'] = df['Close'] - vwap_64
    df['vwap_diff_256'] = df['Close'] - vwap_256
    df['vwap_diff_64_atr'] = df['vwap_diff_64'] / (atr + 1e-9)
    df['vwap_diff_256_atr'] = df['vwap_diff_256'] / (atr + 1e-9)

    # Run-length of consecutive up/down moves (positive for up-runs, negative for down-runs)
    def run_length_up_down(close):
        dif = np.sign(close.diff().fillna(0))
        runs = np.zeros(len(dif), dtype=int)
        run = 0
        for i in range(len(dif)):
            v = dif.iat[i]
            if v > 0:
                run = run + 1 if run >= 0 else 1
            elif v < 0:
                run = run - 1 if run <= 0 else -1
            else:
                run = 0
            runs[i] = run
        return runs

    df['run_len'] = run_length_up_down(df['Close'])

    # Donchian channel distance (distance from recent highs/lows)
    df['donchian_high_60'] = df['High'].rolling(60).max()
    df['donchian_low_60']  = df['Low'].rolling(60).min()
    df['pct_from_high_60'] = (df['Close'] - df['donchian_high_60']) / (df['donchian_high_60'] + 1e-9)
    df['pct_from_low_60']  = (df['Close'] - df['donchian_low_60']) / (df['donchian_low_60'] + 1e-9)
    df['donchian_range_60'] = (df['donchian_high_60'] - df['donchian_low_60']) / (df['donchian_low_60'] + 1e-9)

    df['donchian_trend_5'] = donchian_trend(df, length=5)
    df['time_in_trend_5'] = time_in_trend(df['donchian_trend_5'])
    df['donchian_trend_20'] = donchian_trend(df, length=20)
    df['time_in_trend_20'] = time_in_trend(df['donchian_trend_20'])
    df['donchian_trend_60'] = donchian_trend(df, length=60)
    df['time_in_trend_60'] = time_in_trend(df['donchian_trend_60'])
    # Range-adjusted current bar vs regime
    df['hl_vs_rollrange_256'] = df['hl_range'] / (df.get('roll_range_256', np.nan) + eps)
    donchian_mid_60 = (df['donchian_high_60'] + df['donchian_low_60']) / 2.0
    df['donchian_pressure'] = (df['Close'] - donchian_mid_60) / (atr + 1e-9)

    # EMA gap normalized by ATR (short vs long EMAs)
    ema8 = talib.EMA(df['Close'], timeperiod=8)
    ema34 = talib.EMA(df['Close'], timeperiod=34)
    df['ema8_34_diff'] = ema8 - ema34
    df['ema8_34_diff_norm'] = df['ema8_34_diff'] / (atr + 1e-9)

    # Additional EMA trend context (purely causal)
    ema21 = talib.EMA(df['Close'], timeperiod=21)
    ema50 = talib.EMA(df['Close'], timeperiod=50)
    df['ema8_21_diff'] = ema8 - ema21
    df['ema21_50_diff'] = ema21 - ema50
    df['ema_vote'] = np.sign(ema8 - ema21) + np.sign(ema21 - ema50)
    # Slopes of EMAs normalized by ATR (context of acceleration/decay)
    df['ema21_slope_5'] = pd.Series(ema21).diff(5) / (atr + 1e-9)
    df['ema50_slope_5'] = pd.Series(ema50).diff(5) / (atr + 1e-9)
    # Streaks above/below EMA (trend persistence)
    df['above_ema21'] = (df['Close'] > ema21).astype(int)
    df['above_ema8'] = (df['Close'] > ema8).astype(int)
    df['above_ema50'] = (df['Close'] > ema50).astype(int)
    def _streak(x):
        s = np.zeros(len(x), dtype=int)
        for i in range(1, len(x)):
            s[i] = s[i-1] + 1 if x.iloc[i] == 1 and x.iloc[i-1] == 1 else (1 if x.iloc[i] == 1 else 0)
        return s
    def _streak_down(x):
        s = np.zeros(len(x), dtype=int)
        for i in range(1, len(x)):
            s[i] = s[i-1] + 1 if x.iloc[i] == 0 and x.iloc[i-1] == 0 else (1 if x.iloc[i] == 0 else 0)
        return s
    df['above_ema8_streak'] = _streak(df['above_ema8'])
    df['below_ema8_streak'] = _streak_down(df['above_ema8'])
    df['above_ema21_streak'] = _streak(df['above_ema21'])
    df['below_ema21_streak'] = _streak_down(df['above_ema21'])
    df['above_ema50_streak'] = _streak(df['above_ema50'])
    df['below_ema50_streak'] = _streak_down(df['above_ema50'])

    # Z-Scores
    df["mean"] = df["Close"].rolling(14).mean()
    df["std"]  = df["Close"].rolling(14).std()
    df["z"] = (df["Close"] - df["mean"]) / df["std"]
    df['OH_z_flag1'] = [1 if z > 1 else 0 for z in df['z']]
    df['OH_z_flag2'] = [1 if z < -1 else 0 for z in df['z']]

    # BBands
    bbands = talib.BBANDS(df['Close'], timeperiod=14)
    bb_upper = bbands[0]
    bb_mid = bbands[1]
    bb_lower = bbands[2]

    df['bb_width'] = (bb_upper - bb_lower) / bb_mid

    # Additional One-Hot Features
    df['OH_CCI'] = [1 if x>=100 else -1 if x<=-100 else 0 for x in talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)]

    # ── Price-based indicators ────────────────────────────────────────────────
    for ind in [talib.EMA]:
        for period in [8,21,50,128]:
            df[f'PR_{ind.__name__}_{period}'] = ind(df['Close'], period)

    df[f'OH_LOWEST_LOW_{lookback}'] = (df['Low'] == df['Low'].rolling(lookback, min_periods=1).min()).astype(int)
    df[f'OH_HIGHEST_HIGH_{lookback}'] = (df['High'] == df['High'].rolling(lookback, min_periods=1).max()).astype(int)

    # ── Oscillators ───────────────────────────────────────────────────────────
    df['RSI'] = talib.RSI(df['Close'], timeperiod=7)/100
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)/100
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)/100
    df['WilliamsR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)/100

    df['StochK'], df['StochD'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['StochK'] = df['StochK']/100
    df['StochD'] = df['StochD']/100
 
    df['AroonUp'], df['AroonDown'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
    df['AroonUp'] = df['AroonUp']/100
    df['AroonDown'] = df['AroonDown']/100
    df['AroonOsc'] = df['AroonUp'] - df['AroonDown']

    # TODO: determine appropriate MACD scaling
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # MACD slope (trend momentum)
    try:
        df['macd_hist_slope_9'] = pd.Series(df['MACD_hist']).diff(9) / (atr + 1e-9)
    except Exception:
        df['macd_hist_slope_9'] = np.nan

    # DMI/DI components (trend direction and strength)
    try:
        di_plus14 = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14) / 100.0
        di_minus14 = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14) / 100.0
        df['DI_plus_14'] = di_plus14
        df['DI_minus_14'] = di_minus14
        df['DI_diff_14'] = di_plus14 - di_minus14
    except Exception:
        df['DI_plus_14'] = df['DI_minus_14'] = df['DI_diff_14'] = np.nan

    # Bollinger band position (trend side within band)
    try:
        bb_u, bb_m, bb_l = talib.BBANDS(df['Close'], timeperiod=20)
        df['bb_pos'] = (df['Close'] - bb_m) / ((bb_u - bb_l) + 1e-9)
    except Exception:
        df['bb_pos'] = np.nan
    df['bb_push'] = df['bb_pos'] * df.get('slope_15_norm', df['slope_norm'])

    # Volume regime features (works with tick volume too)
    df['log_volume'] = np.log1p(df['Volume'].astype(float))
    vmean = df['Volume'].rolling(100).mean()
    vstd = df['Volume'].rolling(100).std()
    df['volume_z'] = (df['Volume'] - vmean) / (vstd + eps)
    df['vol_direction'] = np.sign(df['log_return'].fillna(0)) * df['volume_z']

    # Final trend confidence score
    try:
        # Enhanced composite trend score combining ADX, DI bias, and EMA slope
        comp = (
            (df.get('ADX', 0) * 100.0) * 0.5 +
            (df.get('DI_diff_14', 0) * 50.0) +
            (df.get('ema21_slope_5', 0) * 100.0)
        )
        df['trend_score'] = comp
    except Exception:
        df['trend_score'] = (df.get('ADX', 0) * 100.0)

    # Binary flag for strong sustained move (useful for gating preds)
    df['is_strong_trend'] = ((df['trend_score'] > 30) | (df['run_len'].abs() >= 8)).astype(int)
    
    # Add trend alignment features if MTF features exist
    if 'MTF_5min_trend' in df.columns:
        # Create a composite trend alignment score
        # This helps identify when multiple timeframes agree on direction
        df['trend_alignment'] = (
            df.get('MTF_5min_trend', 0) + 
            df.get('MTF_15min_trend', 0) + 
            df.get('MTF_30min_trend', 0)
        ) / 3.0  # Average trend across timeframes
        
        # Strong trend filter: all timeframes aligned
        df['OH_all_tf_bull'] = (
            (df.get('MTF_5min_trend', 0) > 0) & 
            (df.get('MTF_15min_trend', 0) > 0) & 
            (df.get('MTF_30min_trend', 0) > 0)
        ).astype(int)
        
        df['OH_all_tf_bear'] = (
            (df.get('MTF_5min_trend', 0) < 0) & 
            (df.get('MTF_15min_trend', 0) < 0) & 
            (df.get('MTF_30min_trend', 0) < 0)
        ).astype(int)
        
        # Average higher timeframe ADX (trend strength across timeframes)
        df['mtf_avg_adx'] = (
            df.get('MTF_5min_adx', 0) + 
            df.get('MTF_15min_adx', 0) + 
            df.get('MTF_30min_adx', 0)
        ) / 3.0

    # Direction confidence combining EMA/momentum votes with MTF alignment (if available)
    df['direction_confidence'] = df.get('ema_vote', 0) + df.get('momentum_vote', 0) + df.get('trend_alignment', 0)

    return df

def add_selected_features(
        df: pd.DataFrame,
        lookback: int = 8,
        vol_window: int = 20,
        include_mtf: bool = True,
        regime_params: dict = None,
) -> pd.DataFrame:
    """
    Compute a reduced feature set optimised for faster training runs.

    Contains a subset of `add_all_features` indicators. Does not include the
    full ATR/EMA multiverse — only the features found most predictive in
    ablation experiments.
    """
    df = df.copy()
    
    # Add multi-timeframe features first (if requested)
    if include_mtf:
        df = add_multitimeframe_features(df, timeframes=_default_mtf_timeframes(df), causal=True)

    # Add regime from labelling function
    if regime_params is not None:
        df['Regime'] = causal_market_regime(df, **regime_params)

    # Add time based features
    df['hour'] = df['Time'].dt.hour
    df['dayofweek'] = df['Time'].dt.dayofweek
    
    # Compute log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Price action / candle anatomy (helps minority-class precision by better
    # distinguishing clean impulse bars from chop)
    eps = 1e-9
    df['hl_range'] = (df['High'] - df['Low'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body'] = (df['Close'] - df['Open'])
    df['body_to_range'] = df['body'].abs() / (df['hl_range'] + eps)
    df['close_loc'] = (df['Close'] - df['Low']) / (df['hl_range'] + eps)
    
    # Relative OHLC features (normalize by Close)
    df['O_rel'] = (df['Open'] - df['Close']) / df['Close']
    df['H_rel'] = (df['High'] - df['Close']) / df['Close']
    df['L_rel'] = (df['Low'] - df['Close']) / df['Close']
    df['C_rel'] = 0.0  # always baseline
    
    # Volatility scaling (rolling std of returns)
    df['vol'] = df['log_return'].rolling(vol_window).std()
    df['ret_vol_scaled'] = df['log_return'] / df['vol']

    # Multi-horizon z-scores and location (context for 256-length sequences)
    eps = 1e-9
    for w in [256]:
        roll_min = df['Close'].rolling(w).min()
        roll_max = df['Close'].rolling(w).max()
        df[f'roll_range_{w}'] = (roll_max - roll_min)
    
    for w in [128, 256]:
        roll_mean = df['Close'].rolling(w).mean()
        roll_std = df['Close'].rolling(w).std()
        df[f'z_{w}'] = (df['Close'] - roll_mean) / (roll_std + eps)

    # Volatility regime features (minority trades often require non-chop)
    df['rv_10'] = df['log_return'].rolling(10).std()
    df['rv_60'] = df['log_return'].rolling(60).std()
    df['rv_ratio_10_60'] = df['rv_10'] / (df['rv_60'] + eps)

    # Parkinson volatility proxy (uses only current/past H/L)
    df['parkinson_20'] = (np.log((df['High'] + eps) / (df['Low'] + eps)) ** 2).rolling(20).mean()

    # Rolling Slopes
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # ATR-based regime scalars
    df['atr_pct'] = atr / (df['Close'] + eps)
    df['hl_range_atr'] = df['hl_range'] / (atr + eps)
    df['body_atr'] = df['body'] / (atr + eps)
    df['gap_open_atr'] = (df['Open'] - df['Close'].shift(1)) / (atr + eps)

    # Normalize detrended residuals by ATR (available now)
    ema64 = talib.EMA(df['Close'], timeperiod=64)
    df['close_detrended_64_atr'] = (df['Close'] - ema64) / (atr + eps)

    # Efficiency ratio (trend purity) on multiple horizons
    df['efficiency_256'] = efficiency_ratio(df['Close'], 256)

    # Additional longer-horizon slopes to detect sustained moves
    df['slope_60'] = rolling_slope_logprice(df['Close'], window=60)
    df['slope_30_norm'] = rolling_slope_logprice(df['Close'], window=30) / (atr + 1e-9)

    # Cumulative returns over multiple horizons normalized by ATR (trend magnitude)
    for h in (15, 60):
        col = f'cumret_{h}'
        df[col] = df['Close'].pct_change().rolling(h).apply(lambda r: (1 + r).prod() - 1, raw=True)
        df[f'{col}_norm'] = df[col] / (atr + 1e-9)

    # Simple horizon returns (more direct than cumprod; model can choose)
    for h in (5, 15):
        df[f'ret_{h}'] = df['Close'].pct_change(h)

    # VWAP context
    vwap_64 = rolling_vwap(df['Close'], df['Volume'], 64)
    vwap_256 = rolling_vwap(df['Close'], df['Volume'], 256)
    df['vwap_diff_64'] = df['Close'] - vwap_64
    df['vwap_diff_256'] = df['Close'] - vwap_256
    df['vwap_diff_256_atr'] = df['vwap_diff_256'] / (atr + 1e-9)

    # Run-length of consecutive up/down moves (positive for up-runs, negative for down-runs)
    def run_length_up_down(close):
        dif = np.sign(close.diff().fillna(0))
        runs = np.zeros(len(dif), dtype=int)
        run = 0
        for i in range(len(dif)):
            v = dif.iat[i]
            if v > 0:
                run = run + 1 if run >= 0 else 1
            elif v < 0:
                run = run - 1 if run <= 0 else -1
            else:
                run = 0
            runs[i] = run
        return runs

    df['run_len'] = run_length_up_down(df['Close'])

    # Range-adjusted current bar vs regime
    df['hl_vs_rollrange_256'] = df['hl_range'] / (df.get('roll_range_256', np.nan) + eps)
    donchian_mid_60 = (df['High'].rolling(60).max() + df['Low'].rolling(60).min()) / 2.0
    df['donchian_pressure'] = (df['Close'] - donchian_mid_60) / (atr + 1e-9)

    # Z-Scores
    df["mean"] = df["Close"].rolling(14).mean()
    df["std"]  = df["Close"].rolling(14).std()
    df["z"] = (df["Close"] - df["mean"]) / df["std"]
    df['OH_z_flag1'] = [1 if z > 1 else 0 for z in df['z']]
    df['OH_z_flag2'] = [1 if z < -1 else 0 for z in df['z']]

    # Additional One-Hot Features
    df['OH_CCI'] = [1 if x>=100 else -1 if x<=-100 else 0 for x in talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)]

    # ── Price-based indicators ────────────────────────────────────────────────
    for ind in [talib.EMA]:
        for period in [21]:
            df[f'PR_{ind.__name__}_{period}'] = ind(df['Close'], period)

    df[f'OH_LOWEST_LOW_{lookback}'] = (df['Low'] == df['Low'].rolling(lookback, min_periods=1).min()).astype(int)
    df[f'OH_HIGHEST_HIGH_{lookback}'] = (df['High'] == df['High'].rolling(lookback, min_periods=1).max()).astype(int)

    # ── Oscillators ────────────────────────────────────────────────────────────
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)/100

    df['StochK'], df['StochD'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['StochK'] = df['StochK']/100
    df['StochD'] = df['StochD']/100

    # Volume regime features (works with tick volume too)
    df['log_volume'] = np.log1p(df['Volume'].astype(float))

    # Add trend alignment features if MTF features exist
    if 'MTF_5min_trend' in df.columns:
        
        # Strong trend filter: all timeframes aligned
        df['OH_all_tf_bull'] = (
            (df.get('MTF_5min_trend', 0) > 0) & 
            (df.get('MTF_15min_trend', 0) > 0) & 
            (df.get('MTF_30min_trend', 0) > 0)
        ).astype(int)
        
        df['OH_all_tf_bear'] = (
            (df.get('MTF_5min_trend', 0) < 0) & 
            (df.get('MTF_15min_trend', 0) < 0) & 
            (df.get('MTF_30min_trend', 0) < 0)
        ).astype(int)
        
        # Average higher timeframe ADX (trend strength across timeframes)
        df['mtf_avg_adx'] = (
            df.get('MTF_5min_adx', 0) + 
            df.get('MTF_15min_adx', 0) + 
            df.get('MTF_30min_adx', 0)
        ) / 3.0

    # Direction confidence combining EMA/momentum votes with MTF alignment (if available)
    df['direction_confidence'] = df.get('ema_vote', 0) + df.get('momentum_vote', 0) + df.get('trend_alignment', 0)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# Feature Library — broad candidate set for automated feature selection
# ─────────────────────────────────────────────────────────────────────────────

def add_feature_library(
        df: pd.DataFrame,
        include_mtf: bool = False,
        fast_mode: bool = False,
        regime_params: dict = None,
) -> pd.DataFrame:
    """
    Broad feature library for short-term FX / futures classification.

    Generates a large candidate set intended to be passed through automated
    feature selection (MI ranking, permutation importance, correlation dedup).
    Does NOT modify add_all_features() or add_selected_features().

    Groups
    ------
    A  — Core price action (candle anatomy, OHLC relative, log returns)
    B  — Volatility (ATR, realised vol, Parkinson, Garman-Klass, Yang-Zhang)
    C  — Trend / momentum (EMAs, slopes, efficiency ratio, Donchian)
    D  — Oscillators (RSI at 4 windows, Stoch, StochRSI, CMO, TRIX, DPO)
    E  — Ichimoku (causal — all components shifted to avoid lookahead)
    F  — Elder Ray / Keltner / Squeeze Momentum
    G  — Volume / flow proxies (OBV, CMF, Force Index, NVI, VWAP)
    H  — Market structure (HH/HL/LH/LL, fractal pivots, pivot points, run-length)
    I  — Lag features (returns t-1..t-5, lagged RSI-14, lagged MACD hist)
    J  — Time / session (hour sin/cos, DoW sin/cos, session flags)
    K  — Candle patterns (talib CDL* flags, range expansion/contraction)
    L  — Multi-timeframe overlay (optional — controlled by include_mtf)

    Parameters
    ----------
    df          : OHLCV DataFrame with columns Time, Open, High, Low, Close, Volume
    include_mtf : Include MTF features (slower; disable for rapid grid search)
    fast_mode   : Skip the slowest groups (candle CDL scan K, fractal pivots H)
    regime_params : If provided, adds a Regime column via causal_market_regime()
    """
    df = df.copy()
    eps = 1e-9
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    open_ = df['Open']
    vol   = df['Volume'].astype(float)

    # ── Pre-compute shared base indicators ───────────────────────────────────
    atr14   = talib.ATR(high, low, close, timeperiod=14)
    atr28   = talib.ATR(high, low, close, timeperiod=28)
    ema8    = talib.EMA(close, timeperiod=8)
    ema13   = talib.EMA(close, timeperiod=13)
    ema21   = talib.EMA(close, timeperiod=21)
    ema50   = talib.EMA(close, timeperiod=50)
    ema200  = talib.EMA(close, timeperiod=200)
    log_ret = np.log(close / close.shift(1))

    # ── A — Core price action ─────────────────────────────────────────────────

    df['fl_log_return']      = log_ret
    df['fl_hl_range']        = high - low
    df['fl_body']            = close - open_
    df['fl_upper_wick']      = high - df[['Open', 'Close']].max(axis=1)
    df['fl_lower_wick']      = df[['Open', 'Close']].min(axis=1) - low
    df['fl_body_to_range']   = df['fl_body'].abs() / (df['fl_hl_range'] + eps)
    df['fl_close_loc']       = (close - low) / (df['fl_hl_range'] + eps)
    df['fl_O_rel']           = (open_ - close) / (close + eps)
    df['fl_H_rel']           = (high  - close) / (close + eps)
    df['fl_L_rel']           = (low   - close) / (close + eps)
    df['fl_gap_open_atr']    = (open_ - close.shift(1)) / (atr14 + eps)
    df['fl_body_atr']        = df['fl_body'] / (atr14 + eps)
    df['fl_hl_range_atr']    = df['fl_hl_range'] / (atr14 + eps)

    # Multi-horizon z-scores and price location
    for w in (20, 60, 128, 256):
        rm   = close.rolling(w).mean()
        rs   = close.rolling(w).std()
        rmin = close.rolling(w).min()
        rmax = close.rolling(w).max()
        df[f'fl_z_{w}']         = (close - rm) / (rs + eps)
        df[f'fl_price_loc_{w}'] = (close - rmin) / ((rmax - rmin) + eps)

    # ── B — Volatility ────────────────────────────────────────────────────────
    for w in (5, 10, 20, 60):
        df[f'fl_rv_{w}'] = log_ret.rolling(w).std()

    df['fl_rv_ratio_5_20']  = df['fl_rv_5']  / (df['fl_rv_20']  + eps)
    df['fl_rv_ratio_20_60'] = df['fl_rv_20'] / (df['fl_rv_60'] + eps)
    df['fl_atr14_pct']      = atr14 / (close + eps)
    df['fl_atr28_pct']      = atr28 / (close + eps)

    # ATR ratio: current ATR vs rolling average (regime filter)
    df['fl_atr_ratio_14']   = atr14 / (atr14.rolling(60).mean() + eps)

    # Parkinson estimator (causal)
    df['fl_parkinson_20'] = (np.log((high + eps) / (low + eps)) ** 2).rolling(20).mean()

    # Garman-Klass volatility (causal)
    _gk = (
        0.5 * np.log((high + eps) / (low + eps)) ** 2
        - (2 * np.log(2) - 1) * np.log((close + eps) / (open_ + eps)) ** 2
    )
    df['fl_gk_vol_20'] = _gk.rolling(20).mean()

    # Yang-Zhang volatility (causal approximation)
    _oc_log = np.log((open_ + eps) / (close.shift(1) + eps))  # overnight
    _co_log = np.log((close + eps) / (open_  + eps))           # intraday
    _yz = _oc_log.rolling(20).var() + _co_log.rolling(20).var()
    df['fl_yz_vol_20'] = _yz

    # Bollinger Band width
    bb_u, bb_m, bb_l = talib.BBANDS(close, timeperiod=20)
    df['fl_bb_width_20'] = (bb_u - bb_l) / (bb_m + eps)
    df['fl_bb_pos_20']   = (close - bb_m) / ((bb_u - bb_l) + eps)

    # ── C — Trend / momentum ──────────────────────────────────────────────────
    for p, e in [(8, ema8), (13, ema13), (21, ema21), (50, ema50), (200, ema200)]:
        df[f'fl_close_vs_ema{p}']  = (close - e) / (atr14 + eps)
        df[f'fl_ema{p}_slope_5']   = pd.Series(e).diff(5) / (atr14 + eps)

    df['fl_ema8_21_diff']    = (ema8  - ema21)  / (atr14 + eps)
    df['fl_ema21_50_diff']   = (ema21 - ema50)  / (atr14 + eps)
    df['fl_ema50_200_diff']  = (ema50 - ema200) / (atr14 + eps)
    df['fl_ema_vote']        = np.sign(ema8 - ema21) + np.sign(ema21 - ema50)

    # Hull MA (existing helper) vs EMA21
    hma20 = HMA(close, 20)
    df['fl_hma20_vs_ema21'] = (hma20 - ema21) / (atr14 + eps)
    df['fl_hma20_slope_3']  = hma20.diff(3) / (atr14 + eps)

    # DEMA / TEMA
    ema14a = talib.EMA(close, timeperiod=14)
    ema14b = talib.EMA(ema14a, timeperiod=14)
    ema14c = talib.EMA(ema14b, timeperiod=14)
    dema14 = 2 * ema14a - ema14b
    tema14 = 3 * (ema14a - ema14b) + ema14c
    df['fl_dema14_vs_close'] = (close - dema14) / (atr14 + eps)
    df['fl_tema14_vs_close'] = (close - tema14) / (atr14 + eps)

    # Rolling OLS slope
    for w in (10, 20, 60):
        df[f'fl_slope_{w}'] = rolling_slope_logprice(close, window=w)
        df[f'fl_slope_{w}_norm'] = df[f'fl_slope_{w}'] / (atr14 + eps)

    # Efficiency ratio
    for w in (14, 30, 60):
        df[f'fl_er_{w}'] = efficiency_ratio(close, w)

    # Donchian trend & time-in-trend
    for length in (5, 20, 60):
        dt = donchian_trend(df, length=length)
        df[f'fl_donchian_trend_{length}']    = dt
        df[f'fl_time_in_trend_{length}']     = time_in_trend(dt)
    for w in (20, 60):
        dh = high.rolling(w).max()
        dl = low.rolling(w).min()
        dm = (dh + dl) / 2.0
        df[f'fl_pct_from_high_{w}']     = (close - dh) / (dh + eps)
        df[f'fl_pct_from_low_{w}']      = (close - dl) / (dl + eps)
        df[f'fl_donchian_pressure_{w}'] = (close - dm) / (atr14 + eps)

    # Simple horizon returns and momentum vote
    for h in (1, 3, 5, 10, 20, 60):
        df[f'fl_ret_{h}'] = close.pct_change(h)
    df['fl_momentum_vote'] = (
        np.sign(df['fl_ret_5'].fillna(0)) +
        np.sign(df['fl_ret_10'].fillna(0)) +
        np.sign(df['fl_ret_20'].fillna(0))
    )

    # ADX / DMI
    df['fl_adx14']     = talib.ADX(high, low, close, timeperiod=14) / 100.0
    df['fl_di_plus']   = talib.PLUS_DI(high, low, close, timeperiod=14) / 100.0
    df['fl_di_minus']  = talib.MINUS_DI(high, low, close, timeperiod=14) / 100.0
    df['fl_di_diff']   = df['fl_di_plus'] - df['fl_di_minus']

    # ROC multi-period
    for p in (5, 10, 20):
        df[f'fl_roc_{p}'] = talib.ROC(close, timeperiod=p) / 100.0

    # ── D — Oscillators ───────────────────────────────────────────────────────
    for p in (3, 7, 14, 21):
        df[f'fl_rsi_{p}'] = talib.RSI(close, timeperiod=p) / 100.0

    df['fl_rsi14_slope_3']  = df['fl_rsi_14'].diff(3)

    # Stochastic
    sk, sd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3,
                         slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['fl_stoch_k'] = sk / 100.0
    df['fl_stoch_d'] = sd / 100.0
    df['fl_stoch_kd_diff'] = (sk - sd) / 100.0

    # Stochastic RSI (manual: RSI of RSI, then Stoch formula)
    _rsi14 = talib.RSI(close, timeperiod=14)
    _rsi14_min = _rsi14.rolling(14).min()
    _rsi14_max = _rsi14.rolling(14).max()
    df['fl_stochrsi_14'] = (_rsi14 - _rsi14_min) / (_rsi14_max - _rsi14_min + eps)

    # CMO (Chande Momentum Oscillator)
    df['fl_cmo_14'] = talib.CMO(close, timeperiod=14) / 100.0

    # TRIX (1-period ROC of triple-smoothed EMA)
    df['fl_trix_14'] = talib.TRIX(close, timeperiod=14) / 100.0

    # DPO (Detrended Price Oscillator — causal, centered on past EMA)
    _dpo_period = 20
    _dpo_shift  = _dpo_period // 2 + 1
    _ema_dpo    = talib.EMA(close, timeperiod=_dpo_period)
    df['fl_dpo_20'] = (close - _ema_dpo.shift(_dpo_shift)) / (atr14 + eps)

    # CCI
    df['fl_cci_14'] = talib.CCI(high, low, close, timeperiod=14) / 200.0
    df['fl_cci_20'] = talib.CCI(high, low, close, timeperiod=20) / 200.0

    # Williams %R
    df['fl_willr_14'] = talib.WILLR(high, low, close, timeperiod=14) / 100.0

    # Aroon
    aroon_u, aroon_d = talib.AROON(high, low, timeperiod=14)
    df['fl_aroon_osc_14'] = (aroon_u - aroon_d) / 100.0

    # MACD (12/26/9) — normalise by ATR
    macd_line, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['fl_macd_hist_norm']     = macd_hist / (atr14 + eps)
    df['fl_macd_line_norm']     = macd_line / (atr14 + eps)
    df['fl_macd_hist_slope_5']  = macd_hist.diff(5) / (atr14 + eps)

    # MFI
    df['fl_mfi_14'] = talib.MFI(high, low, close, vol, timeperiod=14) / 100.0

    # ── E — Ichimoku (causal) ─────────────────────────────────────────────────
    # Tenkan-sen (9): (Max High + Min Low) / 2 over last 9 bars
    tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2.0
    # Kijun-sen (26)
    kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    # Senkou Span A: (tenkan + kijun) / 2 — shifted forward 26; we consume PAST completed cloud
    senA    = ((tenkan + kijun) / 2.0).shift(26)
    # Senkou Span B: (52-period midpoint) shifted forward 26
    senB    = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
    # Chikou: close shifted back 26 (we see past position — fully causal read)
    chikou  = close.shift(26)

    df['fl_ichi_tk_diff']       = (tenkan - kijun) / (atr14 + eps)
    df['fl_ichi_close_vs_kijun']= (close  - kijun) / (atr14 + eps)
    df['fl_ichi_senA_vs_senB']  = (senA   - senB)  / (atr14 + eps)
    df['fl_ichi_close_vs_cloud']= (close  - ((senA + senB) / 2.0)) / (atr14 + eps)
    df['fl_ichi_chikou_vs_close']= (chikou - close) / (atr14 + eps)

    # ── F — Elder Ray / Keltner / Squeeze ────────────────────────────────────
    # Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df['fl_bull_power'] = (high - ema13) / (atr14 + eps)
    df['fl_bear_power'] = (low  - ema13) / (atr14 + eps)

    # Keltner channel (EMA20 ± 2×ATR14)
    kelt_mid   = ema21
    kelt_upper = ema21 + 2.0 * atr14
    kelt_lower = ema21 - 2.0 * atr14
    df['fl_kelt_pos']   = (close - kelt_mid)   / (atr14 + eps)
    df['fl_kelt_width'] = (kelt_upper - kelt_lower) / (close + eps)

    # Squeeze Momentum: BB inside Keltner = "squeezed"
    df['fl_squeeze'] = ((bb_l > kelt_lower) & (bb_u < kelt_upper)).astype(int)
    # Squeeze histogram: momentum proxy during squeeze
    _sq_mom = (close
               - (high.rolling(20).max() + low.rolling(20).min()) / 2.0
               - (bb_m))
    df['fl_squeeze_mom'] = _sq_mom / (atr14 + eps)

    # ── G — Volume / flow proxies ─────────────────────────────────────────────
    df['fl_log_volume']   = np.log1p(vol)
    _vmean                = vol.rolling(60).mean()
    _vstd                 = vol.rolling(60).std()
    df['fl_volume_z']     = (vol - _vmean) / (_vstd + eps)
    df['fl_vol_direction']= np.sign(log_ret.fillna(0)) * df['fl_volume_z']

    # OBV (On-Balance Volume) normalized
    _obv = talib.OBV(close, vol)
    df['fl_obv_slope_10'] = pd.Series(_obv).diff(10) / (vol.rolling(10).mean() + eps)

    # CMF (Chaikin Money Flow) — 20-bar
    _mf_mult   = ((close - low) - (high - close)) / (df['fl_hl_range'] + eps)
    _mf_vol    = _mf_mult * vol
    df['fl_cmf_20'] = _mf_vol.rolling(20).sum() / (vol.rolling(20).sum() + eps)

    # Force Index (Elder)
    _fi = log_ret * vol
    df['fl_force_index_13'] = talib.EMA(pd.Series(_fi), timeperiod=13) / (vol.rolling(13).mean() + eps)

    # NVI (Negative Volume Index) — normalised z-score
    _nvi = pd.Series(np.ones(len(df)), index=df.index)
    for i in range(1, len(df)):
        if vol.iloc[i] < vol.iloc[i - 1]:
            _nvi.iloc[i] = _nvi.iloc[i - 1] * (1.0 + log_ret.iloc[i])
        else:
            _nvi.iloc[i] = _nvi.iloc[i - 1]
    _nvi_ma = _nvi.rolling(60).mean()
    df['fl_nvi_z'] = (_nvi - _nvi_ma) / (_nvi.rolling(60).std() + eps)

    # Rolling VWAP diff (64 and 256)
    vwap64  = rolling_vwap(close, vol, 64)
    vwap256 = rolling_vwap(close, vol, 256)
    df['fl_vwap64_diff']  = (close - vwap64)  / (atr14 + eps)
    df['fl_vwap256_diff'] = (close - vwap256) / (atr14 + eps)

    # ── H — Market structure ──────────────────────────────────────────────────
    # Consecutive up/down bar run-length (reuse helper logic)
    _sign = np.sign(log_ret.fillna(0))
    _run  = np.zeros(len(df), dtype=int)
    _r    = 0
    for _i in range(len(_sign)):
        _v = _sign.iat[_i]
        if _v > 0:    _r = _r + 1 if _r >= 0 else 1
        elif _v < 0:  _r = _r - 1 if _r <= 0 else -1
        else:         _r = 0
        _run[_i] = _r
    df['fl_run_len'] = _run

    # Classic Pivot Point levels (previous bar: PP, R1, S1, R2, S2)
    _ppH = high.shift(1)
    _ppL = low.shift(1)
    _ppC = close.shift(1)
    _pp  = (_ppH + _ppL + _ppC) / 3.0
    _r1  = 2 * _pp - _ppL
    _s1  = 2 * _pp - _ppH
    df['fl_pp_pos']    = (close - _pp) / (atr14 + eps)
    df['fl_r1_dist']   = (close - _r1) / (atr14 + eps)
    df['fl_s1_dist']   = (close - _s1) / (atr14 + eps)


    # Round-number proximity (% distance from nearest 10-pip / integer level)
    _round_level = (close / 0.001).round() * 0.001   # 1-pip rounding for FX
    df['fl_round_num_dist'] = (close - _round_level).abs() / (atr14 + eps)

    # ── I — Lag features ──────────────────────────────────────────────────────
    for _lag in range(1, 6):
        df[f'fl_ret_lag_{_lag}'] = log_ret.shift(_lag)
    df['fl_rsi14_lag_1']       = talib.RSI(close, timeperiod=14).shift(1) / 100.0
    df['fl_rsi14_lag_2']       = talib.RSI(close, timeperiod=14).shift(2) / 100.0
    df['fl_macd_hist_lag_1']   = macd_hist.shift(1) / (atr14 + eps)
    df['fl_stoch_k_lag_1']     = sk.shift(1) / 100.0

    # ── J — Time / session features ───────────────────────────────────────────
    _h  = df['Time'].dt.hour + df['Time'].dt.minute / 60.0
    _dw = df['Time'].dt.dayofweek
    df['fl_hour_sin']  = np.sin(2 * np.pi * _h  / 24.0)
    df['fl_hour_cos']  = np.cos(2 * np.pi * _h  / 24.0)
    df['fl_dow_sin']   = np.sin(2 * np.pi * _dw / 5.0)
    df['fl_dow_cos']   = np.cos(2 * np.pi * _dw / 5.0)

    # Session flags (UTC hours — adjust for broker offset if needed)
    df['fl_session_tokyo']  = ((_h >= 0)  & (_h < 9)).astype(int)
    df['fl_session_london'] = ((_h >= 7)  & (_h < 16)).astype(int)
    df['fl_session_ny']     = ((_h >= 12) & (_h < 21)).astype(int)
    df['fl_session_overlap']= ((_h >= 12) & (_h < 16)).astype(int)  # London/NY overlap

    # ── K — Candle pattern flags (skipped in fast_mode) ──────────────────────
    if not fast_mode:
        _cdl_funcs = {
            'fl_cdl_hammer':      talib.CDLHAMMER,
            'fl_cdl_inv_hammer':  talib.CDLINVERTEDHAMMER,
            'fl_cdl_engulf_bull': talib.CDLENGULFING,
            'fl_cdl_morning_star':talib.CDLMORNINGSTAR,
            'fl_cdl_evening_star':talib.CDLEVENINGSTAR,
            'fl_cdl_shooting_str':talib.CDLSHOOTINGSTAR,
            'fl_cdl_doji':        talib.CDLDOJI,
            'fl_cdl_harami':      talib.CDLHARAMI,
            'fl_cdl_piercing':    talib.CDLPIERCING,
            'fl_cdl_3_white':     talib.CDL3WHITESOLDIERS,
            'fl_cdl_3_black':     talib.CDL3BLACKCROWS,
        }
        for _name, _fn in _cdl_funcs.items():
            df[_name] = _fn(open_, high, low, close) / 100.0

        # Inside bar / outside bar (range expansion/contraction)
        df['fl_inside_bar']  = (
            (high < high.shift(1)) & (low > low.shift(1))
        ).astype(int)
        df['fl_outside_bar'] = (
            (high > high.shift(1)) & (low < low.shift(1))
        ).astype(int)
        df['fl_range_expand'] = (
            df['fl_hl_range'] > df['fl_hl_range'].rolling(5).mean()
        ).astype(int)
        df['fl_range_contract'] = (
            df['fl_hl_range'] < df['fl_hl_range'].rolling(5).mean()
        ).astype(int)
        # N-bar range expansion vs ATR
        df['fl_range_vs_atr5'] = df['fl_hl_range'].rolling(5).mean() / (atr14 + eps)

    # ── L — Multi-timeframe overlay (optional) ────────────────────────────────
    if include_mtf:
        df = add_multitimeframe_features(df, timeframes=_default_mtf_timeframes(df), causal=True)

    # ── Regime (optional) ─────────────────────────────────────────────────────
    if regime_params is not None:
        df['fl_regime'] = causal_market_regime(df, **regime_params)

    return df


def add_price_features(df: pd.DataFrame, regime_params=None) -> pd.DataFrame:
    """
    Minimal OHLC feature set: time, log-return, and relative OHLC columns only.
    Used as a lightweight baseline for price-action-only models.
    """
    df = df.copy()

    # Add time based features
    df['hour'] = df['Time'].dt.hour
    df['dayofweek'] = df['Time'].dt.dayofweek
    
    # Compute log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Relative OHLC features (normalize by Close)
    df['O_rel'] = (df['Open'] - df['Close']) / df['Close']
    df['H_rel'] = (df['High'] - df['Close']) / df['Close']
    df['L_rel'] = (df['Low'] - df['Close']) / df['Close']

    # Add causal regime data
    if regime_params is not None:
        df['Regime'] = causal_market_regime(df, **regime_params)

    return df 

def _add_features_EURUSD(df: pd.DataFrame, include_mtf: bool = False, regime_params: dict = None) -> pd.DataFrame:
    """
    Add the features selected for EURUSD by the Feature ML Lab.
    Generated automatically — edit with care.

    Pass regime_params to enable the causal market regime feature (fl_regime).

    Selected features (100 total):
        fl_regime, fl_aroon_osc_14, fl_close_vs_ema13, fl_dema14_vs_close,
    fl_stochrsi_14, fl_macd_hist_norm, fl_pct_from_low_60, fl_di_minus,
    fl_momentum_vote, fl_slope_60_norm, fl_rsi14_slope_3, fl_di_plus,
    fl_hl_range_atr, MTF_5min_donchian_trend, fl_z_60, MTF_5min_slope,
    fl_stoch_k, fl_macd_hist_slope_5, fl_stoch_kd_diff, fl_pct_from_high_60,
    fl_donchian_trend_60, fl_bb_width_20, fl_donchian_trend_5, fl_ema_vote,
    fl_force_index_13, fl_ichi_tk_diff, fl_pp_pos, MTF_5min_hh, MTF_5min_ll,
    fl_ret_10, fl_roc_5, fl_price_loc_128, MTF_5min_ema_dist,
    fl_ichi_chikou_vs_close, fl_close_loc, fl_time_in_trend_20, fl_H_rel,
    MTF_5min_trend, fl_rv_ratio_20_60, fl_hl_range, fl_session_ny, fl_ret_3,
    fl_mfi_14, fl_time_in_trend_5, fl_L_rel, fl_cdl_morning_star, MTF_5min_adx,
    fl_range_vs_atr5, MTF_30min_rsi, fl_donchian_trend_20, fl_ret_lag_2,
    fl_body_to_range, fl_cdl_engulf_bull, fl_range_contract, fl_session_london,
    fl_dow_sin, MTF_15min_trend, MTF_15min_ll, fl_obv_slope_10, fl_roc_20,
    MTF_15min_hh, fl_price_loc_256, fl_er_14, MTF_30min_time_in_trend,
    fl_log_volume, fl_adx14, fl_rv_60, fl_round_num_dist, fl_ichi_senA_vs_senB,
    fl_vol_direction, fl_run_len, MTF_15min_time_in_trend, fl_volume_z,
    fl_cdl_shooting_str, fl_squeeze, fl_session_overlap, fl_outside_bar,
    MTF_30min_hh, MTF_30min_ll, fl_inside_bar, MTF_15min_donchian_trend,
    fl_rsi14_lag_2, fl_pct_from_high_20, fl_pct_from_low_20, fl_rv_ratio_5_20,
    fl_hour_sin, fl_atr_ratio_14, fl_time_in_trend_60, fl_nvi_z, fl_er_60,
    MTF_30min_slope, fl_gap_open_atr, MTF_15min_roc, fl_cdl_piercing,
    fl_cdl_evening_star, fl_squeeze_mom, fl_cdl_3_white, fl_parkinson_20,
    fl_cdl_doji, fl_dow_cos
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df = add_feature_library(df, include_mtf=include_mtf, regime_params=regime_params)
    df = df.copy()
    _keep = [
        c for c in [
        'fl_regime',
        'fl_aroon_osc_14',
        'fl_close_vs_ema13',
        'fl_dema14_vs_close',
        'fl_stochrsi_14',
        'fl_macd_hist_norm',
        'fl_pct_from_low_60',
        'fl_di_minus',
        'fl_momentum_vote',
        'fl_slope_60_norm',
        'fl_rsi14_slope_3',
        'fl_di_plus',
        'fl_hl_range_atr',
        'MTF_5min_donchian_trend',
        'fl_z_60',
        'MTF_5min_slope',
        'fl_stoch_k',
        'fl_macd_hist_slope_5',
        'fl_stoch_kd_diff',
        'fl_pct_from_high_60',
        'fl_donchian_trend_60',
        'fl_bb_width_20',
        'fl_donchian_trend_5',
        'fl_ema_vote',
        'fl_force_index_13',
        'fl_ichi_tk_diff',
        'fl_pp_pos',
        'MTF_5min_hh',
        'MTF_5min_ll',
        'fl_ret_10',
        'fl_roc_5',
        'fl_price_loc_128',
        'MTF_5min_ema_dist',
        'fl_ichi_chikou_vs_close',
        'fl_close_loc',
        'fl_time_in_trend_20',
        'fl_H_rel',
        'MTF_5min_trend',
        'fl_rv_ratio_20_60',
        'fl_hl_range',
        'fl_session_ny',
        'fl_ret_3',
        'fl_mfi_14',
        'fl_time_in_trend_5',
        'fl_L_rel',
        'fl_cdl_morning_star',
        'MTF_5min_adx',
        'fl_range_vs_atr5',
        'MTF_30min_rsi',
        'fl_donchian_trend_20',
        'fl_ret_lag_2',
        'fl_body_to_range',
        'fl_cdl_engulf_bull',
        'fl_range_contract',
        'fl_session_london',
        'fl_dow_sin',
        'MTF_15min_trend',
        'MTF_15min_ll',
        'fl_obv_slope_10',
        'fl_roc_20',
        'MTF_15min_hh',
        'fl_price_loc_256',
        'fl_er_14',
        'MTF_30min_time_in_trend',
        'fl_log_volume',
        'fl_adx14',
        'fl_rv_60',
        'fl_round_num_dist',
        'fl_ichi_senA_vs_senB',
        'fl_vol_direction',
        'fl_run_len',
        'MTF_15min_time_in_trend',
        'fl_volume_z',
        'fl_cdl_shooting_str',
        'fl_squeeze',
        'fl_session_overlap',
        'fl_outside_bar',
        'MTF_30min_hh',
        'MTF_30min_ll',
        'fl_inside_bar',
        'MTF_15min_donchian_trend',
        'fl_rsi14_lag_2',
        'fl_pct_from_high_20',
        'fl_pct_from_low_20',
        'fl_rv_ratio_5_20',
        'fl_hour_sin',
        'fl_atr_ratio_14',
        'fl_time_in_trend_60',
        'fl_nvi_z',
        'fl_er_60',
        'MTF_30min_slope',
        'fl_gap_open_atr',
        'MTF_15min_roc',
        'fl_cdl_piercing',
        'fl_cdl_evening_star',
        'fl_squeeze_mom',
        'fl_cdl_3_white',
        'fl_parkinson_20',
        'fl_cdl_doji',
        'fl_dow_cos',
        ]
        if c in df.columns
    ]
    # Retain OHLCV + Time columns alongside features
    _ohlcv = [c for c in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'sell_y', 'buy_y']
              if c in df.columns]
    _final = _ohlcv + [c for c in _keep if c not in _ohlcv]
    return df[_final]


def _add_features_XAUUSD(df: pd.DataFrame, include_mtf: bool = False, regime_params: dict = None) -> pd.DataFrame:
    """
    Add the features selected for XAUUSD by the Feature ML Lab.
    Generated automatically — edit with care.

    Pass regime_params to enable the causal market regime feature (fl_regime).

    Selected features (101 total):
        fl_regime, fl_pct_from_high_60, fl_rv_10, fl_stochrsi_14, fl_close_vs_ema8,
    fl_z_60, fl_pct_from_low_60, fl_macd_hist_slope_5, fl_di_minus, MTF_5min_ll,
    fl_macd_hist_norm, fl_donchian_trend_5, fl_ema_vote, fl_di_plus,
    MTF_5min_slope, fl_stoch_k, fl_donchian_trend_60, fl_ichi_close_vs_kijun,
    fl_pp_pos, MTF_5min_hh, fl_donchian_trend_20, fl_H_rel,
    MTF_5min_donchian_trend, fl_momentum_vote, fl_roc_10, fl_price_loc_128,
    fl_aroon_osc_14, fl_hl_range_atr, fl_cdl_evening_star, fl_roc_5,
    fl_rsi14_slope_3, MTF_5min_ema_dist, MTF_15min_ll, MTF_15min_hh,
    MTF_5min_trend, fl_adx14, fl_price_loc_256, fl_squeeze_mom,
    fl_rv_ratio_5_20, fl_gk_vol_20, fl_bb_width_20, MTF_5min_time_in_trend,
    fl_hl_range, fl_range_vs_atr5, fl_force_index_13, fl_ret_3, fl_er_14,
    fl_mfi_14, fl_obv_slope_10, fl_ret_60, fl_ret_20, fl_close_loc,
    fl_ret_lag_2, fl_cdl_morning_star, fl_volume_z, fl_cdl_piercing,
    fl_range_contract, MTF_30min_donchian_trend, fl_pct_from_high_20,
    fl_time_in_trend_5, fl_rsi14_lag_2, fl_time_in_trend_60, fl_L_rel,
    MTF_30min_time_in_trend, fl_run_len, fl_gap_open_atr, fl_squeeze,
    MTF_30min_ll, fl_cdl_doji, fl_session_london, fl_cdl_engulf_bull,
    fl_dow_cos, MTF_15min_trend, MTF_30min_trend, MTF_15min_donchian_trend,
    fl_ichi_tk_diff, fl_pct_from_low_20, fl_er_30, fl_ichi_senA_vs_senB,
    fl_log_return, fl_hour_cos, fl_time_in_trend_20, fl_atr_ratio_14,
    MTF_30min_adx, fl_rv_60, fl_ret_lag_1, fl_cmf_20, fl_er_60,
    fl_rv_ratio_20_60, MTF_15min_time_in_trend, MTF_30min_rsi, MTF_15min_roc,
    MTF_15min_adx, fl_body_to_range, fl_log_volume, fl_yz_vol_20, fl_ret_lag_3,
    fl_outside_bar, fl_session_overlap, fl_cdl_harami, fl_session_ny
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df = add_feature_library(df, include_mtf=include_mtf, regime_params=regime_params)
    df = df.copy()
    _keep = [
        c for c in [
        'fl_regime',
        'fl_pct_from_high_60',
        'fl_rv_10',
        'fl_stochrsi_14',
        'fl_close_vs_ema8',
        'fl_z_60',
        'fl_pct_from_low_60',
        'fl_macd_hist_slope_5',
        'fl_di_minus',
        'MTF_5min_ll',
        'fl_macd_hist_norm',
        'fl_donchian_trend_5',
        'fl_ema_vote',
        'fl_di_plus',
        'MTF_5min_slope',
        'fl_stoch_k',
        'fl_donchian_trend_60',
        'fl_ichi_close_vs_kijun',
        'fl_pp_pos',
        'MTF_5min_hh',
        'fl_donchian_trend_20',
        'fl_H_rel',
        'MTF_5min_donchian_trend',
        'fl_momentum_vote',
        'fl_roc_10',
        'fl_price_loc_128',
        'fl_aroon_osc_14',
        'fl_hl_range_atr',
        'fl_cdl_evening_star',
        'fl_roc_5',
        'fl_rsi14_slope_3',
        'MTF_5min_ema_dist',
        'MTF_15min_ll',
        'MTF_15min_hh',
        'MTF_5min_trend',
        'fl_adx14',
        'fl_price_loc_256',
        'fl_squeeze_mom',
        'fl_rv_ratio_5_20',
        'fl_gk_vol_20',
        'fl_bb_width_20',
        'MTF_5min_time_in_trend',
        'fl_hl_range',
        'fl_range_vs_atr5',
        'fl_force_index_13',
        'fl_ret_3',
        'fl_er_14',
        'fl_mfi_14',
        'fl_obv_slope_10',
        'fl_ret_60',
        'fl_ret_20',
        'fl_close_loc',
        'fl_ret_lag_2',
        'fl_cdl_morning_star',
        'fl_volume_z',
        'fl_cdl_piercing',
        'fl_range_contract',
        'MTF_30min_donchian_trend',
        'fl_pct_from_high_20',
        'fl_time_in_trend_5',
        'fl_rsi14_lag_2',
        'fl_time_in_trend_60',
        'fl_L_rel',
        'MTF_30min_time_in_trend',
        'fl_run_len',
        'fl_gap_open_atr',
        'fl_squeeze',
        'MTF_30min_ll',
        'fl_cdl_doji',
        'fl_session_london',
        'fl_cdl_engulf_bull',
        'fl_dow_cos',
        'MTF_15min_trend',
        'MTF_30min_trend',
        'MTF_15min_donchian_trend',
        'fl_ichi_tk_diff',
        'fl_pct_from_low_20',
        'fl_er_30',
        'fl_ichi_senA_vs_senB',
        'fl_log_return',
        'fl_hour_cos',
        'fl_time_in_trend_20',
        'fl_atr_ratio_14',
        'MTF_30min_adx',
        'fl_rv_60',
        'fl_ret_lag_1',
        'fl_cmf_20',
        'fl_er_60',
        'fl_rv_ratio_20_60',
        'MTF_15min_time_in_trend',
        'MTF_30min_rsi',
        'MTF_15min_roc',
        'MTF_15min_adx',
        'fl_body_to_range',
        'fl_log_volume',
        'fl_yz_vol_20',
        'fl_ret_lag_3',
        'fl_outside_bar',
        'fl_session_overlap',
        'fl_cdl_harami',
        'fl_session_ny',
        ]
        if c in df.columns
    ]
    # Retain OHLCV + Time columns alongside features
    _ohlcv = [c for c in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'sell_y', 'buy_y']
              if c in df.columns]
    _final = _ohlcv + [c for c in _keep if c not in _ohlcv]
    return df[_final]


def _add_features_US2000(df: pd.DataFrame, include_mtf: bool = False, regime_params: dict = None) -> pd.DataFrame:
    """
    Add the features selected for US2000 by the Feature ML Lab.
    Generated automatically — edit with care.

    Pass regime_params to enable the causal market regime feature (fl_regime).

    Selected features (107 total):
        fl_regime, fl_aroon_osc_14, fl_close_vs_ema8, fl_momentum_vote,
    fl_pct_from_low_60, MTF_5min_ll, fl_stochrsi_14, fl_macd_hist_norm, fl_z_60,
    fl_pct_from_high_60, fl_rsi14_slope_3, fl_di_minus, fl_hl_range_atr,
    MTF_5min_hh, fl_stoch_k, MTF_5min_slope, fl_stoch_kd_diff,
    fl_donchian_trend_60, fl_di_plus, fl_rv_10, fl_squeeze_mom, fl_mfi_14,
    fl_ema_vote, fl_price_loc_128, MTF_5min_ema_dist, fl_ret_lag_2,
    fl_donchian_trend_5, fl_macd_hist_slope_5, fl_force_index_13,
    fl_ichi_close_vs_kijun, fl_ret_3, fl_ichi_tk_diff, fl_obv_slope_10,
    MTF_5min_trend, fl_pp_pos, fl_close_loc, fl_session_ny, MTF_30min_ll,
    fl_range_contract, fl_roc_10, fl_roc_5, MTF_15min_hh, MTF_15min_ll,
    fl_rv_ratio_5_20, fl_adx14, fl_rv_60, fl_er_30, fl_bb_width_20, fl_er_60,
    MTF_30min_rsi, fl_range_vs_atr5, fl_nvi_z, fl_session_london,
    fl_session_overlap, fl_session_tokyo, fl_outside_bar, fl_dow_sin, fl_er_14,
    fl_pct_from_low_20, fl_log_volume, fl_hour_cos, fl_donchian_trend_20,
    fl_time_in_trend_60, fl_price_loc_256, MTF_15min_adx, fl_cdl_morning_star,
    fl_parkinson_20, fl_hour_sin, fl_ema50_200_diff, fl_vol_direction,
    MTF_5min_time_in_trend, fl_ret_lag_1, fl_ret_lag_5, fl_hl_range, fl_run_len,
    MTF_30min_time_in_trend, fl_body_to_range, fl_L_rel,
    MTF_5min_donchian_trend, fl_cdl_engulf_bull, fl_lower_wick,
    fl_cdl_inv_hammer, MTF_15min_trend, fl_pct_from_high_20, fl_ret_60,
    MTF_15min_ema_dist, fl_rsi14_lag_2, fl_cmf_20, fl_rv_ratio_20_60,
    MTF_30min_adx, MTF_5min_adx, fl_ichi_senA_vs_senB, MTF_30min_slope,
    fl_H_rel, fl_ret_lag_3, MTF_15min_slope, fl_gap_open_atr, fl_upper_wick,
    fl_ret_lag_4, fl_cdl_harami, fl_cdl_piercing, MTF_30min_hh,
    fl_cdl_evening_star, MTF_30min_donchian_trend, fl_dow_cos, MTF_30min_trend,
    MTF_15min_donchian_trend
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df = add_feature_library(df, include_mtf=include_mtf, regime_params=regime_params)
    df = df.copy()
    _keep = [
        c for c in [
        'fl_regime',
        'fl_aroon_osc_14',
        'fl_close_vs_ema8',
        'fl_momentum_vote',
        'fl_pct_from_low_60',
        'MTF_5min_ll',
        'fl_stochrsi_14',
        'fl_macd_hist_norm',
        'fl_z_60',
        'fl_pct_from_high_60',
        'fl_rsi14_slope_3',
        'fl_di_minus',
        'fl_hl_range_atr',
        'MTF_5min_hh',
        'fl_stoch_k',
        'MTF_5min_slope',
        'fl_stoch_kd_diff',
        'fl_donchian_trend_60',
        'fl_di_plus',
        'fl_rv_10',
        'fl_squeeze_mom',
        'fl_mfi_14',
        'fl_ema_vote',
        'fl_price_loc_128',
        'MTF_5min_ema_dist',
        'fl_ret_lag_2',
        'fl_donchian_trend_5',
        'fl_macd_hist_slope_5',
        'fl_force_index_13',
        'fl_ichi_close_vs_kijun',
        'fl_ret_3',
        'fl_ichi_tk_diff',
        'fl_obv_slope_10',
        'MTF_5min_trend',
        'fl_pp_pos',
        'fl_close_loc',
        'fl_session_ny',
        'MTF_30min_ll',
        'fl_range_contract',
        'fl_roc_10',
        'fl_roc_5',
        'MTF_15min_hh',
        'MTF_15min_ll',
        'fl_rv_ratio_5_20',
        'fl_adx14',
        'fl_rv_60',
        'fl_er_30',
        'fl_bb_width_20',
        'fl_er_60',
        'MTF_30min_rsi',
        'fl_range_vs_atr5',
        'fl_nvi_z',
        'fl_session_london',
        'fl_session_overlap',
        'fl_session_tokyo',
        'fl_outside_bar',
        'fl_dow_sin',
        'fl_er_14',
        'fl_pct_from_low_20',
        'fl_log_volume',
        'fl_hour_cos',
        'fl_donchian_trend_20',
        'fl_time_in_trend_60',
        'fl_price_loc_256',
        'MTF_15min_adx',
        'fl_cdl_morning_star',
        'fl_parkinson_20',
        'fl_hour_sin',
        'fl_ema50_200_diff',
        'fl_vol_direction',
        'MTF_5min_time_in_trend',
        'fl_ret_lag_1',
        'fl_ret_lag_5',
        'fl_hl_range',
        'fl_run_len',
        'MTF_30min_time_in_trend',
        'fl_body_to_range',
        'fl_L_rel',
        'MTF_5min_donchian_trend',
        'fl_cdl_engulf_bull',
        'fl_lower_wick',
        'fl_cdl_inv_hammer',
        'MTF_15min_trend',
        'fl_pct_from_high_20',
        'fl_ret_60',
        'MTF_15min_ema_dist',
        'fl_rsi14_lag_2',
        'fl_cmf_20',
        'fl_rv_ratio_20_60',
        'MTF_30min_adx',
        'MTF_5min_adx',
        'fl_ichi_senA_vs_senB',
        'MTF_30min_slope',
        'fl_H_rel',
        'fl_ret_lag_3',
        'MTF_15min_slope',
        'fl_gap_open_atr',
        'fl_upper_wick',
        'fl_ret_lag_4',
        'fl_cdl_harami',
        'fl_cdl_piercing',
        'MTF_30min_hh',
        'fl_cdl_evening_star',
        'MTF_30min_donchian_trend',
        'fl_dow_cos',
        'MTF_30min_trend',
        'MTF_15min_donchian_trend',
        ]
        if c in df.columns
    ]
    # Retain OHLCV + Time columns alongside features
    _ohlcv = [c for c in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'sell_y', 'buy_y']
              if c in df.columns]
    _final = _ohlcv + [c for c in _keep if c not in _ohlcv]
    return df[_final]


def _add_features_SpotCrude(df: pd.DataFrame, include_mtf: bool = False, regime_params: dict = None) -> pd.DataFrame:
    """
    Add the features selected for SpotCrude by the Feature ML Lab.
    Generated automatically — edit with care.

    Pass regime_params to enable the causal market regime feature (fl_regime).

    Selected features (104 total):
        fl_regime, fl_stochrsi_14, fl_close_vs_ema13, fl_momentum_vote,
    fl_dema14_vs_close, fl_hma20_vs_ema21, fl_body_atr, fl_rsi_21, fl_di_minus,
    fl_pct_from_low_60, fl_aroon_osc_14, fl_di_plus, fl_macd_hist_norm,
    fl_stoch_k, fl_price_loc_128, fl_donchian_trend_5, fl_macd_hist_slope_5,
    fl_stoch_kd_diff, MTF_15min_hh, fl_hl_range_atr, fl_close_loc, fl_roc_10,
    MTF_5min_slope, fl_donchian_trend_60, fl_price_loc_256, fl_pct_from_high_60,
    fl_rsi14_slope_3, MTF_15min_ll, fl_L_rel, fl_H_rel, MTF_5min_hh,
    MTF_5min_trend, fl_ema_vote, fl_ichi_tk_diff, MTF_15min_ema_dist, fl_er_60,
    fl_ret_20, fl_bb_width_20, fl_rv_60, MTF_5min_donchian_trend, MTF_5min_ll,
    fl_hl_range, MTF_30min_ll, fl_dow_sin, fl_ret_5, fl_force_index_13,
    MTF_5min_ema_dist, fl_obv_slope_10, fl_er_14, fl_pp_pos,
    fl_time_in_trend_20, fl_rv_ratio_20_60, fl_cdl_engulf_bull,
    fl_range_contract, fl_session_ny, fl_ret_lag_2, fl_yz_vol_20,
    fl_outside_bar, MTF_15min_trend, fl_ret_3, fl_ret_60, fl_er_30,
    fl_time_in_trend_5, fl_mfi_14, MTF_15min_slope, fl_cdl_morning_star,
    fl_hour_cos, fl_nvi_z, fl_ret_lag_4, fl_squeeze_mom, fl_session_london,
    fl_volume_z, fl_parkinson_20, fl_dow_cos, MTF_30min_trend,
    MTF_30min_donchian_trend, fl_pct_from_high_20, fl_ichi_senA_vs_senB,
    fl_rv_ratio_5_20, fl_pct_from_low_20, fl_ret_lag_5, MTF_15min_adx,
    fl_range_vs_atr5, fl_adx14, fl_hour_sin, fl_gap_open_atr, fl_cmf_20,
    MTF_30min_adx, MTF_30min_time_in_trend, MTF_15min_time_in_trend,
    fl_upper_wick, MTF_5min_adx, fl_lower_wick, fl_log_volume,
    fl_donchian_trend_20, MTF_30min_hh, fl_ret_lag_1, fl_cdl_evening_star,
    fl_cdl_piercing, fl_session_overlap, fl_inside_bar, fl_cdl_inv_hammer,
    fl_session_tokyo, MTF_15min_donchian_trend
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df = add_feature_library(df, include_mtf=include_mtf, regime_params=regime_params)
    df = df.copy()
    _keep = [
        c for c in [
        'fl_regime',
        'fl_stochrsi_14',
        'fl_close_vs_ema13',
        'fl_momentum_vote',
        'fl_dema14_vs_close',
        'fl_hma20_vs_ema21',
        'fl_body_atr',
        'fl_rsi_21',
        'fl_di_minus',
        'fl_pct_from_low_60',
        'fl_aroon_osc_14',
        'fl_di_plus',
        'fl_macd_hist_norm',
        'fl_stoch_k',
        'fl_price_loc_128',
        'fl_donchian_trend_5',
        'fl_macd_hist_slope_5',
        'fl_stoch_kd_diff',
        'MTF_15min_hh',
        'fl_hl_range_atr',
        'fl_close_loc',
        'fl_roc_10',
        'MTF_5min_slope',
        'fl_donchian_trend_60',
        'fl_price_loc_256',
        'fl_pct_from_high_60',
        'fl_rsi14_slope_3',
        'MTF_15min_ll',
        'fl_L_rel',
        'fl_H_rel',
        'MTF_5min_hh',
        'MTF_5min_trend',
        'fl_ema_vote',
        'fl_ichi_tk_diff',
        'MTF_15min_ema_dist',
        'fl_er_60',
        'fl_ret_20',
        'fl_bb_width_20',
        'fl_rv_60',
        'MTF_5min_donchian_trend',
        'MTF_5min_ll',
        'fl_hl_range',
        'MTF_30min_ll',
        'fl_dow_sin',
        'fl_ret_5',
        'fl_force_index_13',
        'MTF_5min_ema_dist',
        'fl_obv_slope_10',
        'fl_er_14',
        'fl_pp_pos',
        'fl_time_in_trend_20',
        'fl_rv_ratio_20_60',
        'fl_cdl_engulf_bull',
        'fl_range_contract',
        'fl_session_ny',
        'fl_ret_lag_2',
        'fl_yz_vol_20',
        'fl_outside_bar',
        'MTF_15min_trend',
        'fl_ret_3',
        'fl_ret_60',
        'fl_er_30',
        'fl_time_in_trend_5',
        'fl_mfi_14',
        'MTF_15min_slope',
        'fl_cdl_morning_star',
        'fl_hour_cos',
        'fl_nvi_z',
        'fl_ret_lag_4',
        'fl_squeeze_mom',
        'fl_session_london',
        'fl_volume_z',
        'fl_parkinson_20',
        'fl_dow_cos',
        'MTF_30min_trend',
        'MTF_30min_donchian_trend',
        'fl_pct_from_high_20',
        'fl_ichi_senA_vs_senB',
        'fl_rv_ratio_5_20',
        'fl_pct_from_low_20',
        'fl_ret_lag_5',
        'MTF_15min_adx',
        'fl_range_vs_atr5',
        'fl_adx14',
        'fl_hour_sin',
        'fl_gap_open_atr',
        'fl_cmf_20',
        'MTF_30min_adx',
        'MTF_30min_time_in_trend',
        'MTF_15min_time_in_trend',
        'fl_upper_wick',
        'MTF_5min_adx',
        'fl_lower_wick',
        'fl_log_volume',
        'fl_donchian_trend_20',
        'MTF_30min_hh',
        'fl_ret_lag_1',
        'fl_cdl_evening_star',
        'fl_cdl_piercing',
        'fl_session_overlap',
        'fl_inside_bar',
        'fl_cdl_inv_hammer',
        'fl_session_tokyo',
        'MTF_15min_donchian_trend',
        ]
        if c in df.columns
    ]
    # Retain OHLCV + Time columns alongside features
    _ohlcv = [c for c in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'sell_y', 'buy_y']
              if c in df.columns]
    _final = _ohlcv + [c for c in _keep if c not in _ohlcv]
    return df[_final]


def _add_features_US500(df: pd.DataFrame, include_mtf: bool = False, regime_params: dict = None) -> pd.DataFrame:
    """
    Add the features selected for US500 by the Feature ML Lab.
    Generated automatically — edit with care.

    Pass regime_params to enable the causal market regime feature (fl_regime).

    Selected features (113 total):
        fl_regime, fl_rsi_14, fl_close_vs_ema8, fl_aroon_osc_14, fl_hl_range_atr,
    fl_stochrsi_14, fl_momentum_vote, fl_hma20_vs_ema21, fl_pct_from_high_60,
    MTF_15min_ll, fl_macd_hist_norm, fl_price_loc_128, fl_stoch_kd_diff,
    fl_donchian_trend_60, fl_body_atr, fl_range_contract, fl_stoch_k,
    fl_ema_vote, MTF_5min_trend, fl_donchian_trend_5, fl_pct_from_low_60,
    fl_macd_hist_slope_5, MTF_5min_ema_dist, MTF_15min_hh, fl_rsi14_slope_3,
    fl_di_minus, fl_rv_10, fl_bb_width_20, MTF_5min_ll, MTF_30min_ll,
    fl_di_plus, fl_obv_slope_10, fl_session_london, MTF_5min_hh, fl_H_rel,
    fl_cdl_harami, MTF_30min_hh, fl_roc_10, MTF_5min_slope, fl_ret_5,
    fl_donchian_pressure_60, fl_force_index_13, fl_price_loc_256,
    MTF_15min_ema_dist, fl_ichi_tk_diff, fl_er_60, fl_pct_from_high_20,
    fl_time_in_trend_20, MTF_5min_donchian_trend, fl_rv_ratio_5_20,
    fl_session_ny, fl_ema50_200_diff, fl_body_to_range, fl_range_vs_atr5,
    fl_ret_lag_2, fl_lower_wick, fl_volume_z, fl_outside_bar, fl_dow_cos,
    fl_ret_3, fl_er_14, fl_close_loc, fl_pct_from_low_20, fl_hour_cos, fl_adx14,
    MTF_15min_roc, fl_squeeze_mom, fl_mfi_14, fl_rv_ratio_20_60, MTF_5min_adx,
    fl_log_volume, fl_atr_ratio_14, fl_session_overlap, fl_L_rel,
    MTF_15min_time_in_trend, fl_hl_range, MTF_30min_adx, fl_session_tokyo,
    fl_cdl_engulf_bull, fl_cdl_inv_hammer, fl_donchian_trend_20,
    fl_cdl_morning_star, MTF_15min_trend, MTF_15min_donchian_trend,
    MTF_30min_donchian_trend, fl_ret_60, fl_ret_20, fl_pp_pos,
    fl_ichi_senA_vs_senB, MTF_30min_rsi, fl_er_30, fl_time_in_trend_60,
    fl_hour_sin, MTF_30min_time_in_trend, fl_cmf_20, fl_vol_direction,
    MTF_15min_adx, fl_rv_60, fl_nvi_z, fl_parkinson_20, fl_upper_wick,
    fl_run_len, fl_cdl_piercing, fl_ret_lag_4, fl_ret_lag_3, fl_squeeze,
    fl_cdl_doji, fl_cdl_shooting_str, fl_cdl_hammer, fl_cdl_evening_star,
    fl_dow_sin, fl_inside_bar, MTF_30min_trend
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df = add_feature_library(df, include_mtf=include_mtf, regime_params=regime_params)
    df = df.copy()
    _keep = [
        c for c in [
        'fl_regime',
        'fl_rsi_14',
        'fl_close_vs_ema8',
        'fl_aroon_osc_14',
        'fl_hl_range_atr',
        'fl_stochrsi_14',
        'fl_momentum_vote',
        'fl_hma20_vs_ema21',
        'fl_pct_from_high_60',
        'MTF_15min_ll',
        'fl_macd_hist_norm',
        'fl_price_loc_128',
        'fl_stoch_kd_diff',
        'fl_donchian_trend_60',
        'fl_body_atr',
        'fl_range_contract',
        'fl_stoch_k',
        'fl_ema_vote',
        'MTF_5min_trend',
        'fl_donchian_trend_5',
        'fl_pct_from_low_60',
        'fl_macd_hist_slope_5',
        'MTF_5min_ema_dist',
        'MTF_15min_hh',
        'fl_rsi14_slope_3',
        'fl_di_minus',
        'fl_rv_10',
        'fl_bb_width_20',
        'MTF_5min_ll',
        'MTF_30min_ll',
        'fl_di_plus',
        'fl_obv_slope_10',
        'fl_session_london',
        'MTF_5min_hh',
        'fl_H_rel',
        'fl_cdl_harami',
        'MTF_30min_hh',
        'fl_roc_10',
        'MTF_5min_slope',
        'fl_ret_5',
        'fl_donchian_pressure_60',
        'fl_force_index_13',
        'fl_price_loc_256',
        'MTF_15min_ema_dist',
        'fl_ichi_tk_diff',
        'fl_er_60',
        'fl_pct_from_high_20',
        'fl_time_in_trend_20',
        'MTF_5min_donchian_trend',
        'fl_rv_ratio_5_20',
        'fl_session_ny',
        'fl_ema50_200_diff',
        'fl_body_to_range',
        'fl_range_vs_atr5',
        'fl_ret_lag_2',
        'fl_lower_wick',
        'fl_volume_z',
        'fl_outside_bar',
        'fl_dow_cos',
        'fl_ret_3',
        'fl_er_14',
        'fl_close_loc',
        'fl_pct_from_low_20',
        'fl_hour_cos',
        'fl_adx14',
        'MTF_15min_roc',
        'fl_squeeze_mom',
        'fl_mfi_14',
        'fl_rv_ratio_20_60',
        'MTF_5min_adx',
        'fl_log_volume',
        'fl_atr_ratio_14',
        'fl_session_overlap',
        'fl_L_rel',
        'MTF_15min_time_in_trend',
        'fl_hl_range',
        'MTF_30min_adx',
        'fl_session_tokyo',
        'fl_cdl_engulf_bull',
        'fl_cdl_inv_hammer',
        'fl_donchian_trend_20',
        'fl_cdl_morning_star',
        'MTF_15min_trend',
        'MTF_15min_donchian_trend',
        'MTF_30min_donchian_trend',
        'fl_ret_60',
        'fl_ret_20',
        'fl_pp_pos',
        'fl_ichi_senA_vs_senB',
        'MTF_30min_rsi',
        'fl_er_30',
        'fl_time_in_trend_60',
        'fl_hour_sin',
        'MTF_30min_time_in_trend',
        'fl_cmf_20',
        'fl_vol_direction',
        'MTF_15min_adx',
        'fl_rv_60',
        'fl_nvi_z',
        'fl_parkinson_20',
        'fl_upper_wick',
        'fl_run_len',
        'fl_cdl_piercing',
        'fl_ret_lag_4',
        'fl_ret_lag_3',
        'fl_squeeze',
        'fl_cdl_doji',
        'fl_cdl_shooting_str',
        'fl_cdl_hammer',
        'fl_cdl_evening_star',
        'fl_dow_sin',
        'fl_inside_bar',
        'MTF_30min_trend',
        ]
        if c in df.columns
    ]
    # Retain OHLCV + Time columns alongside features
    _ohlcv = [c for c in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'sell_y', 'buy_y']
              if c in df.columns]
    _final = _ohlcv + [c for c in _keep if c not in _ohlcv]
    return df[_final]
