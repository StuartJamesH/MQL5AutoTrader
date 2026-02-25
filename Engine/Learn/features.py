import pandas as pd
import talib
import numpy as np
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
    Donchian Trend (PineScript-equivalent)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'High', 'Low', 'Close'
    length : int
        Donchian Channel lookback period

    Returns
    -------
    pd.Series
        Trend series:
        +1 = uptrend
        -1 = downtrend
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
    """
    Calculate time spent in current trend.

    Parameters
    ----------
    trend_series : pd.Series
        Series containing trend values (+1, -1)

    Returns
    -------
    pd.Series
        Series with time spent in current trend
    """
    time_in_trend = np.zeros(len(trend_series), dtype=int)

    for i in range(1, len(trend_series)):
        if trend_series.iloc[i] == trend_series.iloc[i-1]:
            time_in_trend[i] = time_in_trend[i-1] + 1
        else:
            time_in_trend[i] = 1

    return pd.Series(time_in_trend, index=trend_series.index, name="time_in_trend")

def checkhl(data_back, data_forward, hl):
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

def WMA(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def HMA(series, timeperiod):
    # Step 1: WMA(period/2) * 2
    wma_half = WMA(series, timeperiod // 2) * 2

    # Step 2: WMA(period)
    wma_full = WMA(series, timeperiod)

    # Step 3: WMA(sqrt(period)) on the difference
    diff = wma_half - wma_full
    return WMA(diff, int(np.sqrt(timeperiod)))

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


def add_multitimeframe_features(df, timeframes=['5min', '15min', '30min', '60min'], causal=True):
    """
    Add features from higher timeframes to capture longer-term trends.
    
    Parameters:
    -----------
    df : DataFrame with Time, Open, High, Low, Close, Volume columns
    timeframes : list of pandas resample strings (e.g., '5min' = 5 minutes)
    
    Returns:
    --------
    DataFrame with additional MTF (multi-timeframe) features
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
        
        # Calculate key indicators on higher timeframe
        # 1. Trend direction (EMA crossover)
        ema_fast = talib.EMA(df_tf['Close'], timeperiod=8)
        ema_slow = talib.EMA(df_tf['Close'], timeperiod=21)
        df_tf[f'MTF_{tf}_trend'] = np.sign(ema_fast - ema_slow)  # +1 uptrend, -1 downtrend
        
        # 2. Trend strength (ADX)
        df_tf[f'MTF_{tf}_adx'] = talib.ADX(df_tf['High'], df_tf['Low'], df_tf['Close'], timeperiod=14) / 100
        
        # 3. Price momentum (ROC - Rate of Change)
        df_tf[f'MTF_{tf}_roc'] = talib.ROC(df_tf['Close'], timeperiod=10) / 100
        
        # 4. RSI for overbought/oversold on higher TF
        df_tf[f'MTF_{tf}_rsi'] = talib.RSI(df_tf['Close'], timeperiod=14) / 100
        
        # 5. Distance from EMA (normalized)
        atr_tf = talib.ATR(df_tf['High'], df_tf['Low'], df_tf['Close'], timeperiod=14)
        df_tf[f'MTF_{tf}_ema_dist'] = (df_tf['Close'] - ema_slow) / (atr_tf + 1e-9)
        
        # 6. Slope of higher timeframe
        # IMPORTANT: keep the slope window fixed; do not make it a function of
        # dataset length, otherwise bulk vs live computations diverge.
        df_tf[f'MTF_{tf}_slope'] = rolling_slope_logprice(df_tf['Close'], window=10)
        
        # 7. Higher high / Lower low detection
        df_tf[f'MTF_{tf}_hh'] = (df_tf['High'] >= df_tf['High'].rolling(5).max().shift(1)).astype(int)
        df_tf[f'MTF_{tf}_ll'] = (df_tf['Low'] <= df_tf['Low'].rolling(5).min().shift(1)).astype(int)

        # 8. Donchian Trend and Time in Trend
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

def add_all_features(df, lookback=8, vol_window=20, include_mtf=True, regime_params=None):

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

    # Rolling Slopes
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

    # De-emphasize magnitude-only features by removing absolute-only proxies

    # Z-Scores (as from signal generation)
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
    # df['bb_dist'] = (df['Close'] - bb_mid) / (df['bb_width'] + 1e-6)

    # Additional One-Hot Features
    df['OH_CCI'] = [1 if x>=100 else -1 if x<=-100 else 0 for x in talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)]

    ## Price based indicators
    for ind in [talib.EMA]:
        for period in [8,21,50,128]:
            df[f'PR_{ind.__name__}_{period}'] = ind(df['Close'], period)

    df[f'OH_LOWEST_LOW_{lookback}'] = (df['Low'] == df['Low'].rolling(lookback, min_periods=1).min()).astype(int)
    df[f'OH_HIGHEST_HIGH_{lookback}'] = (df['High'] == df['High'].rolling(lookback, min_periods=1).max()).astype(int)

    ## Oscilators and other misc stuff
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

    # Figure out how to scale later
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # df['OBV'] = talib.OBV(df['Close'], df['Volume'])
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

    # Final trend_score: combine ADX (already scaled /100 above) and normalized slope magnitude
    # If ADX computed earlier as a float between 0-1, rescale to 0-100 for intuitive thresholds
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

def add_selected_features(df, lookback=8, vol_window=20, include_mtf=True, regime_params=None):

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

    # De-emphasize magnitude-only features by removing absolute-only proxies
    # Z-Scores (as from signal generation)
    df["mean"] = df["Close"].rolling(14).mean()
    df["std"]  = df["Close"].rolling(14).std()
    df["z"] = (df["Close"] - df["mean"]) / df["std"]
    df['OH_z_flag1'] = [1 if z > 1 else 0 for z in df['z']]
    df['OH_z_flag2'] = [1 if z < -1 else 0 for z in df['z']]

    # Additional One-Hot Features
    df['OH_CCI'] = [1 if x>=100 else -1 if x<=-100 else 0 for x in talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)]

    ## Price based indicators
    for ind in [talib.EMA]:
        for period in [21]:
            df[f'PR_{ind.__name__}_{period}'] = ind(df['Close'], period)

    df[f'OH_LOWEST_LOW_{lookback}'] = (df['Low'] == df['Low'].rolling(lookback, min_periods=1).min()).astype(int)
    df[f'OH_HIGHEST_HIGH_{lookback}'] = (df['High'] == df['High'].rolling(lookback, min_periods=1).max()).astype(int)

    ## Oscilators and other misc stuff
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)/100

    df['StochK'], df['StochD'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['StochK'] = df['StochK']/100
    df['StochD'] = df['StochD']/100

    # Volume regime features (works with tick volume too)
    df['log_volume'] = np.log1p(df['Volume'].astype(float))

    # Final trend_score: combine ADX (already scaled /100 above) and normalized slope magnitude
    # If ADX computed earlier as a float between 0-1, rescale to 0-100 for intuitive thresholds
    
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

def add_price_features(df):

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

    return df 