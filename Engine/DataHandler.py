import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import time
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

@dataclass
class Order:
    symbol: str
    side: str   # 'buy' or 'sell'
    entry: float
    qty: int
    entry_time: str
    expiration: int
    sl: float
    tp: float

class DataHandler:
    """
    Used mainly for backtesting
    """
    def __init__(self, df):
        self.data = df.rename(columns={'Date': 'Time'}, errors='ignore')
        self.long_position = False
        self.short_position = False

    def get_next_bar(self):
        for row in self.data.itertuples():
            yield row

    # Aggregate data to create a daily resolution chart
    def to_daily(self, inplace=False):
        daily = self.data.copy()
        grouped = daily.groupby(daily['Time'].dt.date)
        result = pd.DataFrame({
            'Open': grouped['Open'].first(),
            'High': grouped['High'].max(),
            'Low': grouped['Low'].min(),
            'Close': grouped['Close'].last()
        })
        result.index.name = 'Time'
        if inplace:
            self.data = result.reset_index()
            return None
        return result

    # Plot OHLC data using plotly
    def plot_ohlc(self, title="OHLC Chart"):
        data = self.data
        fig = go.Figure(data=[go.Candlestick(
            x=data['Time'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_type='category',  # Treat x-axis as categorical to remove gaps
            xaxis=dict(showticklabels=False, rangeslider=dict(visible=False))  # Remove x-axis labels and slider
        )
        
        fig.show()




class MT5DataHandler:
    """
    Simple MT5-backed DataHandler providing a get_next_bar() generator so it
    can be used interchangeably with the CSV `DataHandler` already in this
    project.

    Usage (replay mode):
        dh = MT5DataHandler(symbol='EURUSD', timeframe='M1', mode='replay', start='2025-10-01', end='2025-10-05')
        for bar in dh.get_next_bar():
            ...

    Live mode is a simple poller that yields a synthetic 1-tick 'bar' with
    Close equal to the latest tick price. This is intentionally minimal —
    for production you'd want a separate tick pipeline and proper bar
    aggregation.
    """

    TF_MAP = {
        '1min': lambda: mt5.TIMEFRAME_M1 if mt5 is not None else None,
        'M1': lambda: mt5.TIMEFRAME_M1 if mt5 is not None else None,
        '5min': lambda: mt5.TIMEFRAME_M5 if mt5 is not None else None,
        'M5': lambda: mt5.TIMEFRAME_M5 if mt5 is not None else None,
        '15min': lambda: mt5.TIMEFRAME_M15 if mt5 is not None else None,
        'M15': lambda: mt5.TIMEFRAME_M15 if mt5 is not None else None,
        '1h': lambda: mt5.TIMEFRAME_H1 if mt5 is not None else None,
        'H1': lambda: mt5.TIMEFRAME_H1 if mt5 is not None else None,
        '1d': lambda: mt5.TIMEFRAME_D1 if mt5 is not None else None,
        'D1': lambda: mt5.TIMEFRAME_D1 if mt5 is not None else None,
    }

    def __init__(self, symbol: str = 'EURUSD', timeframe: str = '1min', mode: str = 'replay', start: str = None, end: str = None, max_bars: int = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.start = pd.to_datetime(start) if start is not None else None
        self.end = pd.to_datetime(end) if end is not None else None
        self.max_bars = max_bars
        # Timezone offset detected during live polling (in seconds)
        self.tz_offset_seconds = 0

        if mt5 is None:
            raise RuntimeError('MetaTrader5 package not available. Install with `pip install MetaTrader5`')

        # Initialize the connection to the local MT5 terminal. If it is not
        # running this will return False; callers should ensure MT5 terminal
        # is running and logged in.
        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

        # choose timeframe constant
        tf_func = self.TF_MAP.get(self.timeframe)
        if tf_func is None:
            # allow passing numeric mt5 constant directly
            try:
                self.mt5_timeframe = int(self.timeframe)
            except Exception:
                raise ValueError(f"Unknown timeframe: {self.timeframe}")
        else:
            self.mt5_timeframe = tf_func()

        # load bars for replay mode immediately
        self.data = None
        if self.mode == 'replay':
            self._load_historical()

    def _load_historical(self):
        # Fetch bar history from MT5 between start and end. If end is None
        # we fetch up to now. We use copy_rates_range when both start and end
        # are provided, or copy_rates_from otherwise.
        if self.start is None:
            raise ValueError('start must be provided for replay mode')

        start_dt = pd.to_datetime(self.start).to_pydatetime()
        end_dt = pd.to_datetime(self.end).to_pydatetime() if self.end is not None else datetime.now()

        # Request rates
        rates = None
        try:
            rates = mt5.copy_rates_range(self.symbol, self.mt5_timeframe, start_dt, end_dt)
        except Exception:
            rates = None

        if rates is None or len(rates) == 0:
            # fall back to copy_rates_from if range returned empty
            count = self.max_bars or 7000
            rates = mt5.copy_rates_from(self.symbol, self.mt5_timeframe, end_dt, count)

        if rates is None or len(rates) == 0:
            raise RuntimeError('No historical bars returned from MT5 for the requested range')

        df = pd.DataFrame(rates)
        # mt5 returns epoch seconds in 'time'
        df['Time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        # keep the column names compatible with existing DataHandler (Time, Open, High, Low, Close, Volume)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        df = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        # ensure sorted
        df = df.sort_values('Time').reset_index(drop=True)
        self.data = df

    def get_next_bar(self):
        """Yield bars in the same format as the CSV DataHandler (itertuples())."""
        if self.mode == 'replay':
            if self.data is None:
                self._load_historical()
            for row in self.data.itertuples():
                yield row

        elif self.mode == 'live':
            # Yield only completed 1-minute bars at bar close.
            # Use copy_rates_from_pos to fetch bars from current position backwards.
            # This is more reliable than time-based anchoring which may have timezone issues.
            poll_interval = 1.0
            last_yield_time = None
            # map common MT5 timeframe constants to seconds
            TF_SECONDS = {
                getattr(mt5, 'TIMEFRAME_M1', 0): 60,
                getattr(mt5, 'TIMEFRAME_M5', 0): 300,
                getattr(mt5, 'TIMEFRAME_M15', 0): 900,
                getattr(mt5, 'TIMEFRAME_H1', 0): 3600,
                getattr(mt5, 'TIMEFRAME_D1', 0): 86400,
            }
            timeframe_seconds = TF_SECONDS.get(self.mt5_timeframe, 60)

            # Compute timezone offset on first poll by comparing the most recent bar's
            # epoch time with its expected position. If the bar's epoch seems shifted
            # by whole hours, apply the correction to all subsequent bars.
            tz_offset_seconds = 0
            tz_offset_computed = False

            while True:
                # Get current UTC time for completion check
                now = pd.to_datetime(datetime.utcnow(), utc=True)
                
                # Refresh rates to encourage the terminal to update
                try:
                    mt5.refresh_rates(self.symbol)
                except Exception:
                    pass
                
                rates = None
                try:
                    # Fetch the most recent N bars starting from position 0 (current bar) backwards.
                    # Position 0 is the most recent bar; we fetch backwards.
                    # INCREASED to 7,000 bars to support MTF feature stability in strategies
                    # This provides ~7 days of M1 data for proper indicator warm-up
                    rates = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, 7_000)
                except Exception:
                    rates = None

                if rates is None or len(rates) == 0:
                    time.sleep(poll_interval)
                    continue

                # Compute timezone offset on first successful fetch
                if not tz_offset_computed and len(rates) > 0:
                    try:
                        # The most recent bar (index -1) should be close to 'now'.
                        # If its epoch time is significantly offset, compute the correction.
                        most_recent_rate = rates[-1]
                        raw_epoch = float(most_recent_rate['time'])
                        expected_epoch = time.time()
                        diff = expected_epoch - raw_epoch
                        # Round to nearest hour (3600 seconds) to detect timezone shifts
                        hours_diff = round(diff / 3600.0)
                        if abs(hours_diff) > 0:
                            tz_offset_seconds = hours_diff * 3600
                            self.tz_offset_seconds = tz_offset_seconds
                            print(f"[MT5DataHandler] Detected timezone offset: {hours_diff} hours ({tz_offset_seconds} seconds)")
                    except Exception:
                        pass
                    tz_offset_computed = True

                # rates is an array-like of bars; iterate in chronological order (oldest first)
                try:
                    # copy_rates_from_pos returns oldest-first, so we process in order
                    sorted_rates = list(rates)
                except Exception:
                    sorted_rates = list(rates)

                for rate in sorted_rates:
                    # MT5 returns bar times as epoch seconds. Apply computed timezone offset
                    # to align with UTC if the terminal uses local/server time.
                    raw_epoch = float(rate['time'])
                    adjusted_epoch = raw_epoch + tz_offset_seconds
                    rate_time = pd.to_datetime(adjusted_epoch, unit='s', utc=True)
                    # only consider bars we haven't yielded yet
                    if last_yield_time is not None and rate_time <= last_yield_time:
                        continue
                    # check whether the bar has completed (start + timeframe <= now)
                    if rate_time + timedelta(seconds=timeframe_seconds) <= now:
                        # construct DataFrame row to reuse existing tuple format
                        row = pd.DataFrame([{
                            'Time': rate_time,
                            'Open': rate['open'],
                            'High': rate['high'],
                            'Low': rate['low'],
                            'Close': rate['close'],
                            'Volume': rate['tick_volume']
                        }])
                        for r in row.itertuples():
                            yield r
                        last_yield_time = rate_time

                time.sleep(poll_interval)

        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def shutdown(self):
        try:
            mt5.shutdown()
        except Exception:
            pass