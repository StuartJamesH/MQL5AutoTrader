"""
DataHandler module.

Provides data structures and market-data feeds for both backtesting and live
trading via a local MetaTrader 5 terminal.

Classes
-------
Order
    Lightweight dataclass representing a single trade order.
DataHandler
    CSV-backed handler used for backtesting.
MT5DataHandler
    MT5-backed handler supporting historical *replay* and real-time *live*
    bar delivery.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover – package absent in non-MT5 environments
    mt5 = None

_LOG = logging.getLogger(__name__)

@dataclass
class Order:
    """Represents a single trade order.

    Attributes
    ----------
    symbol : str
        Instrument ticker (e.g. ``'EURUSD'``).
    side : str
        Direction – ``'buy'`` or ``'sell'``.
    entry : float
        Requested entry price (``0.0`` for market orders).
    qty : int
        Order volume in lots.
    entry_time : str
        ISO-8601 timestamp at which the order was created or filled.
    expiration : datetime or None
        UTC datetime at which the pending order expires; ``None`` means
        Good-Till-Cancelled.
    sl : float
        Stop-loss price (``0.0`` disables stop-loss).
    tp : float
        Take-profit price (``0.0`` disables take-profit).
    """

    symbol: str
    side: str           # 'buy' | 'sell'
    entry: float
    qty: int
    entry_time: str
    expiration: Optional[datetime]  # UTC datetime; None means GTC
    sl: float
    tp: float

class DataHandler:
    """CSV-backed data handler, primarily used for backtesting.

    Wraps a pandas DataFrame of OHLC bars and exposes a ``get_next_bar()``
    generator that yields rows as ``itertuples()`` namedtuples, making it
    interchangeable with :class:`MT5DataHandler` inside
    :class:`~Engine.Live_Engine`.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.  A ``'Date'`` column is renamed to ``'Time'``
        automatically if present.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.data: pd.DataFrame = df.rename(columns={'Date': 'Time'}, errors='ignore')
        self.long_position: bool = False
        self.short_position: bool = False

    def get_next_bar(self):
        """Yield OHLC bars one at a time as ``itertuples`` namedtuples."""
        for row in self.data.itertuples():
            yield row

    def to_daily(self, inplace: bool = False) -> Optional[pd.DataFrame]:
        """Resample intraday data to daily OHLC bars.

        Parameters
        ----------
        inplace : bool, optional
            If ``True``, replace ``self.data`` with the daily DataFrame and
            return ``None``.  If ``False`` (default), return the daily
            DataFrame without modifying ``self.data``.

        Returns
        -------
        pd.DataFrame or None
        """
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

    def plot_ohlc(self, title: str = "OHLC Chart") -> None:
        """Render an interactive candlestick chart using Plotly.

        Parameters
        ----------
        title : str, optional
            Chart title shown at the top of the figure.
        """
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
            xaxis_type='category',  # categorical axis removes weekend/holiday gaps
            xaxis=dict(showticklabels=False, rangeslider=dict(visible=False))
        )

        fig.show()




class MT5DataHandler:
    """MT5-backed data handler providing a ``get_next_bar()`` generator.

    Drop-in replacement for :class:`DataHandler` when connected to a live
    MetaTrader 5 terminal.  Two operating modes are supported:

    * **replay** – fetches a historical date range from MT5 on construction
      and replays the bars in order (useful for walk-forward testing with a
      live connection).
    * **live** – polls MT5 continuously and yields each *completed* bar at
      bar-close time.  7 000 bars are fetched on every poll cycle to give
      multi-timeframe strategies sufficient look-back for indicator warm-up.

    Parameters
    ----------
    symbol : str
        MT5 instrument ticker, e.g. ``'EURUSD'``.
    timeframe : str
        Timeframe string (``'M1'``, ``'M5'``, ``'M15'``, ``'H1'``, ``'D1'``,
        or equivalently ``'1min'``, ``'5min'``, etc.).  A raw MT5 timeframe
        integer constant may also be passed as a string.
    mode : str
        ``'replay'`` or ``'live'``.
    start : str, optional
        Start date/time for replay mode (ISO-8601 or any format accepted by
        ``pd.to_datetime``).  Required when *mode* is ``'replay'``.
    end : str, optional
        End date/time for replay mode.  Defaults to the current time when
        omitted.
    max_bars : int, optional
        Maximum bars to fetch when falling back from ``copy_rates_range`` to
        ``copy_rates_from`` in replay mode.  Defaults to ``7 000``.

    Raises
    ------
    RuntimeError
        If the ``MetaTrader5`` package is unavailable or ``mt5.initialize()``
        fails.
    ValueError
        If an unrecognised *timeframe* string is supplied.
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

    def __init__(self, symbol: str = 'EURUSD', timeframe: str = '1min', mode: str = 'replay', start: Optional[str] = None, end: Optional[str] = None, max_bars: Optional[int] = None) -> None:
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

    def _load_historical(self) -> None:
        """Fetch historical bars from MT5 and store them in ``self.data``.

        Uses ``copy_rates_range`` when both *start* and *end* are set, falling
        back to ``copy_rates_from`` if the range query returns no data.
        """
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
            count = self.max_bars or 10_000
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
        """Yield OHLC bars in the same namedtuple format as :class:`DataHandler`.

        In *replay* mode the generator is exhausted once all historical bars
        have been yielded.  In *live* mode the generator runs indefinitely,
        blocking between poll cycles.
        """
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
                            _LOG.info(
                                "Detected timezone offset: %d hours (%d seconds)",
                                hours_diff,
                                tz_offset_seconds,
                            )
                    except Exception:
                        pass
                    tz_offset_computed = True

                # rates is an array-like of bars; iterate in chronological order (oldest first)
                # copy_rates_from_pos returns oldest-first; process chronologically
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

    def shutdown(self) -> None:
        """Cleanly shut down the connection to the MT5 terminal."""
        try:
            mt5.shutdown()
        except Exception:
            pass