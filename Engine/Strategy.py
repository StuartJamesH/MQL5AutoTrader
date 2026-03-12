from __future__ import annotations

import csv
import datetime
import os
from collections import deque
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import talib
import torch

from DataHandler import Order

if TYPE_CHECKING:
    from TicketBook import TicketBook

class TripleBarrier:
    def __init__(self, symbol, model, model_pack, patience, risk=50, ema1_period=8, ema2_period=30, mt5_executor=None, data_handler=None, maxpos=0.5, debug=True, ticket_book: Optional["TicketBook"] = None):
        self.symbol = symbol
        self.order_type = 'market'

        self.signal = 0 # If strategy is primed to enter trade
        self.maxpos = maxpos
        self.patience = patience # How many bars to wait before giving up on a trade
        self.countdown = 0 # Countdown timer for trade entry
        self.debug = debug

        # Init price data deques
        # INCREASED from 350 to 1200: ensures multi-timeframe indicators (30min EMA) have sufficient history
        # Calculation: 30min EMA(21) needs 21*30=630 bars + seq_len(256) + buffer = ~1200 bars minimum
        self.maxlen = 1200
        self.t = deque(maxlen=self.maxlen) # Time
        self.o = deque(maxlen=self.maxlen) # Open
        self.h = deque(maxlen=self.maxlen) # High
        self.l = deque(maxlen=self.maxlen) # Low
        self.c = deque(maxlen=self.maxlen) # Close
        self.v = deque(maxlen=self.maxlen) # Volume

        # Store temporary trade info
        self.order = None
        self.position = 0
        self.entry = 0
        self.stop = 0
        self.take = 0

        self.risk = risk
        self.ema1_period = ema1_period
        self.ema2_period = ema2_period

        # Unpack model info
        self.model = model
        self.model_pack = model_pack
        self.model_info = model_pack['model_info']
        self.preprocess = model_pack['preprocess_functions']
        self.preprocess_args = model_pack['preprocess_args']
        self.preprocess_args['target_col'] = None
        self.target_col = model_pack['target_col'] if 'target_col' in model_pack else None
        self.scaler = model_pack['scaler']
        self.features = model_pack['feature_functions']
        self.seq_len = self.model_info['seq_len']
        # MT5 executor (optional) used to submit/cancel pending orders and query fills
        self.mt5_executor = mt5_executor
        # Data handler reference to access detected timezone offset
        self.data_handler = data_handler
        # Track a single pending ticket for this strategy instance (None when no pending)
        self.pending_order_ticket = None
        # Recorded fills from MT5: list[Order]
        self.fills = []
        # Last known signal value (helps detect transitions)
        self.last_signal = 0
        # TicketBook for state queries (pending / open position checks)
        self.ticket_book = ticket_book

    def get_moving_averages(self):
        """
        Calculates two EMA's to use for entry filtering
        """
        if len(self.t) == self.maxlen:
            c = pd.Series(list(self.c))
            ema1 = c.ewm(span=self.ema1_period, adjust=False).mean().iloc[-1]
            ema2 = c.ewm(span=self.ema2_period, adjust=False).mean().iloc[-1]
            return ema1, ema2
        else:
            return None, None

    def check_pending_orders(self) -> bool:
        """
        Return True if there is an active pending order for this symbol.
        State is read from the TicketBook; no MT5 call is made.
        """
        if self.ticket_book is not None:
            return self.ticket_book.has_pending_order(self.symbol)
        return False

    def check_open_positions(self) -> bool:
        """
        Return True if there is an open (filled) position for this symbol.
        State is read from the TicketBook; no MT5 call is made.
        """
        if self.ticket_book is not None:
            return self.ticket_book.has_open_position(self.symbol)
        return False
    
    def make_prediction(self):
        """
        Helper function: Make a prediction based on the current price data.
        Entry is currently set to market price (close of most recently printed candle).
        """

        if len(self.t) == self.maxlen:
            df = pd.DataFrame(
                data={
                    'Time': self.t,
                    'Open': self.o,
                    'High': self.h,
                    'Low': self.l,
                    'Close': self.c,
                    'Volume': self.v
                }
            )
            df = df.sort_values('Time').reset_index(drop=True)

            # Stop/Take calculations
            _atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            atr = _atr.values[-1]
            too_volatile = atr*100_000 < 25  # ATR threshold for volatility filter

            df = self.features(df)
            X, _, _, _ = self.preprocess(df, scaler=self.scaler, **self.preprocess_args)

            with torch.no_grad(): # Make prediction
                X_input = X[-self.seq_len:, :].reshape(1, self.seq_len, X.shape[1])
                input_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)
                output = self.model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                pred_map = {0: -1, 1: 0, 2: 1}
                signal = pred_map[prediction]
                if self.debug:
                    print(f'\n[[DEBUG PREDICTION]]: Prediction: {prediction}, Signal: {signal}')

            if signal == 1 and not too_volatile: # Long entry predicted
                side = 'buy'
                entry = self.c[-1]
                take = self.c[-1] + (4 * atr)
                stop = self.c[-1] - (2 * atr)
                position_size = (self.risk / abs(entry - stop)) / 100_000

            elif signal == -1 and not too_volatile: # Short entry predicted
                side = 'sell'
                entry = self.c[-1]
                take = self.c[-1] - (4 * atr)
                stop = self.c[-1] + (2 * atr)
                position_size = (self.risk / abs(stop - entry)) / 100_000

            else: # No entry predicted
                signal = 0
                entry = 0
                side = None
                take = 0
                stop = 0
                position_size = 0
        else: # Do nothing if prediction not possible
            signal = 0
            entry = 0
            side = None
            stop = 0
            take = 0
            position_size = 0
        
        position_size = min(position_size, self.maxpos)
        return signal, side, round(entry, 5), round(stop, 5), round(take, 5), round(position_size, 2)

    def on_bar(self, bar):

        orders = []

        # Append raw 1 minute bars to class dataset
        self.t.append(pd.to_datetime(bar.Time))
        self.o.append(bar.Open)
        self.h.append(bar.High)
        self.l.append(bar.Low)
        self.c.append(bar.Close)
        self.v.append(bar.Volume)
    
        # Check mt5 for pending orders
        pending_order = self.check_pending_orders()
        open_position = self.check_open_positions()

        # Decrement countdown timer if active
        if self.countdown > 0:
            self.countdown -= 1

        # Check & create new signal if below criteria is met:
        # A poisition is not open AND
        # No pending orders are active AND
        # No trading singal in place
        if not open_position and not pending_order:
            
            # Make new prediction
            self.signal, self.side, self.entry, self.stop, self.take, self.position_size = self.make_prediction()
            ema1, ema2 = self.get_moving_averages()

            # Act based on prediction, filter based on EMA cross
            if self.signal == 0:
                pass
            elif self.signal == 1: # Open a buy order and ema1 < ema2

                # Start the countdown timer. Countdown logic not used in this strategy
                self.countdown = self.patience

                # Send sell stop order to mt5 terminal
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    entry=self.entry,
                    qty=self.position_size,
                    entry_time=self.t[-1],
                    expiration=None,
                    sl=self.stop,
                    tp=self.take
                )
                orders.append(order)
                
            elif self.signal == -1: # Do not sell using current model

                # Start the countdown timer. Countdown logic not used in this strategy
                self.countdown = self.patience
                
                # Send buy stop order to mt5 terminal
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    entry=self.entry,
                    qty=self.position_size,
                    entry_time=self.t[-1],
                    expiration=None,
                    sl=self.stop,
                    tp=self.take
                )
                orders.append(order)
            else:
                pass
        return orders

class TripleBarrierHiLow:
    """
    Uses similar logic as TripleBarrier, but uses limit orders to time entries and reduce trades that go bad fast.
    """
    def __init__(self, symbol, model, model_pack, patience, risk=50, ema1_period=8, ema2_period=30, mt5_executor=None, data_handler=None, maxpos=0.5, debug=True, log=True, ticket_book: Optional["TicketBook"] = None):
        self.symbol = symbol
        self.order_type = 'stop'

        self.signal = 0 # If strategy is primed to enter trade
        self.maxpos = maxpos
        self.patience = patience # How many bars to wait before giving up on a trade
        self.countdown = 0 # Countdown timer for trade entry
        self.debug = debug

        # Init price data deques
        # CRITICAL: Buffer sizing for accurate MTF feature calculation
        # 
        # Multi-timeframe indicators need significant history:
        # - 30min EMA(21) needs 21 * 30 = 630 1-min bars just to START computing
        # - After that, it needs ~500+ more bars for the EMA to stabilize
        # - Plus seq_len (256) for the model input
        # - Plus buffer for dropna() which removes ~200-400 rows
        #
        # Set to 10,000 bars (~7 days of M1 data) for optimal MTF feature stability
        # This matches the MT5DataHandler fetch size and gives ~333 valid 30-min bars
        self.maxlen = 10_000
        self.t = deque(maxlen=self.maxlen) # Time
        self.o = deque(maxlen=self.maxlen) # Open
        self.h = deque(maxlen=self.maxlen) # High
        self.l = deque(maxlen=self.maxlen) # Low
        self.c = deque(maxlen=self.maxlen) # Close
        self.v = deque(maxlen=self.maxlen) # Volume

        # Store temporary trade info
        self.order = None
        self.position = 0
        self.entry = 0
        self.stop = 0
        self.take = 0

        self.risk = risk
        self.ema1_period = ema1_period
        self.ema2_period = ema2_period

        # Unpack model info
        self.model = model
        self.model_pack = model_pack
        self.model_info = model_pack['model_info']
        self.preprocess = model_pack['preprocess_functions']
        self.preprocess_args = model_pack['preprocess_args']
        self.preprocess_args['target_col'] = None
        self.target_col = model_pack['target_col'] if 'target_col' in model_pack else None
        self.scaler = model_pack['scaler']
        self.features = model_pack['feature_functions']
        self.seq_len = self.model_info['seq_len']
        # MT5 executor (optional) used to submit/cancel pending orders and query fills
        self.mt5_executor = mt5_executor
        # Data handler reference to access detected timezone offset
        self.data_handler = data_handler
        # Track a single pending ticket for this strategy instance (None when no pending)
        self.pending_order_ticket = None
        # Recorded fills from MT5: list[Order]
        self.fills = []
        # Last known signal value (helps detect transitions)
        self.last_signal = 0
        
        # Initialize logging
        self.log = log
        self.log_file = None
        self.csv_writer = None
        if self.log:
            self._initialize_logging()
        
        # Feature stability tracking - ensure indicators have warmed up
        self.features_ready = False
        # With 10,000 bar buffer, wait until we have at least 5,000 bars
        # This ensures MTF 30min features have 150+ periods to stabilize
        self.min_bars_for_features = 5_000

    def _initialize_logging(self):
        """Initialize CSV logging for predictions and trades"""
        # Create log directory if it doesn't exist
        log_dir = 'Engine/Learn/Trade Logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'{self.symbol}_{self.model_info["model_type"]}_log_{timestamp}.csv'
        log_path = os.path.join(log_dir, log_filename)
        
        # Open CSV file and create writer
        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Write header
        header = ['timestamp', 'bar_time', 'open', 'high', 'low', 'close', 'volume',
                  'prediction', 'signal', 'prob_short', 'prob_flat', 'prob_long',
                  'side', 'entry', 'stop', 'take', 
                  'position_size', 'atr', 'buffer_length', 'clean_rows',
                  'pending_order', 'open_position', 
                  'in_restricted_hours', 'action_taken']
        self.csv_writer.writerow(header)
        self.log_file.flush()
        
        print(f'[LOGGING] Initialized trade log: {log_path}')
    
    def _log_prediction(self, bar_time, open_price, high_price, low_price, close_price, volume,
                       prediction, signal, prob_short, prob_flat, prob_long,
                       side, entry, stop, take, position_size, 
                       atr, buffer_len, clean_rows, pending_order, open_position, in_restricted_hours, action_taken):
        """Log prediction details to CSV file"""
        if self.log and self.csv_writer:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            row = [timestamp, bar_time, open_price, high_price, low_price, close_price, volume,
                   prediction, signal, prob_short, prob_flat, prob_long,
                   side, entry, stop, take, 
                   position_size, atr, buffer_len, clean_rows, pending_order, open_position, 
                   in_restricted_hours, action_taken]
            self.csv_writer.writerow(row)
            self.log_file.flush()
    
    def __del__(self):
        """Cleanup: close log file when strategy is destroyed"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def get_moving_averages(self):
        """
        Calculates two EMA's to use for entry filtering
        """
        if len(self.t) == self.maxlen:
            c = pd.Series(list(self.c))
            ema1 = c.ewm(span=self.ema1_period, adjust=False).mean().iloc[-1]
            ema2 = c.ewm(span=self.ema2_period, adjust=False).mean().iloc[-1]
            return ema1, ema2
        else:
            return None, None

    def check_pending_orders(self) -> bool:
        """
        Return True if there is an active pending order for this symbol.
        State is read from the TicketBook; no MT5 call is made.
        """
        if self.ticket_book is not None:
            return self.ticket_book.has_pending_order(self.symbol)
        return False

    def check_open_positions(self) -> bool:
        """
        Return True if there is an open (filled) position for this symbol.
        State is read from the TicketBook; no MT5 call is made.
        """
        if self.ticket_book is not None:
            return self.ticket_book.has_open_position(self.symbol)
        return False
    
    def make_prediction(self, bar_time=None, pending_order=False, open_position=False, in_restricted_hours=False):
        """
        Helper function: Make a prediction based on the current price data.
        Entry is currently set to market price (close of most recently printed candle).
        """

        # Enhanced buffer checks for feature stability
        if len(self.t) < self.min_bars_for_features:
            if self.debug and len(self.t) % 100 == 0:  # Print every 100 bars during warmup
                print(f'[WARMUP] Collecting data: {len(self.t)}/{self.min_bars_for_features} bars')
            return 0, None, 0, 0, 0, 0

        if len(self.t) == self.maxlen and not self.features_ready:
            self.features_ready = True
            print(f'[READY] Feature buffer full ({self.maxlen} bars). Model ready for predictions.')

        if len(self.t) == self.maxlen:
            df = pd.DataFrame(
                data={
                    'Time': self.t,
                    'Open': self.o,
                    'High': self.h,
                    'Low': self.l,
                    'Close': self.c,
                    'Volume': self.v
                }
            )
            df = df.sort_values('Time').reset_index(drop=True)
            
            # Stop/Take calculations
            _atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            atr = _atr.values[-1]

            # Compute features on full buffer (matches Review Model v2 approach)
            df_feat = self.features(df)
            
            # Critical: Drop NaN rows to match training preprocessing
            # This ensures feature indices align correctly
            rows_before_dropna = len(df_feat)
            df_feat_clean = df_feat.dropna(how='any')
            rows_after_dropna = len(df_feat_clean)
            
            if self.debug and not hasattr(self, '_logged_feature_info'):
                # Log feature calculation info once
                print(f'\n[FEATURE DEBUG] Buffer: {len(df)} bars')
                print(f'[FEATURE DEBUG] Rows before dropna: {rows_before_dropna}')
                print(f'[FEATURE DEBUG] Rows after dropna: {rows_after_dropna}')
                print(f'[FEATURE DEBUG] Rows lost to NaN: {rows_before_dropna - rows_after_dropna}')
                print(f'[FEATURE DEBUG] Feature columns: {len(df_feat.columns)}')
                
                # Check which columns have the most NaNs
                nan_counts = df_feat.isna().sum()
                worst_cols = nan_counts.nlargest(5)
                print(f'[FEATURE DEBUG] Columns with most NaNs: {dict(worst_cols)}')
                self._logged_feature_info = True
            
            # Verify we have enough clean data after dropping NaNs
            if len(df_feat_clean) < self.seq_len + 50:  # Need at least seq_len + buffer
                if self.debug:
                    print(f'[WARNING] Insufficient clean features: {len(df_feat_clean)} rows after dropna()')
                return 0, None, 0, 0, 0, 0
            
            # Preprocess features (matching Review Model v2)
            X, y, _, _ = self.preprocess(df_feat_clean, scaler=self.scaler, **self.preprocess_args)
            
            # Verify preprocessing produced valid output
            if len(X) < self.seq_len:
                if self.debug:
                    print(f'[WARNING] Insufficient preprocessed data: {len(X)} < {self.seq_len}')
                return 0, None, 0, 0, 0, 0

            # Extract last sequence (matches Review Model v2 sliding window approach)
            with torch.no_grad():
                input_tensor = torch.tensor(X[-self.seq_len:], dtype=torch.float32).unsqueeze(0)
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                prediction = torch.argmax(output, dim=1).item()
                pred_map = {0: -1, 1: 0, 2: 1}
                signal = pred_map[prediction]
                
                # Store probabilities for logging
                prob_short = probs[0].item()
                prob_flat = probs[1].item()
                prob_long = probs[2].item()
                
                if self.debug:
                    print(f'\n[[DEBUG PREDICTION]]: Prediction: {prediction}, Signal: {signal}')
                    print(f'   Probabilities: Short={prob_short:.3f}, Flat={prob_flat:.3f}, Long={prob_long:.3f}')
                    print(f'   Feature shape: {X.shape}, Sequence used: last {self.seq_len} of {len(X)} rows')
                    print(f'   Clean rows available: {rows_after_dropna}, Buffer size: {len(df)}')

            if signal == 1: # Long entry predicted
                side = 'buy'
                entry = self.h[-1] + 0.00001 # Only enter when price exceeds previous high
                take = self.h[-1] + (2.5 * atr)
                stop = self.h[-1] - (1 * atr)
                position_size = (self.risk / abs(entry - stop)) / 100_000

            elif signal == -1: # Short entry predicted
                side = 'sell'
                entry = self.l[-1] - 0.00001 # Only enter when price exceeds previous low
                take = self.l[-1] - (2.5 * atr)
                stop = self.l[-1] + (1 * atr)
                position_size = (self.risk / abs(stop - entry)) / 100_000

            else: # No entry predicted
                signal = 0
                entry = 0
                side = None
                take = 0
                stop = 0
                position_size = 0
                prob_short = prob_flat = prob_long = 0.0
        else: # Do nothing if prediction not possible
            signal = 0
            entry = 0
            side = None
            stop = 0
            take = 0
            position_size = 0
            atr = 0
            prediction = None
            prob_short = prob_flat = prob_long = 0.0
            rows_after_dropna = 0
        
        position_size = min(position_size, self.maxpos)
        
        # Log prediction if logging is enabled
        if self.log and bar_time is not None:
            action_taken = 'warmup' if len(self.t) < self.min_bars_for_features else 'none'
            self._log_prediction(
                bar_time=bar_time,
                open_price=round(self.o[-1], 5) if len(self.o) > 0 else 0,
                high_price=round(self.h[-1], 5) if len(self.h) > 0 else 0,
                low_price=round(self.l[-1], 5) if len(self.l) > 0 else 0,
                close_price=round(self.c[-1], 5) if len(self.c) > 0 else 0,
                volume=int(self.v[-1]) if len(self.v) > 0 else 0,
                prediction=prediction,
                signal=signal,
                prob_short=round(prob_short, 4),
                prob_flat=round(prob_flat, 4),
                prob_long=round(prob_long, 4),
                side=side,
                entry=round(entry, 5) if entry else 0,
                stop=round(stop, 5) if stop else 0,
                take=round(take, 5) if take else 0,
                position_size=round(position_size, 2),
                atr=round(atr * 100000, 2) if atr else 0,  # Convert to pips
                buffer_len=len(self.t),
                clean_rows=rows_after_dropna,
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours,
                action_taken=action_taken
            )
        
        return signal, side, round(entry, 5), round(stop, 5), round(take, 5), round(position_size, 2)

    def on_bar(self, bar):

        if self.debug:
            print(bar)

        orders = []

        # Append raw 1 minute bars to class dataset
        self.t.append(pd.to_datetime(bar.Time))
        self.o.append(bar.Open)
        self.h.append(bar.High)
        self.l.append(bar.Low)
        self.c.append(bar.Close)
        self.v.append(bar.Volume)
    
        # Check mt5 for pending orders
        pending_order = self.check_pending_orders()
        open_position = self.check_open_positions()

        # Decrement countdown timer if active
        if self.countdown > 0:
            self.countdown -= 1

        # Check if current local time is outside restricted hours (8:30 AM - 11:00 AM)
        current_time = datetime.datetime.now().time()
        restricted_start = datetime.time(8, 30)
        restricted_end = datetime.time(11, 0)
        in_restricted_hours = restricted_start <= current_time <= restricted_end

        if self.debug:
            print(f'Pending Order: {pending_order}, Open Position: {open_position}, Countdown: {self.countdown}')
            print(f'Current local time: {current_time}, In restricted hours: {in_restricted_hours}')

        # Check & create new signal if below criteria is met:
        # A poisition is not open AND
        # No pending orders are active AND
        # No trading singal in place AND
        # Outside restricted hours (NOT between 8:30 AM - 11:00 AM local time)
        if not open_position and not pending_order and not in_restricted_hours: # 
            
            # Make new prediction
            self.signal, self.side, self.entry, self.stop, self.take, self.position_size = self.make_prediction(
                bar_time=self.t[-1] if len(self.t) > 0 else None,
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours
            )
            ema1, ema2 = self.get_moving_averages()

            # Act based on prediction, filter based on EMA cross
            if self.signal == 0:
                pass
            elif self.signal == 1: # Open a buy order and ema1 < ema2

                # Start the countdown timer. Countdown logic not used in this strategy
                self.countdown = self.patience

                # Send sell stop order to mt5 terminal
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    qty=self.position_size,
                    entry=self.entry,
                    entry_time=self.t[-1],
                    expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                    sl=self.stop,
                    tp=self.take
                )
                orders.append(order)
                
                # Update log action
                if self.log and len(self.t) > 0:
                    # Re-log with actual action taken
                    self._log_prediction(
                        bar_time=self.t[-1],
                        open_price=round(self.o[-1], 5),
                        high_price=round(self.h[-1], 5),
                        low_price=round(self.l[-1], 5),
                        close_price=round(self.c[-1], 5),
                        volume=int(self.v[-1]),
                        prediction=2,  # Buy signal is prediction class 2
                        signal=self.signal,
                        prob_short=0, prob_flat=0, prob_long=0,  # Probs already logged
                        side=self.side,
                        entry=self.entry,
                        stop=self.stop,
                        take=self.take,
                        position_size=self.position_size,
                        atr=round(talib.ATR(pd.Series(self.h), pd.Series(self.l), pd.Series(self.c), timeperiod=14).iloc[-1] * 100000, 2),
                        buffer_len=len(self.t),
                        clean_rows=0,
                        pending_order=pending_order,
                        open_position=open_position,
                        in_restricted_hours=in_restricted_hours,
                        action_taken='buy_order_placed'
                    )
                
                if self.debug:
                    print(f'Placing buy stop order with following details:')
                    print(orders)
            elif self.signal == -1: # Do not sell using current model

                # Start the countdown timer. Countdown logic not used in this strategy
                self.countdown = self.patience
                
                # Send buy stop order to mt5 terminal
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    qty=self.position_size,
                    entry=self.entry,
                    entry_time=self.t[-1],
                    expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                    sl=self.stop,
                    tp=self.take
                )
                orders.append(order)
                
                # Update log action
                if self.log and len(self.t) > 0:
                    # Re-log with actual action taken
                    self._log_prediction(
                        bar_time=self.t[-1],
                        open_price=round(self.o[-1], 5),
                        high_price=round(self.h[-1], 5),
                        low_price=round(self.l[-1], 5),
                        close_price=round(self.c[-1], 5),
                        volume=int(self.v[-1]),
                        prediction=0,  # Sell signal is prediction class 0
                        signal=self.signal,
                        prob_short=0, prob_flat=0, prob_long=0,  # Probs already logged
                        side=self.side,
                        entry=self.entry,
                        stop=self.stop,
                        take=self.take,
                        position_size=self.position_size,
                        atr=round(talib.ATR(pd.Series(self.h), pd.Series(self.l), pd.Series(self.c), timeperiod=14).iloc[-1] * 100000, 2),
                        buffer_len=len(self.t),
                        clean_rows=0,
                        pending_order=pending_order,
                        open_position=open_position,
                        in_restricted_hours=in_restricted_hours,
                        action_taken='sell_order_placed'
                    )
                
                if self.debug:
                    print(f'Placing sell stop order with following details:')
                    print(orders)
            else:
                pass
        return orders

class TripleBarrierHiLow_XAUUSD:
    """
    Uses similar logic as TripleBarrier, but uses limit orders to time entries and reduce trades that go bad fast.
    """
    def __init__(self, symbol, model, model_pack, patience, volume=0.1, mt5_executor=None, data_handler=None, debug=True, log=True, ticket_book: Optional["TicketBook"] = None):
        self.symbol = symbol
        self.order_type = 'stop'

        self.signal = 0 # If strategy is primed to enter trade
        self.patience = patience # How many bars to wait before giving up on a trade
        self.countdown = 0 # Countdown timer for trade entry
        self.debug = debug

        # Init price data deques
        # CRITICAL: Buffer sizing for accurate MTF feature calculation
        # 
        # Multi-timeframe indicators need significant history:
        # - 30min EMA(21) needs 21 * 30 = 630 1-min bars just to START computing
        # - After that, it needs ~500+ more bars for the EMA to stabilize
        # - Plus seq_len (256) for the model input
        # - Plus buffer for dropna() which removes ~200-400 rows
        #
        # Set to 10,000 bars (~7 days of M1 data) for optimal MTF feature stability
        # This matches the MT5DataHandler fetch size and gives ~333 valid 30-min bars
        self.maxlen = 7_000
        self.t = deque(maxlen=self.maxlen) # Time
        self.o = deque(maxlen=self.maxlen) # Open
        self.h = deque(maxlen=self.maxlen) # High
        self.l = deque(maxlen=self.maxlen) # Low
        self.c = deque(maxlen=self.maxlen) # Close
        self.v = deque(maxlen=self.maxlen) # Volume

        # Store temporary trade info
        self.order = None
        self.position = 0
        self.entry = 0
        self.stop = 0
        self.take = 0
        self.volume = volume
        self.regim_params = {
            'ma_period': 50,
            'slope_smoothness': 30,
            'regime_min_duration': 0,
            'atr_window': 60,
            'atr_lookback': 60*24,
            'atr_percentile': 1,
            'slope_threshold': 0.01
        }

        # Unpack model info
        self.model = model
        self.model_pack = model_pack
        self.model_info = model_pack['model_info']
        self.preprocess = model_pack['preprocess_functions']
        self.preprocess_args = model_pack['preprocess_args']
        self.preprocess_args['target_col'] = None
        self.target_col = model_pack['target_col'] if 'target_col' in model_pack else None
        self.scaler = model_pack['scaler']
        self.features = model_pack['feature_functions']
        self.seq_len = self.model_info['seq_len']
        # MT5 executor (optional) used to submit/cancel pending orders and query fills
        self.mt5_executor = mt5_executor
        # Data handler reference to access detected timezone offset
        self.data_handler = data_handler
        # Track a single pending ticket for this strategy instance (None when no pending)
        self.pending_order_ticket = None
        # Recorded fills from MT5: list[Order]
        self.fills = []
        # Last known signal value (helps detect transitions)
        self.last_signal = 0
        # TicketBook for state queries (pending / open position checks)
        self.ticket_book = ticket_book
        
        # Initialize logging
        self.log = log
        self.log_file = None
        self.csv_writer = None
        if self.log:
            self._initialize_logging()
        
        # Feature stability tracking - ensure indicators have warmed up
        self.features_ready = False
        # With 10,000 bar buffer, wait until we have at least 5,000 bars
        # This ensures MTF 30min features have 150+ periods to stabilize
        self.min_bars_for_features = self.maxlen

    def _initialize_logging(self):
        """Initialize CSV logging for predictions and trades"""
        # Create log directory if it doesn't exist
        log_dir = 'Engine/Learn/Trade Logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'{self.symbol}_{self.model_info["model_type"]}_log_{timestamp}.csv'
        log_path = os.path.join(log_dir, log_filename)
        
        # Open CSV file and create writer
        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Write header
        header = ['timestamp', 'bar_time', 'open', 'high', 'low', 'close', 'volume',
                  'prediction', 'signal', 'prob_short', 'prob_flat', 'prob_long',
                  'side', 'entry', 'stop', 'take', 
                  'position_size', 'atr', 'buffer_length', 'clean_rows',
                  'pending_order', 'open_position', 
                  'in_restricted_hours', 'action_taken']
        self.csv_writer.writerow(header)
        self.log_file.flush()
        
        print(f'[LOGGING] Initialized trade log: {log_path}')
    
    def _log_prediction(self, bar_time, open_price, high_price, low_price, close_price, volume,
                       prediction, signal, prob_short, prob_flat, prob_long,
                       side, entry, stop, take, position_size, 
                       atr, buffer_len, clean_rows, pending_order, open_position, in_restricted_hours, action_taken):
        """Log prediction details to CSV file"""
        if self.log and self.csv_writer:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            row = [timestamp, bar_time, open_price, high_price, low_price, close_price, volume,
                   prediction, signal, prob_short, prob_flat, prob_long,
                   side, entry, stop, take, 
                   position_size, atr, buffer_len, clean_rows, pending_order, open_position, 
                   in_restricted_hours, action_taken]
            self.csv_writer.writerow(row)
            self.log_file.flush()
    
    def __del__(self):
        """Cleanup: close log file when strategy is destroyed"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def check_pending_orders(self) -> bool:
        """
        Return True if there is an active pending order for this symbol.
        State is read from the TicketBook; no MT5 call is made.
        """
        if self.ticket_book is not None:
            return self.ticket_book.has_pending_order(self.symbol)
        return False

    def check_open_positions(self) -> bool:
        """
        Return True if there is an open (filled) position for this symbol.
        State is read from the TicketBook; no MT5 call is made.
        """
        if self.ticket_book is not None:
            return self.ticket_book.has_open_position(self.symbol)
        return False

    def make_prediction(self, bar_time=None, pending_order=False, open_position=False, in_restricted_hours=False):
        """
        Helper function: Make a prediction based on the current price data.
        Entry is currently set to market price (close of most recently printed candle).
        """

        # Enhanced buffer checks for feature stability
        if len(self.t) < self.min_bars_for_features:
            if self.debug and len(self.t) % 100 == 0:  # Print every 100 bars during warmup
                print(f'[WARMUP] Collecting data: {len(self.t)}/{self.min_bars_for_features} bars')
            return 0, None, 0, 0, 0, 0

        if len(self.t) == self.maxlen and not self.features_ready:
            self.features_ready = True
            print(f'[READY] Feature buffer full ({self.maxlen} bars). Model ready for predictions.')

        if len(self.t) == self.maxlen:
            df = pd.DataFrame(
                data={
                    'Time': self.t,
                    'Open': self.o,
                    'High': self.h,
                    'Low': self.l,
                    'Close': self.c,
                    'Volume': self.v
                }
            )
            df = df.sort_values('Time').reset_index(drop=True)
            
            # Stop/Take calculations
            _atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            atr = _atr.values[-1]

            # Compute features on full buffer (matches Review Model v2 approach)
            df_feat = self.features(df, regime_params=self.regim_params)
            
            # Critical: Drop NaN rows to match training preprocessing
            # This ensures feature indices align correctly
            rows_before_dropna = len(df_feat)
            df_feat_clean = df_feat.dropna(how='any')
            rows_after_dropna = len(df_feat_clean)
            
            if self.debug and not hasattr(self, '_logged_feature_info'):
                # Log feature calculation info once
                print(f'\n[FEATURE DEBUG] Buffer: {len(df)} bars')
                print(f'[FEATURE DEBUG] Rows before dropna: {rows_before_dropna}')
                print(f'[FEATURE DEBUG] Rows after dropna: {rows_after_dropna}')
                print(f'[FEATURE DEBUG] Rows lost to NaN: {rows_before_dropna - rows_after_dropna}')
                print(f'[FEATURE DEBUG] Feature columns: {len(df_feat.columns)}')
                
                # Check which columns have the most NaNs
                nan_counts = df_feat.isna().sum()
                worst_cols = nan_counts.nlargest(5)
                print(f'[FEATURE DEBUG] Columns with most NaNs: {dict(worst_cols)}')
                self._logged_feature_info = True
            
            # Verify we have enough clean data after dropping NaNs
            if len(df_feat_clean) < self.seq_len + 50:  # Need at least seq_len + buffer
                if self.debug:
                    print(f'[WARNING] Insufficient clean features: {len(df_feat_clean)} rows after dropna()')
                return 0, None, 0, 0, 0, 0
            
            # Preprocess features (matching Review Model v2)
            X, y, _, _ = self.preprocess(df_feat_clean, scaler=self.scaler, **self.preprocess_args)
            
            # Verify preprocessing produced valid output
            if len(X) < self.seq_len:
                if self.debug:
                    print(f'[WARNING] Insufficient preprocessed data: {len(X)} < {self.seq_len}')
                return 0, None, 0, 0, 0, 0

            # Extract last sequence (matches Review Model v2 sliding window approach)
            with torch.no_grad():
                input_tensor = torch.tensor(X[-self.seq_len:], dtype=torch.float32).unsqueeze(0)
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                signal = -1 if probs[0] > 0.65 else 1 if probs[2] > 0.64 else 0
                prediction = signal
                
                # Store probabilities for logging
                prob_short = probs[0].item()
                prob_flat = probs[1].item()
                prob_long = probs[2].item()
                
                if self.debug:
                    print(f'\n[[DEBUG PREDICTION]]: Prediction: {prediction}, Signal: {signal}')
                    print(f'   Probabilities: Short={prob_short:.3f}, Flat={prob_flat:.3f}, Long={prob_long:.3f}')
                    print(f'   Feature shape: {X.shape}, Sequence used: last {self.seq_len} of {len(X)} rows')
                    print(f'   Clean rows available: {rows_after_dropna}, Buffer size: {len(df)}')

            if signal == 1: # Long entry predicted
                side = 'buy'
                entry = self.h[-1] + 0.00001 # Only enter when price exceeds previous high
                take = self.h[-1] + (4 * atr)
                stop = self.h[-1] - (2 * atr)
                position_size = self.volume  # Fixed volume for XAUUSD

            elif signal == -1: # Short entry predicted
                side = 'sell'
                entry = self.l[-1] - 0.00001 # Only enter when price exceeds previous low
                take = self.l[-1] - (4 * atr)
                stop = self.l[-1] + (2 * atr)
                position_size = self.volume  # Fixed volume for XAUUSD

            else: # No entry predicted
                signal = 0
                entry = 0
                side = None
                take = 0
                stop = 0
                position_size = 0
                prob_short = prob_flat = prob_long = 0.0
        else: # Do nothing if prediction not possible
            signal = 0
            entry = 0
            side = None
            stop = 0
            take = 0
            position_size = 0
            atr = 0
            prediction = None
            prob_short = prob_flat = prob_long = 0.0
            rows_after_dropna = 0
        
        # Log prediction if logging is enabled
        if self.log and bar_time is not None:
            action_taken = 'warmup' if len(self.t) < self.min_bars_for_features else 'none'
            self._log_prediction(
                bar_time=bar_time,
                open_price=round(self.o[-1], 5) if len(self.o) > 0 else 0,
                high_price=round(self.h[-1], 5) if len(self.h) > 0 else 0,
                low_price=round(self.l[-1], 5) if len(self.l) > 0 else 0,
                close_price=round(self.c[-1], 5) if len(self.c) > 0 else 0,
                volume=int(self.v[-1]) if len(self.v) > 0 else 0,
                prediction=prediction,
                signal=signal,
                prob_short=round(prob_short, 4),
                prob_flat=round(prob_flat, 4),
                prob_long=round(prob_long, 4),
                side=side,
                entry=round(entry, 5) if entry else 0,
                stop=round(stop, 5) if stop else 0,
                take=round(take, 5) if take else 0,
                position_size=round(position_size, 2),
                atr=round(atr * 100000, 2) if atr else 0,  # Convert to pips
                buffer_len=len(self.t),
                clean_rows=rows_after_dropna,
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours,
                action_taken=action_taken
            )
        
        return signal, side, round(entry, 5), round(stop, 5), round(take, 5), round(position_size, 2)

    def on_bar(self, bar):

        if self.debug:
            print(bar)

        orders = []

        # Append raw 1 minute bars to class dataset
        self.t.append(pd.to_datetime(bar.Time))
        self.o.append(bar.Open)
        self.h.append(bar.High)
        self.l.append(bar.Low)
        self.c.append(bar.Close)
        self.v.append(bar.Volume)
    
        # Check mt5 for pending orders
        pending_order = self.check_pending_orders()
        open_position = self.check_open_positions()

        # Decrement countdown timer if active
        if self.countdown > 0:
            self.countdown -= 1

        # Check if current local time is outside restricted hours (8:30 AM - 11:00 AM)
        current_time = datetime.datetime.now().time()
        restricted_start = datetime.time(8, 30)
        restricted_end = datetime.time(11, 0)
        in_restricted_hours = restricted_start <= current_time <= restricted_end

        if self.debug:
            print(f'Pending Order: {pending_order}, Open Position: {open_position}, Countdown: {self.countdown}')
            print(f'Current local time: {current_time}, In restricted hours: {in_restricted_hours}')

        # Check & create new signal if below criteria is met:
        # A poisition is not open AND
        # No pending orders are active AND
        # No trading singal in place AND
        # Outside restricted hours (NOT between 8:30 AM - 11:00 AM local time)
        if not open_position and not pending_order and not in_restricted_hours: 
            
            # Make new prediction
            self.signal, self.side, self.entry, self.stop, self.take, self.position_size = self.make_prediction(
                bar_time=self.t[-1] if len(self.t) > 0 else None,
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours
            )

            # Act based on prediction, filter based on EMA cross
            if self.signal == 0:
                pass
            elif self.signal == 1: # Open a buy order

                # Start the countdown timer. Countdown logic not used in this strategy
                self.countdown = self.patience

                # Send sell stop order to mt5 terminal
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    qty=self.position_size,
                    entry=self.entry,
                    entry_time=self.t[-1],
                    expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                    sl=self.stop,
                    tp=self.take
                )
                orders.append(order)
                
                # Update log action
                if self.log and len(self.t) > 0:
                    # Re-log with actual action taken
                    self._log_prediction(
                        bar_time=self.t[-1],
                        open_price=round(self.o[-1], 5),
                        high_price=round(self.h[-1], 5),
                        low_price=round(self.l[-1], 5),
                        close_price=round(self.c[-1], 5),
                        volume=int(self.v[-1]),
                        prediction=2,  # Buy signal is prediction class 2
                        signal=self.signal,
                        prob_short=0, prob_flat=0, prob_long=0,  # Probs already logged
                        side=self.side,
                        entry=self.entry,
                        stop=self.stop,
                        take=self.take,
                        position_size=self.position_size,
                        atr=round(talib.ATR(pd.Series(self.h), pd.Series(self.l), pd.Series(self.c), timeperiod=14).iloc[-1] * 100000, 2),
                        buffer_len=len(self.t),
                        clean_rows=0,
                        pending_order=pending_order,
                        open_position=open_position,
                        in_restricted_hours=in_restricted_hours,
                        action_taken='buy_order_placed'
                    )
                
                if self.debug:
                    print(f'Placing buy stop order with following details:')
                    print(orders)

            elif self.signal == -1:

                # Start the countdown timer. Countdown logic not used in this strategy
                self.countdown = self.patience
                
                # Send buy stop order to mt5 terminal
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    qty=self.position_size,
                    entry=self.entry,
                    entry_time=self.t[-1],
                    expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                    sl=self.stop,
                    tp=self.take
                )
                orders.append(order)
                
                # Update log action
                if self.log and len(self.t) > 0:
                    # Re-log with actual action taken
                    self._log_prediction(
                        bar_time=self.t[-1],
                        open_price=round(self.o[-1], 5),
                        high_price=round(self.h[-1], 5),
                        low_price=round(self.l[-1], 5),
                        close_price=round(self.c[-1], 5),
                        volume=int(self.v[-1]),
                        prediction=0,  # Sell signal is prediction class 0
                        signal=self.signal,
                        prob_short=0, prob_flat=0, prob_long=0,  # Probs already logged
                        side=self.side,
                        entry=self.entry,
                        stop=self.stop,
                        take=self.take,
                        position_size=self.position_size,
                        atr=round(talib.ATR(pd.Series(self.h), pd.Series(self.l), pd.Series(self.c), timeperiod=14).iloc[-1] * 100000, 2),
                        buffer_len=len(self.t),
                        clean_rows=0,
                        pending_order=pending_order,
                        open_position=open_position,
                        in_restricted_hours=in_restricted_hours,
                        action_taken='sell_order_placed'
                    )
                
                if self.debug:
                    print(f'Placing sell stop order with following details:')
                    print(orders)
            else:
                pass
        return orders


from Learn.features import donchian_trend

if TYPE_CHECKING:
    from TicketBook import TicketBook


class TripleBarrierHiLowMulticlass:
    """Single multiclass model version of TripleBarrierHiLow.

    Uses one 3-class classifier whose output classes are:
        0 = SELL  → signal = -1  (short stop-order)
        1 = FLAT  → signal =  0  (no trade)
        2 = BUY   → signal = +1  (long stop-order)

    A trade is only placed when the predicted class is SELL or BUY **and**
    the softmax probability for that class meets or exceeds ``trade_threshold``.
    Entry/SL/TP sizing uses the Hi/Low stop-entry logic (matching TripleBarrierHiLow).
    """

    def __init__(
        self,
        symbol: str,
        model: torch.nn.Module,
        model_pack: dict,
        patience: int,
        maxlen: int = 7_000,
        risk: float = 50.0,
        trade_threshold: float = 0.5,
        donchian_length: int = 20,
        mt5_executor: Any = None,
        data_handler: Any = None,
        maxpos: float = 0.5,
        debug: bool = True,
        log: bool = True,
        ticket_book: Optional["TicketBook"] = None,
    ):
        self.symbol = symbol
        self.order_type = "stop"

        self.signal = 0
        self.maxpos = maxpos
        self.patience = patience
        self.countdown = 0
        self.debug = debug
        self.trade_threshold = float(trade_threshold)
        self.donchian_length = int(donchian_length)

        # --- Price buffers ---
        # 10,000 bars (~7 days of M1) ensures MTF indicators have stabilised.
        self.maxlen = maxlen
        self.t = deque(maxlen=self.maxlen)
        self.o = deque(maxlen=self.maxlen)
        self.h = deque(maxlen=self.maxlen)
        self.l = deque(maxlen=self.maxlen)
        self.c = deque(maxlen=self.maxlen)
        self.v = deque(maxlen=self.maxlen)

        # --- Trade state ---
        self.order: Optional[Order] = None
        self.position = 0
        self.entry = 0.0
        self.stop = 0.0
        self.take = 0.0

        self.risk = risk

        # --- Model + preprocessing ---
        self.model = model
        self.model_pack = model_pack
        self.model_info = model_pack["model_info"]
        self.seq_len = int(self.model_info.get("seq_len", 256))
        self.preprocess = model_pack["preprocess_function"]
        self.preprocess_args = dict(model_pack.get("preprocess_args", {}))
        self.preprocess_args["target_col"] = None
        self.scaler = model_pack.get("scaler")
        self.features = model_pack["feature_function"]
        self.regime_params = model_pack.get("regime_params")

        # --- MT5 executor + data handler ---
        self.mt5_executor = mt5_executor
        self.data_handler = data_handler

        self.pending_order_ticket = None
        self.fills = []
        self.last_signal = 0
        self.ticket_book = ticket_book

        # --- Logging ---
        self.log = log
        self.log_file = None
        self.csv_writer = None
        if self.log:
            self._initialize_logging()

        # Feature stability: wait until buffer holds enough bars for MTF indicators.
        self.features_ready = False
        self.min_bars_for_features = 5_000

        # Cache last prediction metrics for logging
        self._last_metrics: dict = {}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _initialize_logging(self) -> None:
        log_dir = "Engine/Learn/Trade Logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.model_info.get("model_type", "Multiclass")
        log_filename = f"{self.symbol}_{model_type}_log_{timestamp}.csv"
        log_path = os.path.join(log_dir, log_filename)

        self.log_file = open(log_path, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)

        header = [
            "timestamp",
            "bar_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "prediction",
            "signal",
            "prob_sell",
            "prob_flat",
            "prob_buy",
            "side",
            "entry",
            "stop",
            "take",
            "position_size",
            "atr_pips",
            "buffer_length",
            "clean_rows",
            "pending_order",
            "open_position",
            "in_restricted_hours",
            "action_taken",
        ]
        self.csv_writer.writerow(header)
        self.log_file.flush()

        print(f"[LOGGING] Initialized trade log: {log_path}")

    def _log_row(
        self,
        *,
        bar_time,
        pending_order: bool,
        open_position: bool,
        in_restricted_hours: bool,
        action_taken: str,
    ) -> None:
        if not (self.log and self.csv_writer):
            return

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts,
            bar_time,
            round(self.o[-1], 5) if len(self.o) else 0,
            round(self.h[-1], 5) if len(self.h) else 0,
            round(self.l[-1], 5) if len(self.l) else 0,
            round(self.c[-1], 5) if len(self.c) else 0,
            int(self.v[-1]) if len(self.v) else 0,
            self._last_metrics.get("prediction"),
            self._last_metrics.get("signal"),
            self._last_metrics.get("prob_sell"),
            self._last_metrics.get("prob_flat"),
            self._last_metrics.get("prob_buy"),
            self._last_metrics.get("side"),
            self._last_metrics.get("entry"),
            self._last_metrics.get("stop"),
            self._last_metrics.get("take"),
            self._last_metrics.get("position_size"),
            self._last_metrics.get("atr_pips"),
            self._last_metrics.get("buffer_len"),
            self._last_metrics.get("clean_rows"),
            pending_order,
            open_position,
            in_restricted_hours,
            action_taken,
        ]
        self.csv_writer.writerow(row)
        self.log_file.flush()

    def __del__(self):
        if hasattr(self, "log_file") and self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def check_pending_orders(self) -> bool:
        """Return True if there is an active pending order for this symbol."""
        if self.ticket_book is not None:
            return self.ticket_book.has_pending_order(self.symbol)
        return False

    def check_open_positions(self) -> bool:
        """Return True if there is an open (filled) position for this symbol."""
        if self.ticket_book is not None:
            return self.ticket_book.has_open_position(self.symbol)
        return False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _run_model(
        self, df_ohlcv: pd.DataFrame
    ) -> Tuple[Optional[int], float, float, float, int]:
        """Run the model on *df_ohlcv* and return (pred, prob_sell, prob_flat, prob_buy, clean_rows).

        Returns (None, 0, 1, 0, clean_rows) when there is insufficient data.
        """
        df_feat = self.features(df_ohlcv, regime_params=self.regime_params)
        df_clean = df_feat.dropna(how="any")
        clean_rows = int(len(df_clean))

        if clean_rows < self.seq_len + 50:
            if self.debug:
                print(f"Not enough clean data for model: {clean_rows} rows (need at least {self.seq_len + 50})")
            return None, 0.0, 1.0, 0.0, clean_rows

        X, _, _, _ = self.preprocess(df_clean, scaler=self.scaler, **self.preprocess_args)
        if len(X) < self.seq_len:
            if self.debug:
                print(f"Not enough preprocessed data for model: {len(X)} rows (need at least {self.seq_len})")
            return None, 0.0, 1.0, 0.0, clean_rows

        seq = X[-self.seq_len:]
        with torch.no_grad():
            input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(logits, dim=1).item())

        prob_sell = float(probs[0].item())
        prob_flat = float(probs[1].item())
        prob_buy  = float(probs[2].item())

        if self.debug:
            print("\n[[DEBUG MODEL OUTPUT - MULTICLASS]]")
            print(f"Raw logits: {logits.numpy()}")
            print(f"Softmax probabilities: sell={prob_sell:.3f}, flat={prob_flat:.3f}, buy={prob_buy:.3f}")
            print(f"Predicted class: {pred} ({'SELL' if pred == 0 else 'FLAT' if pred == 1 else 'BUY'})")
            print(f"Clean rows after feature engineering: {clean_rows}")

        return pred, prob_sell, prob_flat, prob_buy, clean_rows

    def make_prediction(
        self,
        *,
        bar_time=None,
        pending_order: bool = False,
        open_position: bool = False,
        in_restricted_hours: bool = False,
    ) -> Tuple[int, Optional[str], float, float, float, float]:
        """Return (signal, side, entry, stop, take, position_size).

        signal: -1 = short, 0 = flat/no trade, +1 = long.
        """
        _null_metrics = {
            "prediction": None,
            "signal": 0,
            "prob_sell": 0.0,
            "prob_flat": 1.0,
            "prob_buy": 0.0,
            "side": None,
            "entry": 0.0,
            "stop": 0.0,
            "take": 0.0,
            "position_size": 0.0,
            "atr_pips": 0.0,
            "buffer_len": len(self.t),
            "clean_rows": 0,
        }

        # Warmup — wait for feature stability
        if len(self.t) < self.min_bars_for_features:
            if self.debug: #and len(self.t) % 100 == 0:
                print(f"[WARMUP] Collecting data: {len(self.t)}/{self.min_bars_for_features} bars")
            self._last_metrics = _null_metrics
            return 0, None, 0.0, 0.0, 0.0, 0.0

        if len(self.t) == self.maxlen and not self.features_ready:
            self.features_ready = True
            print(f"[READY] Feature buffer full ({self.maxlen} bars). Model ready for predictions.")

        if len(self.t) != self.maxlen:
            self._last_metrics = _null_metrics
            return 0, None, 0.0, 0.0, 0.0, 0.0

        df = pd.DataFrame(
            {
                "Time": self.t,
                "Open": self.o,
                "High": self.h,
                "Low": self.l,
                "Close": self.c,
                "Volume": self.v,
            }
        ).sort_values("Time").reset_index(drop=True)

        _atr = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
        atr = float(_atr.values[-1])

        pred, prob_sell, prob_flat, prob_buy, clean_rows = self._run_model(df)

        # Donchian trend gate: +1 = uptrend, -1 = downtrend, 0 = neutral
        don_series = donchian_trend(df, length=self.donchian_length)
        don = int(don_series.iloc[-1])

        # Map prediction class to trading signal, gated by threshold
        if pred == 2 and prob_buy >= self.trade_threshold:
            signal = 1
        elif pred == 0 and prob_sell >= self.trade_threshold:
            signal = -1
        else:
            signal = 0

        # Donchian gate: BUY only in uptrend, SELL only in downtrend
        if signal == 1 and don <= 0:
            signal = 0
        elif signal == -1 and don >= 0:
            signal = 0

        if self.debug:
            print("\n[[DEBUG PREDICTION - MULTICLASS]]")
            print(f"   pred={pred}  prob_sell={prob_sell:.3f}  prob_flat={prob_flat:.3f}  prob_buy={prob_buy:.3f}")
            print(f"   donchian_trend={don}  trade_threshold={self.trade_threshold}  final signal={signal}")

        # Compute Hi/Low stop-order entry, stop, take
        if signal == 1:
            side = "buy"
            entry = float(self.h[-1]) + 0.00001
            take  = float(self.h[-1]) + (2.5 * atr)
            stop  = float(self.h[-1]) - (2.5 * atr)
            position_size = self.maxpos #(self.risk / abs(entry - stop)) / 100_000
        elif signal == -1:
            side = "sell"
            entry = float(self.l[-1]) - 0.00001
            take  = float(self.l[-1]) - (2.5 * atr)
            stop  = float(self.l[-1]) + (2.5 * atr)
            position_size = self.maxpos #(self.risk / abs(stop - entry)) / 100_000
        else:
            side = None
            entry = stop = take = 0.0
            position_size = 0.0

        position_size = float(min(position_size, self.maxpos))

        self._last_metrics = {
            "prediction": pred,
            "signal": int(signal),
            "prob_sell": round(prob_sell, 4),
            "prob_flat": round(prob_flat, 4),
            "prob_buy": round(prob_buy, 4),
            "side": side,
            "entry": round(entry, 5) if entry else 0.0,
            "stop": round(stop, 5) if stop else 0.0,
            "take": round(take, 5) if take else 0.0,
            "position_size": round(position_size, 2),
            "atr_pips": round(atr * 100_000, 2) if atr else 0.0,
            "donchian_trend": don,
            "buffer_len": len(self.t),
            "clean_rows": clean_rows,
        }

        return int(signal), side, round(entry, 5), round(stop, 5), round(take, 5), round(position_size, 2)

    # ------------------------------------------------------------------
    # Main event handler
    # ------------------------------------------------------------------

    def on_bar(self, bar):
        if self.debug:
            print(f"{len(self.t)}/{self.maxlen} bars in buffer. Processing new bar:")
            print(bar)

        orders = []

        # Feed incoming bar into the price buffers
        self.t.append(pd.to_datetime(bar.Time))
        self.o.append(bar.Open)
        self.h.append(bar.High)
        self.l.append(bar.Low)
        self.c.append(bar.Close)
        self.v.append(bar.Volume)

        pending_order = self.check_pending_orders()
        open_position = self.check_open_positions()

        if self.countdown > 0:
            self.countdown -= 1

        # Restricted trading hours — skip new signals between 7:30 and 10:00 local time
        current_time = datetime.datetime.now().time()
        restricted_start = datetime.time(7, 30)
        restricted_end = datetime.time(10, 0)
        in_restricted_hours = restricted_start <= current_time <= restricted_end

        if self.debug:
            print(
                f"Pending Order: {pending_order}, Open Position: {open_position}, Countdown: {self.countdown}"
            )
            print(f"Current local time: {current_time}, In restricted hours: {in_restricted_hours}")

        # Only open new trades when idle and outside restricted hours
        if not open_position and not pending_order and not in_restricted_hours:
            if self.debug:
                print("Checking for new trade signal...")

        self.signal, self.side, self.entry, self.stop, self.take, self.position_size = (
            self.make_prediction(
                bar_time=self.t[-1] if len(self.t) > 0 else None,
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours,
            )
        )

        if self.debug:
            print(f"make_prediction() returned signal={self.signal}, side={self.side}, "
                    f"entry={self.entry}, stop={self.stop}, take={self.take}, size={self.position_size}")

        if self.signal == 1:
            self.countdown = self.patience
            order = Order(
                symbol=self.symbol,
                side=self.side,
                qty=self.position_size,
                entry=self.entry,
                entry_time=self.t[-1],
                expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                sl=self.stop,
                tp=self.take,
            )
            orders.append(order)

            self._log_row(
                bar_time=self.t[-1],
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours,
                action_taken="buy_order_placed",
            )

            if self.debug:
                print("Placing buy stop order:")
                print(orders)

        elif self.signal == -1:
            self.countdown = self.patience
            order = Order(
                symbol=self.symbol,
                side=self.side,
                qty=self.position_size,
                entry=self.entry,
                entry_time=self.t[-1],
                expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                sl=self.stop,
                tp=self.take,
            )
            orders.append(order)

            self._log_row(
                bar_time=self.t[-1],
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours,
                action_taken="sell_order_placed",
            )

            if self.debug:
                print("Placing sell stop order:")
                print(orders)

        else:
            if self.log and len(self.t) > 0:
                self._log_row(
                    bar_time=self.t[-1],
                    pending_order=pending_order,
                    open_position=open_position,
                    in_restricted_hours=in_restricted_hours,
                    action_taken="none",
                )

        return orders

class TripleBarrierHiLowBinary:
    """Dual-binary-model version of `TripleBarrierHiLow`.

    Uses two separate binary classifiers:
    - buy model: predicts whether to take a BUY trade (1=trade, 0=hold)
    - sell model: predicts whether to take a SELL trade (1=trade, 0=hold)

    The final action is selected as:
    - buy_trade and not sell_trade  -> long
    - sell_trade and not buy_trade  -> short
    - both trade                    -> choose higher trade probability
    - neither trade                 -> flat

    Entry/SL/TP sizing follows the Hi/Low stop-entry logic from `TripleBarrierHiLow`.
    """

    def __init__(
        self,
        symbol: str,
        buy_model: torch.nn.Module,
        buy_model_pack: dict,
        sell_model: torch.nn.Module,
        sell_model_pack: dict,
        patience: int,
        risk: float = 50,
        ema1_period: int = 8,
        ema2_period: int = 30,
        buy_threshold: float = 0.5,
        sell_threshold: float = 0.5,
        mt5_executor: Any = None,
        data_handler: Any = None,
        maxpos: float = 0.5,
        debug: bool = True,
        log: bool = True,
        ticket_book: Optional["TicketBook"] = None,
    ):
        self.symbol = symbol
        self.order_type = "stop"

        self.signal = 0
        self.maxpos = maxpos
        self.patience = patience
        self.countdown = 0
        self.debug = debug

        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)

        # --- Price buffers (same sizing rationale as TripleBarrierHiLow) ---
        self.maxlen = 10_000
        self.t = deque(maxlen=self.maxlen)
        self.o = deque(maxlen=self.maxlen)
        self.h = deque(maxlen=self.maxlen)
        self.l = deque(maxlen=self.maxlen)
        self.c = deque(maxlen=self.maxlen)
        self.v = deque(maxlen=self.maxlen)

        # --- Trade state ---
        self.order: Optional[Order] = None
        self.position = 0
        self.entry = 0.0
        self.stop = 0.0
        self.take = 0.0

        self.risk = risk
        self.ema1_period = ema1_period
        self.ema2_period = ema2_period

        # --- Model + preprocessing (buy) ---
        self.buy_model = buy_model
        self.buy_model_pack = buy_model_pack
        self.buy_model_info = buy_model_pack["model_info"]
        self.buy_seq_len = int(self.buy_model_info.get("seq_len", 256))
        self.buy_preprocess = buy_model_pack["preprocess_functions"]
        self.buy_preprocess_args = dict(buy_model_pack.get("preprocess_args", {}))
        self.buy_preprocess_args["target_col"] = None
        self.buy_scaler = buy_model_pack.get("scaler")
        self.buy_features = buy_model_pack["feature_functions"]

        # --- Model + preprocessing (sell) ---
        self.sell_model = sell_model
        self.sell_model_pack = sell_model_pack
        self.sell_model_info = sell_model_pack["model_info"]
        self.sell_seq_len = int(self.sell_model_info.get("seq_len", 256))
        self.sell_preprocess = sell_model_pack["preprocess_functions"]
        self.sell_preprocess_args = dict(sell_model_pack.get("preprocess_args", {}))
        self.sell_preprocess_args["target_col"] = None
        self.sell_scaler = sell_model_pack.get("scaler")
        self.sell_features = sell_model_pack["feature_functions"]

        # --- MT5 executor + data handler ---
        self.mt5_executor = mt5_executor
        self.data_handler = data_handler

        self.pending_order_ticket = None
        self.fills = []
        self.last_signal = 0
        self.ticket_book = ticket_book

        # --- Logging ---
        self.log = log
        self.log_file = None
        self.csv_writer = None
        if self.log:
            self._initialize_logging()

        # Feature stability tracking
        self.features_ready = False
        self.min_bars_for_features = 5_000

        # Cache last prediction metrics for logging
        self._last_metrics: dict = {}

    def _initialize_logging(self) -> None:
        log_dir = "Engine/Learn/Trade Logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.symbol}_BinaryDual_log_{timestamp}.csv"
        log_path = os.path.join(log_dir, log_filename)

        self.log_file = open(log_path, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)

        header = [
            "timestamp",
            "bar_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "buy_pred",
            "buy_prob_trade",
            "sell_pred",
            "sell_prob_trade",
            "signal",
            "side",
            "entry",
            "stop",
            "take",
            "position_size",
            "atr_pips",
            "buffer_length",
            "buy_clean_rows",
            "sell_clean_rows",
            "pending_order",
            "open_position",
            "in_restricted_hours",
            "action_taken",
        ]
        self.csv_writer.writerow(header)
        self.log_file.flush()

        print(f"[LOGGING] Initialized trade log: {log_path}")

    def _log_row(
        self,
        *,
        bar_time,
        pending_order: bool,
        open_position: bool,
        in_restricted_hours: bool,
        action_taken: str,
    ) -> None:
        if not (self.log and self.csv_writer):
            return

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts,
            bar_time,
            round(self.o[-1], 5) if len(self.o) else 0,
            round(self.h[-1], 5) if len(self.h) else 0,
            round(self.l[-1], 5) if len(self.l) else 0,
            round(self.c[-1], 5) if len(self.c) else 0,
            int(self.v[-1]) if len(self.v) else 0,
            self._last_metrics.get("buy_pred"),
            self._last_metrics.get("buy_prob_trade"),
            self._last_metrics.get("sell_pred"),
            self._last_metrics.get("sell_prob_trade"),
            self._last_metrics.get("signal"),
            self._last_metrics.get("side"),
            self._last_metrics.get("entry"),
            self._last_metrics.get("stop"),
            self._last_metrics.get("take"),
            self._last_metrics.get("position_size"),
            self._last_metrics.get("atr_pips"),
            self._last_metrics.get("buffer_len"),
            self._last_metrics.get("buy_clean_rows"),
            self._last_metrics.get("sell_clean_rows"),
            pending_order,
            open_position,
            in_restricted_hours,
            action_taken,
        ]
        self.csv_writer.writerow(row)
        self.log_file.flush()

    def __del__(self):
        if hasattr(self, "log_file") and self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass

    def get_moving_averages(self) -> Tuple[Optional[float], Optional[float]]:
        if len(self.t) == self.maxlen:
            c = pd.Series(list(self.c))
            ema1 = c.ewm(span=self.ema1_period, adjust=False).mean().iloc[-1]
            ema2 = c.ewm(span=self.ema2_period, adjust=False).mean().iloc[-1]
            return float(ema1), float(ema2)
        return None, None

    def check_pending_orders(self) -> bool:
        """Return True if there is an active pending order for this symbol.
        State is read from the TicketBook; no MT5 call is made."""
        if self.ticket_book is not None:
            return self.ticket_book.has_pending_order(self.symbol)
        return False

    def check_open_positions(self) -> bool:
        """Return True if there is an open (filled) position for this symbol.
        State is read from the TicketBook; no MT5 call is made."""
        if self.ticket_book is not None:
            return self.ticket_book.has_open_position(self.symbol)
        return False

    def _compute_binary_signal(
        self,
        *,
        model: torch.nn.Module,
        features_fn: Callable[[pd.DataFrame], pd.DataFrame],
        preprocess_fn: Callable[..., Any],
        preprocess_args: dict,
        scaler: Any,
        seq_len: int,
        df_ohlcv: pd.DataFrame,
    ) -> Tuple[Optional[int], float, int]:
        """Return (pred_class, prob_trade, clean_rows) for a binary classifier.

        Assumes class index 1 corresponds to "trade".
        """

        df_feat = features_fn(df_ohlcv)
        df_feat_clean = df_feat.dropna(how="any")
        clean_rows = int(len(df_feat_clean))

        if clean_rows < seq_len + 50:
            return None, 0.0, clean_rows

        X, _, _, _ = preprocess_fn(df_feat_clean, scaler=scaler, **preprocess_args)
        if len(X) < seq_len:
            return None, 0.0, clean_rows

        seq = X[-seq_len:]
        with torch.no_grad():
            input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(logits, dim=1).item())
            prob_trade = float(probs[1].item()) if probs.numel() >= 2 else float(probs[0].item())

        return pred, prob_trade, clean_rows

    def make_prediction(
        self,
        *,
        bar_time=None,
        pending_order: bool = False,
        open_position: bool = False,
        in_restricted_hours: bool = False,
    ) -> Tuple[int, Optional[str], float, float, float, float]:
        # Warmup for feature stability
        if len(self.t) < self.min_bars_for_features:
            if self.debug and len(self.t) % 100 == 0:
                print(f"[WARMUP] Collecting data: {len(self.t)}/{self.min_bars_for_features} bars")

            self._last_metrics = {
                "buy_pred": None,
                "buy_prob_trade": 0.0,
                "sell_pred": None,
                "sell_prob_trade": 0.0,
                "signal": 0,
                "side": None,
                "entry": 0.0,
                "stop": 0.0,
                "take": 0.0,
                "position_size": 0.0,
                "atr_pips": 0.0,
                "buffer_len": len(self.t),
                "buy_clean_rows": 0,
                "sell_clean_rows": 0,
            }

            return 0, None, 0.0, 0.0, 0.0, 0.0

        if len(self.t) == self.maxlen and not self.features_ready:
            self.features_ready = True
            print(f"[READY] Feature buffer full ({self.maxlen} bars). Models ready for predictions.")

        if len(self.t) != self.maxlen:
            self._last_metrics = {
                "buy_pred": None,
                "buy_prob_trade": 0.0,
                "sell_pred": None,
                "sell_prob_trade": 0.0,
                "signal": 0,
                "side": None,
                "entry": 0.0,
                "stop": 0.0,
                "take": 0.0,
                "position_size": 0.0,
                "atr_pips": 0.0,
                "buffer_len": len(self.t),
                "buy_clean_rows": 0,
                "sell_clean_rows": 0,
            }
            return 0, None, 0.0, 0.0, 0.0, 0.0

        df = pd.DataFrame(
            {
                "Time": self.t,
                "Open": self.o,
                "High": self.h,
                "Low": self.l,
                "Close": self.c,
                "Volume": self.v,
            }
        ).sort_values("Time").reset_index(drop=True)

        _atr = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
        atr = float(_atr.values[-1])

        buy_pred, buy_prob_trade, buy_clean_rows = self._compute_binary_signal(
            model=self.buy_model,
            features_fn=self.buy_features,
            preprocess_fn=self.buy_preprocess,
            preprocess_args=self.buy_preprocess_args,
            scaler=self.buy_scaler,
            seq_len=self.buy_seq_len,
            df_ohlcv=df,
        )

        sell_pred, sell_prob_trade, sell_clean_rows = self._compute_binary_signal(
            model=self.sell_model,
            features_fn=self.sell_features,
            preprocess_fn=self.sell_preprocess,
            preprocess_args=self.sell_preprocess_args,
            scaler=self.sell_scaler,
            seq_len=self.sell_seq_len,
            df_ohlcv=df,
        )

        buy_trade = buy_prob_trade >= self.buy_threshold
        sell_trade = sell_prob_trade >= self.sell_threshold

        # Decide final signal
        signal = 0
        if buy_trade and not sell_trade:
            signal = 1
        elif sell_trade and not buy_trade:
            signal = -1
        elif buy_trade and sell_trade:
            if buy_prob_trade > sell_prob_trade:
                signal = 1
            elif sell_prob_trade > buy_prob_trade:
                signal = -1
            else:
                signal = 0

        if self.debug:
            print("\n[[DEBUG PREDICTION - BINARY DUAL]]")
            print(f"   Buy:  pred={buy_pred}, prob_trade={buy_prob_trade:.3f}, thr={self.buy_threshold}")
            print(f"   Sell: pred={sell_pred}, prob_trade={sell_prob_trade:.3f}, thr={self.sell_threshold}")
            print(f"   Final signal: {signal}")

        # Convert signal to order details (Hi/Low stop-entry)
        if signal == 1:
            side = "buy"
            entry = float(self.h[-1]) + 0.00001
            take = float(self.h[-1]) + (2.5 * atr)
            stop = float(self.h[-1]) - (1.0 * atr)
            position_size = (self.risk / abs(entry - stop)) / 100_000
        elif signal == -1:
            side = "sell"
            entry = float(self.l[-1]) - 0.00001
            take = float(self.l[-1]) - (2.5 * atr)
            stop = float(self.l[-1]) + (1.0 * atr)
            position_size = (self.risk / abs(stop - entry)) / 100_000
        else:
            side = None
            entry = stop = take = 0.0
            position_size = 0.0

        position_size = float(min(position_size, self.maxpos))

        self._last_metrics = {
            "buy_pred": buy_pred,
            "buy_prob_trade": round(buy_prob_trade, 4),
            "sell_pred": sell_pred,
            "sell_prob_trade": round(sell_prob_trade, 4),
            "signal": int(signal),
            "side": side,
            "entry": round(entry, 5) if entry else 0.0,
            "stop": round(stop, 5) if stop else 0.0,
            "take": round(take, 5) if take else 0.0,
            "position_size": round(position_size, 2),
            "atr_pips": round(atr * 100000, 2) if atr else 0.0,
            "buffer_len": len(self.t),
            "buy_clean_rows": buy_clean_rows,
            "sell_clean_rows": sell_clean_rows,
        }

        return int(signal), side, round(entry, 5), round(stop, 5), round(take, 5), round(position_size, 2)

    def on_bar(self, bar):
        if self.debug:
            print(bar)

        orders = []

        # Append raw 1-minute bars
        self.t.append(pd.to_datetime(bar.Time))
        self.o.append(bar.Open)
        self.h.append(bar.High)
        self.l.append(bar.Low)
        self.c.append(bar.Close)
        self.v.append(bar.Volume)

        pending_order = self.check_pending_orders()
        open_position = self.check_open_positions()

        if self.countdown > 0:
            self.countdown -= 1

        current_time = datetime.datetime.now().time()
        restricted_start = datetime.time(8, 30)
        restricted_end = datetime.time(11, 0)
        in_restricted_hours = restricted_start <= current_time <= restricted_end

        if self.debug:
            print(
                f"Pending Order: {pending_order}, Open Position: {open_position}, Countdown: {self.countdown}"
            )
            print(f"Current local time: {current_time}, In restricted hours: {in_restricted_hours}")

        # Only create new signals when not in a trade/pending and not restricted
        if not open_position and not pending_order and not in_restricted_hours:
            self.signal, self.side, self.entry, self.stop, self.take, self.position_size = self.make_prediction(
                bar_time=self.t[-1] if len(self.t) > 0 else None,
                pending_order=pending_order,
                open_position=open_position,
                in_restricted_hours=in_restricted_hours,
            )

            ema1, ema2 = self.get_moving_averages()
            _ = (ema1, ema2)  # keep parity with original signature/use

            if self.signal == 1:
                self.countdown = self.patience
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    qty=self.position_size,
                    entry=self.entry,
                    entry_time=self.t[-1],
                    expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                    sl=self.stop,
                    tp=self.take,
                )
                orders.append(order)

                self._log_row(
                    bar_time=self.t[-1],
                    pending_order=pending_order,
                    open_position=open_position,
                    in_restricted_hours=in_restricted_hours,
                    action_taken="buy_order_placed",
                )

                if self.debug:
                    print("Placing buy stop order with following details:")
                    print(orders)

            elif self.signal == -1:
                self.countdown = self.patience
                order = Order(
                    symbol=self.symbol,
                    side=self.side,
                    qty=self.position_size,
                    entry=self.entry,
                    entry_time=self.t[-1],
                    expiration=self.t[-1] + datetime.timedelta(minutes=self.patience),
                    sl=self.stop,
                    tp=self.take,
                )
                orders.append(order)

                self._log_row(
                    bar_time=self.t[-1],
                    pending_order=pending_order,
                    open_position=open_position,
                    in_restricted_hours=in_restricted_hours,
                    action_taken="sell_order_placed",
                )

                if self.debug:
                    print("Placing sell stop order with following details:")
                    print(orders)

            else:
                # No trade
                if self.log and len(self.t) > 0:
                    self._log_row(
                        bar_time=self.t[-1],
                        pending_order=pending_order,
                        open_position=open_position,
                        in_restricted_hours=in_restricted_hours,
                        action_taken="none",
                    )

        return orders
