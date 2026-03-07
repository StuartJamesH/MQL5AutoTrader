from DataHandler import Order
from collections import deque
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib


class PriceActionTrader:
    """
    Price Action Trading Strategy based on classic PAT principles:
    - Pin bars (rejection candles)
    - Inside bars
    - Engulfing patterns
    - Support/Resistance breaks with retests
    - Trend following with higher highs/lower lows
    """
    
    def __init__(self, symbol, patience=10, risk=50, lookback=20, data_handler=None, 
                 maxpos=0.5, debug=True, strategy_name="PriceActionTrader", ticketbook=None):
        self.symbol = symbol
        self.order_type = 'stop'  # Use stop orders for breakout entries
        self.strategy_name = strategy_name
        self.ticketbook = ticketbook
        
        self.signal = 0
        self.maxpos = maxpos
        self.patience = patience
        self.countdown = 0
        self.debug = debug
        
        # Price data storage
        self.maxlen = 500
        self.t = deque(maxlen=self.maxlen)
        self.o = deque(maxlen=self.maxlen)
        self.h = deque(maxlen=self.maxlen)
        self.l = deque(maxlen=self.maxlen)
        self.c = deque(maxlen=self.maxlen)
        self.v = deque(maxlen=self.maxlen)
        
        # Strategy parameters
        self.risk = risk
        self.lookback = lookback
        self.data_handler = data_handler
        
        # Pattern detection flags
        self.last_pattern = None
        self.support_level = None
        self.resistance_level = None
        
    def identify_pin_bar(self, idx=-1):
        """
        Identify pin bar (rejection candle) pattern.
        A pin bar has:
        - Small body (less than 1/3 of total range)
        - Long wick (at least 2/3 of total range)
        - Wick should be on one side (rejection)
        
        Returns: 1 for bullish pin, -1 for bearish pin, 0 for none
        """
        if len(self.c) < 3:
            return 0
            
        o, h, l, c = self.o[idx], self.h[idx], self.l[idx], self.c[idx]
        
        total_range = h - l
        if total_range == 0:
            return 0
            
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        # Bullish pin bar: long lower wick, small body
        if lower_wick > (2/3) * total_range and body < (1/3) * total_range:
            # Confirm with previous candle being bearish
            if idx > -len(self.c) and self.c[idx-1] < self.o[idx-1]:
                return 1
                
        # Bearish pin bar: long upper wick, small body
        if upper_wick > (2/3) * total_range and body < (1/3) * total_range:
            # Confirm with previous candle being bullish
            if idx > -len(self.c) and self.c[idx-1] > self.o[idx-1]:
                return -1
                
        return 0
    
    def identify_inside_bar(self):
        """
        Inside bar: current bar's high and low are within previous bar's range
        Returns: True if inside bar detected
        """
        if len(self.c) < 2:
            return False
            
        curr_h, curr_l = self.h[-1], self.l[-1]
        prev_h, prev_l = self.h[-2], self.l[-2]
        
        return curr_h <= prev_h and curr_l >= prev_l
    
    def identify_engulfing(self):
        """
        Engulfing pattern: current candle completely engulfs previous candle
        Returns: 1 for bullish engulfing, -1 for bearish engulfing, 0 for none
        """
        if len(self.c) < 2:
            return 0
            
        curr_o, curr_c = self.o[-1], self.c[-1]
        prev_o, prev_c = self.o[-2], self.c[-2]
        
        # Bullish engulfing
        if curr_c > curr_o and prev_c < prev_o:  # Current bullish, previous bearish
            if curr_o <= prev_c and curr_c >= prev_o:  # Current engulfs previous
                return 1
                
        # Bearish engulfing
        if curr_c < curr_o and prev_c > prev_o:  # Current bearish, previous bullish
            if curr_o >= prev_c and curr_c <= prev_o:  # Current engulfs previous
                return -1
                
        return 0
    
    def calculate_support_resistance(self):
        """
        Calculate key support and resistance levels using swing highs/lows
        """
        if len(self.c) < self.lookback:
            return None, None
            
        highs = list(self.h)[-self.lookback:]
        lows = list(self.l)[-self.lookback:]
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs) - 2):
            # Swing high: higher than 2 bars on each side
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
                
            # Swing low: lower than 2 bars on each side
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        resistance = max(swing_highs) if swing_highs else None
        support = min(swing_lows) if swing_lows else None
        
        return support, resistance
    
    def detect_trend(self):
        """
        Detect trend using higher highs/lower lows
        Returns: 1 for uptrend, -1 for downtrend, 0 for sideways
        """
        if len(self.c) < self.lookback:
            return 0
            
        highs = list(self.h)[-self.lookback:]
        lows = list(self.l)[-self.lookback:]
        
        # Check for higher highs and higher lows (uptrend)
        higher_highs = sum([1 for i in range(1, len(highs)) if highs[i] > highs[i-1]])
        higher_lows = sum([1 for i in range(1, len(lows)) if lows[i] > lows[i-1]])
        
        # Check for lower highs and lower lows (downtrend)
        lower_highs = sum([1 for i in range(1, len(highs)) if highs[i] < highs[i-1]])
        lower_lows = sum([1 for i in range(1, len(lows)) if lows[i] < lows[i-1]])
        
        uptrend_score = higher_highs + higher_lows
        downtrend_score = lower_highs + lower_lows
        
        threshold = self.lookback * 0.6  # 60% of bars should confirm trend
        
        if uptrend_score > threshold:
            return 1
        elif downtrend_score > threshold:
            return -1
        else:
            return 0
    
    def calculate_position_size_and_levels(self, signal, entry_price):
        """
        Calculate position size, stop loss, and take profit based on ATR
        """
        if len(self.c) < 14:
            return 0, 0, 0
            
        df = pd.DataFrame({
            'High': list(self.h)[-50:],
            'Low': list(self.l)[-50:],
            'Close': list(self.c)[-50:]
        })
        
        atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14).iloc[-1]
        
        if signal == 1:  # Long entry
            stop = entry_price - (2 * atr)
            take = entry_price + (3 * atr)
        elif signal == -1:  # Short entry
            stop = entry_price + (2 * atr)
            take = entry_price - (3 * atr)
        else:
            return 0, 0, 0
            
        # Calculate position size based on risk
        position_size = (self.risk / abs(entry_price - stop)) / 100_000
        position_size = min(position_size, self.maxpos)
        
        return round(position_size, 2), round(stop, 5), round(take, 5)
    
    def check_pending_orders(self):
        """Check for pending orders using TicketBook"""
        if self.ticketbook:
            from Engine.TicketBook import OrderStatus
            pending = self.ticketbook.get_active_pending_orders(symbol=self.symbol)
            return len(pending) > 0
        else:
            try:
                import MetaTrader5 as mt5
                orders = mt5.orders_get(symbol=self.symbol)
                return orders is not None and len(orders) > 0
            except:
                return False
    
    def check_open_positions(self):
        """Check for open positions using TicketBook"""
        if self.ticketbook:
            from Engine.TicketBook import OrderStatus
            filled = self.ticketbook.get_order_history(symbol=self.symbol, status=OrderStatus.FILLED)
            return not filled.empty
        else:
            try:
                import MetaTrader5 as mt5
                positions = mt5.positions_get(symbol=self.symbol)
                return positions is not None and len(positions) > 0
            except:
                return False
    
    def generate_signal(self):
        """
        Main signal generation logic combining multiple price action patterns
        """
        if len(self.c) < self.lookback:
            return 0, None, 0, 0, 0
            
        # Update support/resistance levels
        self.support_level, self.resistance_level = self.calculate_support_resistance()
        
        # Detect current trend
        trend = self.detect_trend()
        
        # Check for patterns
        pin_bar = self.identify_pin_bar()
        inside_bar = self.identify_inside_bar()
        engulfing = self.identify_engulfing()
        
        current_price = self.c[-1]
        signal = 0
        entry = 0
        pattern = None
        
        # BULLISH SETUPS
        # 1. Bullish pin bar at support in uptrend
        if pin_bar == 1 and trend >= 0:
            if self.support_level and current_price <= self.support_level * 1.002:
                signal = 1
                entry = self.h[-1] + 0.00001  # Enter above pin bar high
                pattern = "Bullish Pin at Support"
                
        # 2. Bullish engulfing in uptrend
        elif engulfing == 1 and trend == 1:
            signal = 1
            entry = self.h[-1] + 0.00001
            pattern = "Bullish Engulfing"
            
        # 3. Inside bar breakout (bullish) in uptrend
        elif inside_bar and trend == 1:
            if self.c[-1] > self.o[-1]:  # Bullish inside bar
                signal = 1
                entry = self.h[-2] + 0.00001  # Enter above mother bar high
                pattern = "Inside Bar Breakout Bull"
        
        # BEARISH SETUPS
        # 1. Bearish pin bar at resistance in downtrend
        elif pin_bar == -1 and trend <= 0:
            if self.resistance_level and current_price >= self.resistance_level * 0.998:
                signal = -1
                entry = self.l[-1] - 0.00001  # Enter below pin bar low
                pattern = "Bearish Pin at Resistance"
                
        # 2. Bearish engulfing in downtrend
        elif engulfing == -1 and trend == -1:
            signal = -1
            entry = self.l[-1] - 0.00001
            pattern = "Bearish Engulfing"
            
        # 3. Inside bar breakout (bearish) in downtrend
        elif inside_bar and trend == -1:
            if self.c[-1] < self.o[-1]:  # Bearish inside bar
                signal = -1
                entry = self.l[-2] - 0.00001  # Enter below mother bar low
                pattern = "Inside Bar Breakout Bear"
        
        # Calculate position sizing and levels if signal found
        if signal != 0:
            position_size, stop, take = self.calculate_position_size_and_levels(signal, entry)
            side = 'buy' if signal == 1 else 'sell'
            
            if self.debug:
                print(f"\n[{self.strategy_name}] SIGNAL DETECTED")
                print(f"Pattern: {pattern}")
                print(f"Trend: {trend}")
                print(f"Entry: {entry}, Stop: {stop}, Take: {take}")
                print(f"Position Size: {position_size}")
            
            return signal, side, entry, stop, take, position_size
        
        return 0, None, 0, 0, 0
    
    def on_bar(self, bar):
        """
        Process new bar and generate trading signals
        """
        orders = []
        
        # Append bar data
        self.t.append(pd.to_datetime(bar.Time))
        self.o.append(bar.Open)
        self.h.append(bar.High)
        self.l.append(bar.Low)
        self.c.append(bar.Close)
        self.v.append(bar.Volume)
        
        # Check for existing positions/orders
        pending_order = self.check_pending_orders()
        open_position = self.check_open_positions()
        
        # Decrement countdown
        if self.countdown > 0:
            self.countdown -= 1
        
        # Only look for new signals if no positions/orders exist
        if not open_position and not pending_order:
            signal, side, entry, stop, take, position_size = self.generate_signal()
            
            if signal != 0:
                # Create order
                order = Order(
                    symbol=self.symbol,
                    side=side,
                    entry=entry,
                    qty=position_size,
                    entry_time=self.t[-1],
                    expiration=self.t[-1] + timedelta(minutes=self.patience),
                    sl=stop,
                    tp=take,
                    strategy_name=self.strategy_name
                )
                orders.append(order)
                
                # Start countdown
                self.countdown = self.patience
                
                if self.debug:
                    print(f"\n[{self.strategy_name}] Order Created:")
                    print(f"Side: {side}, Entry: {entry}")
                    print(f"Stop: {stop}, Take: {take}")
                    print(f"Size: {position_size} lots")
        
        return orders
