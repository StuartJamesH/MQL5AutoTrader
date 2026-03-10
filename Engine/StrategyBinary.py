from __future__ import annotations

from collections import deque
import datetime
import os
import csv
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import torch
import talib

from DataHandler import Order

if TYPE_CHECKING:
    from TicketBook import TicketBook


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
