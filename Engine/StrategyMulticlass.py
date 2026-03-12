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
