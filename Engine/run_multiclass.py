"""
run_multiclass.py — Launcher for TripleBarrierHiLowMulticlass.

Uses a single 3-class model (SELL / FLAT / BUY) to generate trade signals.
A trade is only placed when the model's predicted class is SELL or BUY and its
softmax probability meets or exceeds TRADE_THRESHOLD.

Usage:
    python Engine/run_multiclass.py

Replace every value marked with # <-- REPLACE before running.
"""
import logging
import os
import pickle
import signal
import sys

import torch

from Engine import Live_Engine
from DataHandler import MT5DataHandler
from Executor import MT5LiveExecutionHandler
from TicketBook import TicketBook
from StrategyMulticlass import TripleBarrierHiLowMulticlass


# ---------------------------------------------------------------------------
# Logging — writes to both console and a rotating file.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading.log", encoding="utf-8"),
    ],
)
_LOG = logging.getLogger(__name__)


def _load_model_pack(path: str) -> dict:
    """Load and return a model pack from a .pkl file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model pack not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_model(model_pack: dict) -> torch.nn.Module:
    """Instantiate and load weights for the model described by *model_pack*."""
    from Learn.Models import (
        LSTMAttentionSEClassifier,
        LSTMClassifier,
        TCNAttentionSEClassifier,
        TransformerClassifier,
        TransformerSEClassifier,
        HybridLSTMTransformer,
    )

    model_info = model_pack.get("model_info", {})
    model_params = model_pack.get("model_params", {})
    model_type = str(model_info.get("model_type", ""))

    if "TCN" in model_type:
        model_cls = TCNAttentionSEClassifier
    elif "TransformerSE" in model_type:
        model_cls = TransformerSEClassifier
    elif "Hybrid" in model_type:
        model_cls = HybridLSTMTransformer
    elif "Transformer" in model_type:
        model_cls = TransformerClassifier
    elif "LSTM" in model_type or model_type in ("LSTM_TripleBarrier", "LSTM_TripleBarrier_HiLow"):
        model_cls = LSTMAttentionSEClassifier
    else:
        model_cls = LSTMClassifier

    model = model_cls(**model_params)
    model.load_state_dict(model_pack["model"])
    model.eval()
    return model


if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # DATA SOURCE
    # -----------------------------------------------------------------------
    SYMBOL       = "US500.a"        # <-- MT5 symbol name
    TIMEFRAME    = "1min"           # <-- '1min' | '5min' | '15min' | '1h'
    MAXBARS      = 7_000            # <-- max bars to keep in memory (oldest are dropped first)
    MT5_MODE     = "live"           # <-- 'live' | 'replay'
    REPLAY_START = "2026-01-01"     # Only used when MT5_MODE = 'replay'

    # -----------------------------------------------------------------------
    # MODEL PACK
    # -----------------------------------------------------------------------
    MODEL_PACK_PATH = "Engine/Model Packs/US500_1minute_TCN_Multiclass_256seq_20260309_fastma_very_selective_model.pkl"  # <-- e.g. "Engine/Model Packs/US500_multiclass.pkl"

    # -----------------------------------------------------------------------
    # STRATEGY PARAMETERS
    # -----------------------------------------------------------------------
    PATIENCE         = 1            # <-- bars before an unfilled stop order expires
    RISK             = 20.0         # <-- fixed-risk amount per trade in account currency
    MAXPOS           = 5            # <-- maximum position size cap in lots
    TRADE_THRESHOLD  = 0.5          # <-- minimum class probability required to trade
    EMA1_PERIOD      = 8            # <-- fast EMA period (reserved for future filter use)
    EMA2_PERIOD      = 30           # <-- slow EMA period (reserved for future filter use)
    DEBUG            = True        # <-- True for verbose per-bar output
    LOG_TRADES       = True         # <-- False to disable CSV trade logging

    # -----------------------------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------------------------
    DEVIATION = 10                  # <-- max price deviation in points for market orders
    MAGIC     = 235000              # <-- EA magic number — must be unique per running instance

    # -----------------------------------------------------------------------
    # TICKETBOOK (order journal + state store)
    # -----------------------------------------------------------------------
    DB_PATH = f"ticketbook_{SYMBOL.replace('.', '_')}.db"  # one DB file per symbol

    # -----------------------------------------------------------------------
    # Build components
    # -----------------------------------------------------------------------
    _LOG.info("Loading model pack...")
    model_pack = _load_model_pack(MODEL_PACK_PATH)
    model = _build_model(model_pack)
    _LOG.info("Model pack loaded.")

    ticket_book = TicketBook(db_path=DB_PATH)
    data        = MT5DataHandler(symbol=SYMBOL, timeframe=TIMEFRAME, mode=MT5_MODE, start=REPLAY_START, max_bars=MAXBARS)
    executor    = MT5LiveExecutionHandler(deviation=DEVIATION, magic=MAGIC, ticket_book=ticket_book)
    strategy    = TripleBarrierHiLowMulticlass(
        symbol=SYMBOL,
        model=model,
        model_pack=model_pack,
        patience=PATIENCE,
        maxlen=MAXBARS,
        risk=RISK,
        maxpos=MAXPOS,
        trade_threshold=TRADE_THRESHOLD,
        ema1_period=EMA1_PERIOD,
        ema2_period=EMA2_PERIOD,
        mt5_executor=executor,
        data_handler=data,
        debug=DEBUG,
        log=LOG_TRADES,
        ticket_book=ticket_book,
    )
    engine = Live_Engine(data, strategy, executor)

    # -----------------------------------------------------------------------
    # Graceful shutdown on SIGINT / SIGTERM
    # -----------------------------------------------------------------------
    def _shutdown(signum, frame):
        _LOG.info("Shutdown signal received — stopping engine.")
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    _LOG.info(
        "Starting engine — symbol=%s timeframe=%s mode=%s magic=%d db=%s",
        SYMBOL, TIMEFRAME, MT5_MODE, MAGIC, DB_PATH,
    )
    engine.run()
