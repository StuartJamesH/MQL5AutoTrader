"""
run_binary.py — Template launcher for TripleBarrierHiLowBinary.

Usage:
    python Engine/run_binary.py

To create a new strategy instance, copy this file and replace all values
marked with # <-- REPLACE with your own.  All tuneable parameters are
declared explicitly at the top of the __main__ block.
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
from StrategyBinary import TripleBarrierHiLowBinary


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


if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # DATA SOURCE
    # -----------------------------------------------------------------------
    SYMBOL      = "EURUSD.a"        # <-- MT5 symbol name
    TIMEFRAME   = "1min"            # <-- '1min' | '5min' | '15min' | '1h'
    MT5_MODE    = "live"            # <-- 'live' | 'replay'
    REPLAY_START = "2025-01-01"     # Only used when MT5_MODE = 'replay'

    # -----------------------------------------------------------------------
    # MODEL PACKS
    # -----------------------------------------------------------------------
    BUY_PACK_PATH  = "placeholder"  # <-- e.g. "Engine/Model Packs/EURUSD_BUY.pkl"
    SELL_PACK_PATH = "placeholder"  # <-- e.g. "Engine/Model Packs/EURUSD_SELL.pkl"

    # -----------------------------------------------------------------------
    # STRATEGY PARAMETERS
    # -----------------------------------------------------------------------
    PATIENCE       = 1              # <-- bars before an unfilled stop order expires
    RISK           = 20.0           # <-- fixed-risk amount per trade in account currency
    MAXPOS         = 1.5            # <-- maximum position size cap in lots
    BUY_THRESHOLD  = 0.5            # <-- minimum buy-model probability to enter long
    SELL_THRESHOLD = 0.5            # <-- minimum sell-model probability to enter short
    EMA1_PERIOD    = 8              # <-- fast EMA period (reserved for future filter use)
    EMA2_PERIOD    = 30             # <-- slow EMA period (reserved for future filter use)
    DEBUG          = False          # <-- True for verbose per-bar output
    LOG_TRADES     = True           # <-- False to disable CSV trade logging

    # -----------------------------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------------------------
    DEVIATION   = 10                # <-- max price deviation in points for market orders
    MAGIC       = 234000            # <-- EA magic number — must be unique per running instance

    # -----------------------------------------------------------------------
    # TICKETBOOK (order journal + state store)
    # -----------------------------------------------------------------------
    DB_PATH = f"ticketbook_{SYMBOL.replace('.', '_')}.db"  # one DB file per symbol

    # -----------------------------------------------------------------------
    # Build components
    # -----------------------------------------------------------------------
    _LOG.info("Loading model packs...")
    buy_pack  = _load_model_pack(BUY_PACK_PATH)
    sell_pack = _load_model_pack(SELL_PACK_PATH)
    _LOG.info("Model packs loaded.")

    ticket_book = TicketBook(db_path=DB_PATH)
    data        = MT5DataHandler(symbol=SYMBOL, timeframe=TIMEFRAME, mode=MT5_MODE, start=REPLAY_START)
    executor    = MT5LiveExecutionHandler(deviation=DEVIATION, magic=MAGIC, ticket_book=ticket_book)
    strategy    = TripleBarrierHiLowBinary(
        symbol=SYMBOL,
        buy_model_pack=buy_pack,
        sell_model_pack=sell_pack,
        patience=PATIENCE,
        risk=RISK,
        maxpos=MAXPOS,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
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
