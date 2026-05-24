"""Engine module.

Provides :class:`Live_Engine`, the top-level orchestrator that drives the live
trading loop by wiring together a data handler, a strategy, and an execution
handler, and :class:`Backtest_Engine`, its offline counterpart for stepwise
CSV-based backtests.
"""
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

_LOG_FORMAT = "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s"


def configure_logging(log_file: str = "trading.log", cloud_log: bool = True) -> None:
    """Configure root-level logging with a console handler, a local file handler,
    and an optional second file handler that mirrors output to the Google Drive
    directory specified by ``CLOUD_LOG_DIR`` in the project ``.env`` file.

    Parameters
    ----------
    log_file:
        Filename for the local log (relative to the current working directory).
    cloud_log:
        When ``True``, also write to ``{CLOUD_LOG_DIR}/{log_file}``.
        If ``CLOUD_LOG_DIR`` is not set in ``.env`` a warning is emitted and
        logging continues with only the local handlers.
    """
    load_dotenv()

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    if cloud_log:
        cloud_dir = os.getenv("CLOUD_LOG_DIR")
        if cloud_dir:
            try:
                os.makedirs(cloud_dir, exist_ok=True)
                cloud_path = os.path.join(cloud_dir, log_file)
                handlers.append(logging.FileHandler(cloud_path, encoding="utf-8"))
            except OSError as exc:
                # Don't prevent the bot from starting if the cloud path is unavailable
                logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT,
                                    handlers=[logging.StreamHandler(sys.stdout)])
                logging.getLogger(__name__).warning(
                    "Could not set up cloud log at '%s': %s — falling back to local only.",
                    cloud_dir, exc,
                )
                handlers = [
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file, encoding="utf-8"),
                ]
        else:
            # Configure temporarily so the warning itself is visible
            logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT,
                                handlers=[logging.StreamHandler(sys.stdout)])
            logging.getLogger(__name__).warning(
                "CLOUD_LOG=True but CLOUD_LOG_DIR is not set in .env — "
                "logging to local file only."
            )
            handlers = [
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding="utf-8"),
            ]
            # Reset so basicConfig below takes effect cleanly
            logging.root.handlers.clear()

    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=handlers)


def _bar_time_as_utc(bar) -> datetime:
    """Extract a naive UTC datetime from the bar's ``Time`` field.

    Handles both timezone-aware :class:`pandas.Timestamp` objects (returned
    by :class:`~DataHandler.MT5DataHandler`) and plain Python datetimes.
    Also handles ISO-8601 strings so that a CSV loaded without ``parse_dates``
    does not silently fall back to wall-clock time.
    Falls back to ``datetime.utcnow()`` if the field is absent or unparseable.
    """
    t = getattr(bar, "Time", None)
    if t is None:
        return datetime.utcnow()
    if hasattr(t, "to_pydatetime"):
        t = t.to_pydatetime()
    elif isinstance(t, str):
        try:
            import pandas as pd
            t = pd.to_datetime(t).to_pydatetime()
        except Exception:
            return datetime.utcnow()
    if isinstance(t, datetime) and t.tzinfo is not None:
        return t.replace(tzinfo=None)
    if isinstance(t, datetime):
        return t
    return datetime.utcnow()


class Live_Engine:
    """Orchestrates the live trading loop.

    Pulls bars from *data_handler* one at a time, passes each bar to *strategy*
    via ``on_bar()``, and forwards any returned orders to *executor* for
    execution.

    Parameters
    ----------
    data_handler :
        Source of market bars.  Must expose a ``get_next_bar()`` generator
        (compatible with both :class:`~DataHandler.DataHandler` and
        :class:`~DataHandler.MT5DataHandler`).
    strategy :
        Trading strategy that implements ``on_bar(bar) -> list[Order]`` and
        exposes an ``order_type`` attribute (``'market'`` or ``'stop'``).
    executor :
        Execution handler that sends orders to the MT5 terminal
        (see :class:`~Executor.MT5LiveExecutionHandler`).
    """

    def __init__(self, data_handler, strategy, executor) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.executor = executor
        self.order_type: str = strategy.order_type

    def run(self) -> None:
        """Start the trading loop.

        Iterates over bars from ``data_handler.get_next_bar()`` until the
        generator is exhausted (replay mode) or indefinitely (live mode).
        Each bar is passed to ``strategy.on_bar()`` and every returned order
        is routed to the executor via the method appropriate for
        ``self.order_type``.
        """
        for bar in self.data_handler.get_next_bar():
            orders = self.strategy.on_bar(bar)
            for order in orders:
                if self.order_type == 'market':
                    self.executor.execute_market_order(order)
                elif self.order_type == 'stop':
                    self.executor.submit_stop_order(order)
            # After all orders for this bar have been submitted, run the
            # per-bar lifecycle batch: expire stale pending orders and detect
            # any fills that materialised since the previous bar.
            bar_time = _bar_time_as_utc(bar)
            self.executor.process_pending_batch(bar_time)
            self.executor.process_position_updates_batch(bar_time)


def configure_backtest_logging(log_file: str = "trading_backtest.log") -> None:
    """Configure root-level logging for a backtest run.

    Writes to both stdout and *log_file*.  No cloud mirroring.

    Parameters
    ----------
    log_file : str, optional
        Local log filename.  Defaults to ``'trading_backtest.log'``.
    """
    configure_logging(log_file=log_file, cloud_log=False)


class Backtest_Engine:
    """Orchestrates a stepwise backtest over CSV / DataFrame data.

    Drop-in counterpart to :class:`Live_Engine` for offline testing.
    Drives the same per-bar loop — data handler → strategy → executor —
    but routes orders to :class:`~Executor.BacktestExecutionHandler`
    instead of the live MT5 terminal.  Fill detection and SL/TP position
    closures are simulated from OHLC bars.

    Typical usage::

        from Engine import Backtest_Engine, configure_backtest_logging
        from DataHandler import DataHandler
        from Executor import BacktestExecutionHandler
        from TicketBook import TicketBook
        from Strategy import TripleBarrierHiLowMulticlass
        import pickle

        configure_backtest_logging()

        model_pack = pickle.load(open("Engine/Model Packs/my_model.pkl", "rb"))
        ticket_book = TicketBook(use_memory_only=True)
        executor    = BacktestExecutionHandler(point_value=10.0, ticket_book=ticket_book)
        data        = DataHandler.from_csv("my_data.csv")
        strategy    = TripleBarrierHiLowMulticlass(
            symbol="US500",
            model_pack=model_pack,
            patience=1,
            mt5_executor=executor,
            ticket_book=ticket_book,
            debug=False,
        )
        engine = Backtest_Engine(data, strategy, executor)
        summary = engine.run()
        print(summary)

    Parameters
    ----------
    data_handler :
        Source of market bars.  Any object with a ``get_next_bar()``
        generator is accepted; use :class:`~DataHandler.DataHandler`
        loaded from a CSV file.
    strategy :
        Trading strategy that implements ``on_bar(bar) -> list[Order]``
        and exposes an ``order_type`` attribute (``'market'`` or
        ``'stop'``).
    executor : BacktestExecutionHandler
        Simulated execution handler.  Must be the same instance whose
        :class:`~TicketBook.TicketBook` is shared with *strategy*.
    log_file : str, optional
        Filename for the local backtest log.  Defaults to
        ``'trading_backtest.log'``.

    Notes
    -----
    The strategy's restricted-hours gate uses ``datetime.now()`` (local
    wall-clock time), not bar time.  Run backtests outside 06:00–10:00
    local time, or set ``debug=False`` on the strategy and verify the
    gate is not blocking signals unexpectedly.
    """

    def __init__(
        self,
        data_handler,
        strategy,
        executor,
        log_file: str = "trading_backtest.log",
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.executor = executor
        self.order_type: str = strategy.order_type
        self._log_file = log_file

    def run(self) -> dict:
        """Run the backtest to completion and return a performance summary.

        Iterates over all bars from ``data_handler.get_next_bar()``,
        passing each bar to ``strategy.on_bar()`` and forwarding returned
        orders to the executor.  After each bar the executor's lifecycle
        batch passes simulate order fills and position exits.

        Returns
        -------
        dict
            Performance summary from
            :meth:`~Executor.BacktestExecutionHandler.get_trade_summary`:
            ``trades``, ``total_pnl``, ``win_rate``, ``avg_pnl``.
        """
        _log = logging.getLogger(__name__)
        _log.info("Backtest started — log=%s", self._log_file)

        bar_count = 0
        total_bars = len(self.data_handler.data)

        for bar in self.data_handler.get_next_bar():
            bar_count += 1
            # Give the executor visibility of the current bar before any
            # batch call so fill / exit simulation is always in sync.
            self.executor.set_current_bar(bar)
            self.executor.set_bar_progress(bar_count, total_bars)

            orders = self.strategy.on_bar(bar)
            for order in orders:
                if self.order_type == "market":
                    self.executor.execute_market_order(order)
                elif self.order_type == "stop":
                    self.executor.submit_stop_order(order)

            bar_time = _bar_time_as_utc(bar)
            self.executor.process_pending_batch(bar_time, current_bar=bar)
            self.executor.process_position_updates_batch(bar_time, current_bar=bar)

            if bar_count % 10_000 == 0:
                _log.info("Backtest progress: %d bars processed", bar_count)

        summary = self.executor.get_trade_summary()
        _log.info(
            "Backtest complete — bars=%d trades=%d total_pnl=%.2f win_rate=%.1f%%",
            bar_count,
            summary["trades"],
            summary["total_pnl"],
            summary["win_rate"] * 100,
        )
        return summary