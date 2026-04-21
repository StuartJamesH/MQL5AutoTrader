"""Engine module.

Provides :class:`Live_Engine`, the top-level orchestrator that drives the live
trading loop by wiring together a data handler, a strategy, and an execution
handler.
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
    Falls back to ``datetime.utcnow()`` if the field is absent.
    """
    t = getattr(bar, "Time", None)
    if t is None:
        return datetime.utcnow()
    if hasattr(t, "to_pydatetime"):
        t = t.to_pydatetime()
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