"""Engine module.

Provides :class:`Live_Engine`, the top-level orchestrator that drives the live
trading loop by wiring together a data handler, a strategy, and an execution
handler.
"""
from datetime import datetime


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