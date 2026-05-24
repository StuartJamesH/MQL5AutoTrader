"""
Executor module.

Provides :class:`MT5LiveExecutionHandler`, which is responsible for sending
trade orders to a locally running MetaTrader 5 terminal, and
:class:`BacktestExecutionHandler`, a pure-Python simulation executor used
by :class:`~Engine.Backtest_Engine` for stepwise CSV-based backtests.

Supported operations
--------------------
* Market (IOC fill-or-kill) order execution.
* Pending stop-order submission and cancellation.
* Pending-ticket tracking and fill detection via deal history or open positions.
* Simulated stop-order fills and SL/TP closures from OHLC bar data (backtest).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover – package absent in non-MT5 environments
    mt5 = None

from DataHandler import Order
from TicketBook import TicketBook, OrderStatus

_LOG = logging.getLogger(__name__)


class MT5LiveExecutionHandler:
    """Live-only MT5 execution handler.

    Interfaces with a locally running MetaTrader 5 terminal via the
    ``MetaTrader5`` Python package.  All public methods require that the
    terminal is running and a trading account is logged in.

    Parameters
    ----------
    deviation : int, optional
        Maximum allowed price deviation in points for market orders.
        Defaults to ``0``.
    magic : int, optional
        Expert Advisor magic number attached to every order sent by this
        handler.  Defaults to ``234000``.

    Raises
    ------
    RuntimeError
        If the ``MetaTrader5`` package is unavailable or ``mt5.initialize()``
        fails.
    """

    def __init__(self, deviation: int = 0, magic: int = 234000, ticket_book: Optional[TicketBook] = None) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module not available")

        self.deviation = deviation
        self.magic = magic
        self.ticket_book = ticket_book
        self.pending_orders = {}  # symbol -> ticket
        self.pending_orders_info = {}  # ticket -> Order

        # Initialize MT5 (raise on failure)
        initialized = mt5.initialize()
        if not initialized:
            last = mt5.last_error()
            raise RuntimeError(f"Failed to initialize MT5: {last}")

    # -------------------- Public helpers --------------------

    def get_point_value(self, symbol: str) -> float:
        """Return the account-currency value of 1 lot per 1.0 price-unit move.

        Used by strategies for dollar-risk position sizing::

            lots = risk_dollars / (sl_distance * point_value)

        Raises
        ------
        RuntimeError
            If symbol info is unavailable from the MT5 terminal.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Cannot get symbol info for {symbol}")
        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value
        if tick_size <= 0:
            raise RuntimeError(f"Invalid tick_size={tick_size} for {symbol}")
        return tick_value / tick_size

    # -------------------- Private helpers --------------------

    def _market_price(self, symbol: str, side: str) -> float:
        """Return the current ask (buy) or bid (sell) price for *symbol*."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get market tick for symbol {symbol}")
        return tick.ask if side.lower() == "buy" else tick.bid

    def _get_order_type(self, side: str) -> int:
        """Return the MT5 market-order type constant for *side*."""
        return mt5.ORDER_TYPE_BUY if side.lower() == "buy" else mt5.ORDER_TYPE_SELL

    def _get_stop_order_type(self, side: str) -> int:
        """Return the MT5 pending stop-order type constant for *side*."""
        return mt5.ORDER_TYPE_BUY_STOP if side.lower() == "buy" else mt5.ORDER_TYPE_SELL_STOP

    def _cleanup_pending_order(self, symbol: str, ticket: int) -> None:
        """Remove *ticket* from the internal pending-order indexes."""
        if symbol in self.pending_orders and self.pending_orders[symbol] == ticket:
            del self.pending_orders[symbol]
        if ticket in self.pending_orders_info:
            del self.pending_orders_info[ticket]

    def _build_order_request(
        self,
        action: int,
        symbol: str,
        side: str,
        qty: int,
        order_type: int,
        price: Optional[float] = None,
        sl: float = 0,
        tp: float = 0,
        comment: str = "",
    ) -> dict:
        """Assemble and return an MT5 order request dictionary.

        Parameters
        ----------
        action : int
            MT5 trade action constant (e.g. ``mt5.TRADE_ACTION_DEAL``).
        symbol : str
            Instrument ticker.
        side : str
            ``'buy'`` or ``'sell'`` (for context only; *order_type* sets the
            actual direction).
        qty : int
            Volume in lots.
        order_type : int
            MT5 order type constant.
        price : float, optional
            Limit / stop trigger price.  Omitted from the request when ``None``.
        sl : float, optional
            Stop-loss price.  ``0`` disables stop-loss.
        tp : float, optional
            Take-profit price.  ``0`` disables take-profit.
        comment : str, optional
            Free-text comment attached to the order.
        """
        request = {
            "action": action,
            "symbol": symbol,
            "side": side,
            "volume": float(qty),
            "type": order_type,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        if price is not None:
            request["price"] = price
        if sl:
            request["sl"] = sl
        if tp:
            request["tp"] = tp
        return request

    # -------------------- Public API --------------------

    def execute_market_order(self, order: Order) -> Order:
        """Execute a market order and return the filled :class:`~DataHandler.Order`.

        Sends a ``TRADE_ACTION_DEAL`` request with ``ORDER_FILLING_IOC`` and
        returns a new :class:`~DataHandler.Order` populated with the actual
        fill price and timestamp.

        Parameters
        ----------
        order : Order
            Template order carrying *symbol*, *side*, *qty*, *sl*, and *tp*.
            The *entry* field is ignored; the live market price is used instead.

        Returns
        -------
        Order
            A new order with ``entry`` set to the actual fill price.

        Raises
        ------
        RuntimeError
            If ``MetaTrader5`` is unavailable, the symbol is not found, or
            the terminal does not return ``TRADE_RETCODE_DONE``.
        """
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module not available")

        symbol = order.symbol
        side = order.side.lower()
        qty = order.qty

        # Validate symbol
        if mt5.symbol_info(symbol) is None:
            raise RuntimeError(f"Symbol {symbol} not available in MT5 terminal")

        price = self._market_price(symbol, side)
        order_type = self._get_order_type(side)

        request = self._build_order_request(
            action=mt5.TRADE_ACTION_DEAL,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            sl=order.sl,
            tp=order.tp,
            comment="market_order",
        )
        request["type_filling"] = mt5.ORDER_FILLING_IOC

        result = mt5.order_send(request)
        last_error = mt5.last_error()
        if result is None or getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"mt5.order_send failed: {result}, error={last_error}")

        fill_price = float(getattr(result, "price", price))
        fill_time = datetime.utcnow()

        if self.ticket_book is not None:
            order_ticket = getattr(result, "order", 0)
            self.ticket_book.record_order(
                ticket=order_ticket,
                symbol=symbol,
                side=side,
                qty=float(qty),
                entry_price=fill_price,
                sl=order.sl or 0.0,
                tp=order.tp or 0.0,
                submission_time=fill_time,
                expiration_time=None,
                strategy_name="",
                status=OrderStatus.FILLED,
            )
            self.ticket_book.record_fill(
                ticket=order_ticket,
                fill_price=fill_price,
                fill_time=fill_time,
            )
            _LOG.info(
                "Market fill recorded: ticket=%d symbol=%s side=%s price=%.5f",
                order_ticket, symbol, side, fill_price,
            )

        return Order(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_time=fill_time.isoformat(),
            entry=fill_price,
            expiration=None,
            sl=order.sl,
            tp=order.tp,
        )

    def submit_stop_order(self, order: Order) -> int:
        """Submit a pending stop order to the MT5 terminal.

        Parameters
        ----------
        order : Order
            Order whose *entry* field is used as the stop trigger price.

        Returns
        -------
        int
            MT5 ticket number of the accepted pending order.

        Raises
        ------
        RuntimeError
            If ``MetaTrader5`` is unavailable or the order is rejected by the
            terminal.
        """
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module not available")

        _LOG.debug("Submitting stop order: %s %s @ %.5f", order.symbol, order.side, order.entry)
        symbol = order.symbol
        side = order.side.lower()
        qty = order.qty
        price = order.entry
        order_type = self._get_stop_order_type(side)

        request = self._build_order_request(
            action=mt5.TRADE_ACTION_PENDING,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            sl=order.sl,
            tp=order.tp,
            comment="pending_stop_order",
        )
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        _LOG.debug("Stop order request: %s", request)

        result = mt5.order_send(request)
        last_error = mt5.last_error()
        if result is None or not hasattr(result, "order"):
            raise RuntimeError(f"Failed to submit pending stop order: {result}, error={last_error}")

        ticket = result.order
        self.track_pending_order(symbol, ticket)
        self.pending_orders_info[ticket] = order

        if self.ticket_book is not None:
            from datetime import datetime as _dt
            self.ticket_book.record_order(
                ticket=ticket,
                symbol=symbol,
                side=side,
                qty=float(qty),
                entry_price=price,
                sl=order.sl or 0.0,
                tp=order.tp or 0.0,
                submission_time=_dt.utcnow(),
                expiration_time=order.expiration,
                strategy_name="",
            )
            _LOG.info(
                "Stop order recorded: ticket=%d symbol=%s side=%s price=%.5f",
                ticket, symbol, side, price,
            )

        return ticket

    def delete_order(self, ticket: int, cancel_reason: str = "manual") -> bool:
        """Cancel a pending order by its MT5 ticket number.

        Parameters
        ----------
        ticket : int
            MT5 ticket of the pending order to cancel.
        cancel_reason : str, optional
            Reason string recorded in the TicketBook
            (e.g. ``'manual'``, ``'expired'``).  Defaults to
            ``'manual'``.

        Returns
        -------
        bool
            ``True`` if the terminal confirmed cancellation
            (``TRADE_RETCODE_DONE``), ``False`` otherwise.
        """
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module not available")

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
            "magic": self.magic,
            "comment": "cancel_order",
        }

        result = mt5.order_send(request)
        ok = result is not None and getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE

        if ok and ticket in self.pending_orders_info:
            orig = self.pending_orders_info.pop(ticket)
            self._cleanup_pending_order(orig.symbol, ticket)

        if ok and self.ticket_book is not None:
            self.ticket_book.record_cancellation(ticket, reason=cancel_reason)
            _LOG.info("Order cancellation recorded: ticket=%d reason=%s", ticket, cancel_reason)

        return ok

    def track_pending_order(self, symbol: str, ticket: int) -> None:
        """Register *ticket* as the active pending order for *symbol*."""
        self.pending_orders[symbol] = ticket

    def check_and_delete_expired_order(self, symbol: str) -> None:
        """Cancel and clean up the active pending order for *symbol*, if any."""
        ticket = self.pending_orders.get(symbol)
        if ticket is not None:
            self.delete_order(ticket)
            # ensure cleaned
            if symbol in self.pending_orders and self.pending_orders[symbol] == ticket:
                del self.pending_orders[symbol]
            if ticket in self.pending_orders_info:
                del self.pending_orders_info[ticket]

    def is_pending_ticket_present(self, ticket: int) -> bool:
        """Return ``True`` if *ticket* is still listed in the MT5 pending orders queue."""
        if ticket is None:
            return False
        try:
            orders = mt5.orders_get()
            if orders is None:
                return False
            for o in orders:
                if getattr(o, "ticket", None) == ticket:
                    return True
        except Exception:
            pass
        return False

    def get_fill_for_ticket(self, ticket: int) -> Optional[Order]:
        """Attempt to resolve a pending ticket to a filled :class:`~DataHandler.Order`.

        Searches MT5 deal history for the past 24 hours first, then falls back
        to checking open positions for the original order's symbol.

        Parameters
        ----------
        ticket : int
            MT5 ticket of the pending order to look up.

        Returns
        -------
        Order or None
            A filled order if the pending order has been executed, otherwise
            ``None``.
        """
        if ticket is None:
            return None

        orig = self.pending_orders_info.get(ticket)

        # Search recent deal history
        try:
            now = datetime.utcnow()
            start_time = now - timedelta(days=1)
            deals = mt5.history_deals_get(start_time, now)
            if deals:
                for deal in deals:
                    deal_order = getattr(deal, "order", None)
                    deal_ticket = getattr(deal, "ticket", None)
                    if deal_order == ticket or deal_ticket == ticket:
                        side = (
                            "buy"
                            if getattr(deal, "type", mt5.DEAL_TYPE_SELL) == mt5.DEAL_TYPE_BUY
                            else "sell"
                        )
                        filled = Order(
                            symbol=deal.symbol,
                            side=side,
                            qty=getattr(deal, "volume", 0),
                            entry_time=datetime.utcnow().isoformat(),
                            entry=float(getattr(deal, "price", 0.0)),
                            expiration=None,
                            sl=0,
                            tp=0,
                        )
                        if orig:
                            self._cleanup_pending_order(orig.symbol, ticket)
                        return filled
        except Exception:
            pass

        # Fallback: search current positions by symbol
        try:
            if orig is not None:
                positions = mt5.positions_get(symbol=orig.symbol)
                if positions:
                    pos = positions[0]
                    side = "buy" if pos.volume > 0 else "sell"
                    filled = Order(
                        symbol=pos.symbol,
                        side=side,
                        qty=abs(pos.volume),
                        entry_time=datetime.utcnow().isoformat(),
                        entry=float(getattr(pos, "price_open", 0.0)),
                        expiration=None,
                        sl=0,
                        tp=0,
                    )
                    self._cleanup_pending_order(orig.symbol, ticket)
                    return filled
        except Exception:
            pass

        return None

    def process_pending_batch(self, current_time: Optional[datetime] = None) -> None:
        """Process pending order lifecycle updates for the current bar.

        Should be called once per bar **after** all new orders for that bar
        have been submitted to MT5.  Performs two passes:

        1. **Expiry pass** – any pending order whose
           :attr:`~TicketBook.TicketRecord.expiration_time` has elapsed is
           cancelled via MT5 and recorded as ``CANCELLED`` with reason
           ``'expired'``.
        2. **Fill detection pass** – for every remaining active pending order
           that has left the MT5 pending queue, deal history and open
           positions are searched.  If a fill is found it is recorded;
           otherwise the order is marked ``'broker_cancelled'``.

        Parameters
        ----------
        current_time : datetime, optional
            Timestamp used for expiry evaluation.  Passing the bar-close time
            ensures consistent behaviour in both live and replay modes.
            Defaults to ``datetime.utcnow()`` when omitted.
        """
        if self.ticket_book is None:
            return

        if current_time is None:
            current_time = datetime.utcnow()

        # --- Pass 1: cancel expired orders ---
        expired_tickets = self.ticket_book.get_expired_orders(current_time)
        for ticket in expired_tickets:
            if self.is_pending_ticket_present(ticket):
                # Order still pending in MT5 — cancel it.  delete_order also
                # records the cancellation in the TicketBook.
                ok = self.delete_order(ticket, cancel_reason="expired")
                if not ok:
                    _LOG.warning("Failed to cancel expired order: ticket=%d", ticket)
            # If the ticket is no longer in MT5 it was either filled or cancelled
            # by the broker; leave it in active_pending so Pass 2 can classify it.

        # --- Pass 2: detect fills and broker cancellations ---
        for record in list(self.ticket_book.get_active_pending_orders()):
            ticket = record.ticket
            if self.is_pending_ticket_present(ticket):
                continue  # still in the MT5 pending queue; nothing to do

            filled = self.get_fill_for_ticket(ticket)
            if filled is not None:
                self.ticket_book.record_fill(
                    ticket=ticket,
                    fill_price=filled.entry,
                    fill_time=datetime.utcnow(),
                )
                _LOG.info(
                    "Fill recorded: ticket=%d symbol=%s side=%s price=%.5f",
                    ticket, record.symbol, record.side, filled.entry,
                )
            else:
                # Not in fills or positions — broker cancelled it
                self.ticket_book.record_cancellation(ticket, reason="broker_cancelled")
                _LOG.info("Broker cancellation recorded: ticket=%d", ticket)
        
    def process_position_updates_batch(self, current_time: Optional[datetime] = None) -> None:
        """Process updates for currently open positions.

        Should be called once per bar after all new orders for that bar have been
        submitted to MT5.  For every currently open position, checks if it has
        been closed since the previous update (i.e. by an independent market
        order or by an attached stop-loss / take-profit) and records any
        detected closure in the TicketBook.

        Parameters
        ----------
        current_time : datetime, optional
            Timestamp used for update evaluation.  Passing the bar-close time
            ensures consistent behaviour in both live and replay modes.
            Defaults to ``datetime.utcnow()`` when omitted.
        """
        if self.ticket_book is None:
            return

        if current_time is None:
            current_time = datetime.utcnow()

        for record in list(self.ticket_book.get_open_positions()):
            ticket = record.ticket
            try:
                # Check whether this specific position is still open in MT5.
                # Using ticket= is more precise than symbol= and works correctly
                # on both netting and hedging accounts.
                positions = mt5.positions_get(ticket=ticket)
                if positions:
                    continue  # position still open, nothing to do

                # Position is gone — search deal history for the closing deal.
                # MT5 represents a close as a deal with DEAL_ENTRY_OUT whose
                # position_id matches the original fill ticket.
                now = datetime.utcnow()
                deals = mt5.history_deals_get(now - timedelta(days=7), now)

                close_price = 0.0
                close_pnl = 0.0
                close_swap = 0.0
                close_time_dt = current_time

                if deals:
                    for deal in deals:
                        if (
                            getattr(deal, "position_id", None) == ticket
                            and getattr(deal, "entry", None) == mt5.DEAL_ENTRY_OUT
                        ):
                            close_price = float(getattr(deal, "price", 0.0))
                            close_pnl = float(getattr(deal, "profit", 0.0))
                            close_swap = float(getattr(deal, "swap", 0.0))
                            deal_ts = getattr(deal, "time", None)
                            if deal_ts:
                                close_time_dt = datetime.utcfromtimestamp(deal_ts)
                            break
                    else:
                        _LOG.warning(
                            "No closing deal found in history for ticket=%d — recording closure with zero values",
                            ticket,
                        )

                self.ticket_book.record_close(
                    ticket=ticket,
                    close_price=close_price,
                    close_time=close_time_dt,
                    pnl=close_pnl,
                    swap=close_swap,
                )
                _LOG.info(
                    "Position closure recorded: ticket=%d symbol=%s side=%s close_price=%.5f pnl=%.2f swap=%.2f",
                    ticket, record.symbol, record.side, close_price, close_pnl, close_swap,
                )
            except Exception as e:
                _LOG.warning("Error checking position closure for ticket=%d: %s", ticket, e)


# ---------------------------------------------------------------------------
# Backtest execution handler
# ---------------------------------------------------------------------------

def _bar_to_datetime(bar) -> datetime:
    """Extract a naive UTC :class:`datetime` from a bar's ``Time`` field."""
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


class BacktestExecutionHandler:
    """Simulated execution handler for CSV-based stepwise backtests.

    Drop-in counterpart to :class:`MT5LiveExecutionHandler` for offline
    testing.  Requires no MetaTrader 5 connection; all fills and position
    closures are simulated from OHLC bar data by
    :class:`~Engine.Backtest_Engine`.

    Fill simulation rules
    ---------------------
    * **BUY STOP** @ *entry*: triggered when ``bar.High >= entry``.
      Fill price = *entry* if ``bar.Open < entry`` else ``bar.Open``
      (gap-up scenario).
    * **SELL STOP** @ *entry*: triggered when ``bar.Low <= entry``.
      Fill price = *entry* if ``bar.Open > entry`` else ``bar.Open``
      (gap-down scenario).

    Position exit simulation
    ------------------------
    * Long SL: ``bar.Low <= sl``  → close at ``min(sl, bar.Open)``
    * Long TP: ``bar.High >= tp`` → close at ``tp``
    * Short SL: ``bar.High >= sl`` → close at ``max(sl, bar.Open)``
    * Short TP: ``bar.Low <= tp``  → close at ``tp``
    * When both SL and TP are triggered on the same bar, SL takes
      priority (conservative assumption).

    Parameters
    ----------
    point_value : float, optional
        Default dollar value per 1.0 price-unit move per lot.  Used for
        P&L calculation and passed back to the strategy via
        :meth:`get_point_value`.  Defaults to ``1.0``.
    ticket_book : TicketBook, optional
        Shared order-state journal.  The same instance must be passed to
        the strategy so that ``has_pending_order`` / ``has_open_position``
        queries reflect simulated state.  When omitted a memory-only
        :class:`~TicketBook.TicketBook` is created automatically.
    """

    def __init__(
        self,
        point_value: float = 1.0,
        ticket_book: Optional[TicketBook] = None,
    ) -> None:
        self._default_point_value = point_value
        self._point_value_map: dict = {}

        if ticket_book is None:
            ticket_book = TicketBook(use_memory_only=True)
        self.ticket_book = ticket_book

        self._next_ticket: int = 1
        # Internal cache of pending Order objects keyed by ticket id.
        self._pending_orders: dict = {}
        # Bar set by Backtest_Engine before each batch call.
        self._current_bar = None

        # Aggregate statistics updated as positions close.
        self.total_pnl: float = 0.0
        self.closed_trades: list = []

        # Progress counters updated by Backtest_Engine each bar.
        self._bar_idx: int = 0
        self._bar_total: int = 0

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_point_value(self, symbol: str, value: float) -> None:
        """Override the point value for *symbol* (e.g. ``10.0`` for US500)."""
        self._point_value_map[symbol] = value

    def get_point_value(self, symbol: str) -> float:
        """Return the point value for *symbol*, falling back to the default."""
        return self._point_value_map.get(symbol, self._default_point_value)

    def set_current_bar(self, bar) -> None:
        """Set the bar used for fill simulation.  Called by :class:`~Engine.Backtest_Engine`."""
        self._current_bar = bar

    def set_bar_progress(self, current: int, total: int) -> None:
        """Record the current bar index and total bar count for log annotations.

        Called by :class:`~Engine.Backtest_Engine` at the start of each bar
        so that ticket log lines include ``(candle X of Y)`` context.

        Parameters
        ----------
        current : int
            1-based index of the bar currently being processed.
        total : int
            Total number of bars in the dataset.
        """
        self._bar_idx = current
        self._bar_total = total

    def _progress_tag(self) -> str:
        """Return a ``' (candle X of Y)'`` suffix, or ``''`` if totals are unset."""
        if self._bar_total > 0:
            return f" (candle {self._bar_idx} of {self._bar_total})"
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alloc_ticket(self) -> int:
        ticket = self._next_ticket
        self._next_ticket += 1
        return ticket

    # ------------------------------------------------------------------
    # Public API (mirrors MT5LiveExecutionHandler)
    # ------------------------------------------------------------------

    def execute_market_order(self, order: Order) -> Order:
        """Simulate an immediate market fill at the current bar's open price.

        Parameters
        ----------
        order : Order
            Template order.  The *entry* field is ignored; fill price is
            taken from ``bar.Open``.

        Returns
        -------
        Order
            A new :class:`~DataHandler.Order` with ``entry`` set to the
            simulated fill price.
        """
        bar = self._current_bar
        fill_price = float(bar.Open) if bar is not None else (order.entry or 0.0)
        fill_time = _bar_to_datetime(bar) if bar is not None else datetime.utcnow()

        ticket = self._alloc_ticket()
        self.ticket_book.record_order(
            ticket=ticket,
            symbol=order.symbol,
            side=order.side,
            qty=float(order.qty),
            entry_price=order.entry or fill_price,
            sl=order.sl or 0.0,
            tp=order.tp or 0.0,
            submission_time=fill_time,
            expiration_time=None,
            strategy_name="",
            status=OrderStatus.FILLED,
        )
        self.ticket_book.record_fill(ticket=ticket, fill_price=fill_price, fill_time=fill_time)

        _LOG.info(
            "Backtest market fill: ticket=%d %s %s @ %.5f%s",
            ticket, order.symbol, order.side, fill_price, self._progress_tag(),
        )
        return Order(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            entry_time=fill_time.isoformat(),
            entry=fill_price,
            expiration=None,
            sl=order.sl,
            tp=order.tp,
        )

    def submit_stop_order(self, order: Order) -> int:
        """Register a pending stop order for bar-by-bar fill simulation.

        Parameters
        ----------
        order : Order
            Stop order with *entry* set to the stop trigger price and
            *expiration* set to the bar-time cutoff.

        Returns
        -------
        int
            Simulated ticket number.
        """
        ticket = self._alloc_ticket()
        self._pending_orders[ticket] = order

        submit_time = (
            _bar_to_datetime(self._current_bar)
            if self._current_bar is not None
            else datetime.utcnow()
        )

        self.ticket_book.record_order(
            ticket=ticket,
            symbol=order.symbol,
            side=order.side,
            qty=float(order.qty),
            entry_price=order.entry,
            sl=order.sl or 0.0,
            tp=order.tp or 0.0,
            submission_time=submit_time,
            expiration_time=order.expiration,
            strategy_name="",
        )

        _LOG.info(
            "Backtest stop order submitted: ticket=%d %s %s @ %.5f%s",
            ticket, order.symbol, order.side, order.entry, self._progress_tag(),
        )
        return ticket

    def process_pending_batch(
        self,
        bar_time: Optional[datetime] = None,
        current_bar=None,
    ) -> None:
        """Expire stale pending orders and simulate stop fills for the current bar.

        Should be called once per bar **after** new orders have been
        submitted for that bar, matching the live-engine call pattern.

        Parameters
        ----------
        bar_time : datetime, optional
            Bar-close timestamp used for expiry evaluation.  Defaults to
            ``datetime.utcnow()``.
        current_bar : bar namedtuple, optional
            OHLC bar used for fill-price simulation.  Falls back to the
            bar set via :meth:`set_current_bar` if not provided.
        """
        if current_bar is not None:
            self._current_bar = current_bar

        if bar_time is None:
            bar_time = datetime.utcnow()

        # --- Pass 1: simulate fills from bar High / Low ---
        # Fills are checked BEFORE expiry so that an order whose expiration
        # timestamp equals the current bar time is still eligible to fill on
        # that bar (e.g. patience=1 on M1 data: order expires at bar N+1 but
        # should fill if bar N+1's price action reaches the stop level).
        bar = self._current_bar
        if bar is not None:
            bar_open = float(bar.Open)
            bar_high = float(bar.High)
            bar_low  = float(bar.Low)

            for record in list(self.ticket_book.get_active_pending_orders()):
                ticket = record.ticket
                if ticket not in self._pending_orders:
                    continue

                entry = record.entry_price
                side  = record.side.lower()
                filled = False
                fill_price = 0.0

                if side == "buy":
                    # BUY STOP: triggered when the bar's High reaches the stop price.
                    if bar_high >= entry:
                        # If the bar opened above the entry (gap up), fill at open.
                        fill_price = bar_open if bar_open >= entry else entry
                        filled = True
                elif side == "sell":
                    # SELL STOP: triggered when the bar's Low reaches the stop price.
                    if bar_low <= entry:
                        # If the bar opened below the entry (gap down), fill at open.
                        fill_price = bar_open if bar_open <= entry else entry
                        filled = True

                if filled:
                    fill_time = _bar_to_datetime(bar)
                    self.ticket_book.record_fill(
                        ticket=ticket, fill_price=fill_price, fill_time=fill_time
                    )
                    self._pending_orders.pop(ticket, None)
                    _LOG.info(
                        "Backtest fill: ticket=%d %s %s entry=%.5f fill=%.5f%s",
                        ticket, record.symbol, side, entry, fill_price, self._progress_tag(),
                    )

        # --- Pass 2: expire orders whose deadline has elapsed ---
        # Only orders that were NOT filled in Pass 1 will still be in
        # _active_pending (record_fill removes them), so no double-processing.
        for ticket in self.ticket_book.get_expired_orders(bar_time):
            order = self._pending_orders.get(ticket)
            self.ticket_book.record_cancellation(ticket, reason="expired")
            self._pending_orders.pop(ticket, None)
            _LOG.info(
                "Backtest cancelled (expired): ticket=%d %s %s @ %.5f%s",
                ticket,
                order.symbol if order else "?",
                order.side if order else "?",
                order.entry if order else 0.0,
                self._progress_tag(),
            )

    def process_position_updates_batch(
        self,
        bar_time: Optional[datetime] = None,
        current_bar=None,
    ) -> None:
        """Simulate SL/TP closures for all currently open positions.

        Should be called once per bar **after** :meth:`process_pending_batch`,
        matching the live-engine call pattern.

        Parameters
        ----------
        bar_time : datetime, optional
            Bar-close timestamp recorded against any closed position.
            Defaults to ``datetime.utcnow()``.
        current_bar : bar namedtuple, optional
            OHLC bar used to check SL/TP trigger levels.  Falls back to
            the bar set via :meth:`set_current_bar` if not provided.
        """
        if current_bar is not None:
            self._current_bar = current_bar

        if bar_time is None:
            bar_time = datetime.utcnow()

        bar = self._current_bar
        if bar is None:
            return

        bar_open = float(bar.Open)
        bar_high = float(bar.High)
        bar_low  = float(bar.Low)

        for record in list(self.ticket_book.get_open_positions()):
            ticket     = record.ticket
            side       = record.side.lower()
            sl         = record.sl or 0.0
            tp         = record.tp or 0.0
            fill_price = (
                record.fill_price
                if record.fill_price is not None
                else record.entry_price
            )

            close_price: Optional[float] = None
            close_reason: str = ""

            if side == "buy":
                sl_hit = sl > 0 and bar_low <= sl
                tp_hit = tp > 0 and bar_high >= tp
                if sl_hit:
                    close_price  = min(sl, bar_open)
                    close_reason = "sl"
                elif tp_hit:
                    close_price  = tp
                    close_reason = "tp"

            elif side == "sell":
                sl_hit = sl > 0 and bar_high >= sl
                tp_hit = tp > 0 and bar_low <= tp
                if sl_hit:
                    close_price  = max(sl, bar_open)
                    close_reason = "sl"
                elif tp_hit:
                    close_price  = tp
                    close_reason = "tp"

            if close_price is None:
                continue

            point_value = self.get_point_value(record.symbol)
            pnl = (
                (close_price - fill_price) * record.qty * point_value
                if side == "buy"
                else (fill_price - close_price) * record.qty * point_value
            )

            self.ticket_book.record_close(
                ticket=ticket,
                close_price=close_price,
                close_time=bar_time,
                pnl=pnl,
            )
            self.total_pnl += pnl

            closed_record = self.ticket_book.get_order(ticket)
            if closed_record is not None:
                self.closed_trades.append(closed_record)

            _LOG.info(
                "Backtest close (%s): ticket=%d %s %s fill=%.5f close=%.5f pnl=%.2f%s",
                close_reason, ticket, record.symbol, side,
                fill_price, close_price, pnl, self._progress_tag(),
            )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_trade_summary(self) -> dict:
        """Return a performance summary over all closed trades.

        Returns
        -------
        dict
            Includes overall and side-split counts, P&L statistics, profit
            factor, per-trade extremes, and consecutive win/loss streaks.
        """
        n = len(self.closed_trades)
        _zero: dict = {
            "trades": 0, "buy_trades": 0, "sell_trades": 0,
            "total_pnl": 0.0, "avg_pnl": 0.0,
            "win_rate": 0.0, "buy_win_rate": 0.0, "sell_win_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
            "max_win": 0.0, "max_loss": 0.0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        }
        if n == 0:
            return _zero

        buys  = [t for t in self.closed_trades if t.side.lower() == "buy"]
        sells = [t for t in self.closed_trades if t.side.lower() == "sell"]

        pnls        = [(t.pnl or 0.0) for t in self.closed_trades]
        buy_pnls    = [(t.pnl or 0.0) for t in buys]
        sell_pnls   = [(t.pnl or 0.0) for t in sells]

        total_wins  = sum(1 for p in pnls     if p > 0)
        buy_wins    = sum(1 for p in buy_pnls  if p > 0)
        sell_wins   = sum(1 for p in sell_pnls if p > 0)

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss   = abs(sum(p for p in pnls if p < 0))

        # Consecutive win/loss streaks
        max_cons_wins = max_cons_losses = 0
        cur_wins = cur_losses = 0
        for p in pnls:
            if p > 0:
                cur_wins += 1
                cur_losses = 0
                max_cons_wins = max(max_cons_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_cons_losses = max(max_cons_losses, cur_losses)

        return {
            "trades":               n,
            "buy_trades":           len(buys),
            "sell_trades":          len(sells),
            "total_pnl":            round(self.total_pnl, 2),
            "avg_pnl":              round(self.total_pnl / n, 2),
            "win_rate":             round(total_wins / n, 4),
            "buy_win_rate":         round(buy_wins / len(buys),   4) if buys  else 0.0,
            "sell_win_rate":        round(sell_wins / len(sells), 4) if sells else 0.0,
            "gross_profit":         round(gross_profit, 2),
            "gross_loss":           round(gross_loss, 2),
            "profit_factor":        round(gross_profit / gross_loss, 4) if gross_loss > 0 else 0.0,
            "max_win":              round(max(pnls), 2),
            "max_loss":             round(min(pnls), 2),
            "max_consecutive_wins": max_cons_wins,
            "max_consecutive_losses": max_cons_losses,
        }
