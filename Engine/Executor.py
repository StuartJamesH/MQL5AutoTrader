"""
Executor module.

Provides :class:`MT5LiveExecutionHandler`, which is responsible for sending
trade orders to a locally running MetaTrader 5 terminal.

Supported operations
--------------------
* Market (IOC fill-or-kill) order execution.
* Pending stop-order submission and cancellation.
* Pending-ticket tracking and fill detection via deal history or open positions.
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
