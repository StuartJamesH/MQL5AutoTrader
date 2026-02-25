import time
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from DataHandler import Order


class MT5LiveExecutionHandler:
    """
    Single-class live-only MT5 execution handler.

    - Requires the `MetaTrader5` Python package and an initialized MT5 terminal.
    - Executes market orders and submits pending stop orders.
    - Tracks pending tickets and can query fills from deal history or positions.
    """

    def __init__(self, deviation: int = 0, magic: int = 234000):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module not available")

        self.deviation = deviation
        self.magic = magic
        self.pending_orders = {}  # symbol -> ticket
        self.pending_orders_info = {}  # ticket -> Order

        # Initialize MT5 (raise on failure)
        initialized = mt5.initialize()
        if not initialized:
            last = mt5.last_error()
            raise RuntimeError(f"Failed to initialize MT5: {last}")

    # -------------------- Private helpers --------------------

    def _market_price(self, symbol: str, side: str) -> float:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get market tick for symbol {symbol}")
        return tick.ask if side.lower() == "buy" else tick.bid

    def _get_order_type(self, side: str) -> int:
        return mt5.ORDER_TYPE_BUY if side.lower() == "buy" else mt5.ORDER_TYPE_SELL

    def _get_stop_order_type(self, side: str) -> int:
        return mt5.ORDER_TYPE_BUY_STOP if side.lower() == "buy" else mt5.ORDER_TYPE_SELL_STOP

    def _cleanup_pending_order(self, symbol: str, ticket: int):
        if symbol in self.pending_orders and self.pending_orders[symbol] == ticket:
            del self.pending_orders[symbol]
        if ticket in self.pending_orders_info:
            del self.pending_orders_info[ticket]

    def _build_order_request(
        self, action: int, symbol: str, side: str, qty: int, order_type: int,
        price: float = None, sl: float = 0, tp: float = 0, comment: str = ""
    ) -> dict:
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
        """Execute a market (deal) order immediately and return filled Order."""
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
        fill_time = datetime.utcnow().isoformat()
        return Order(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_time=fill_time,
            entry=fill_price,
            expiration=0,
            sl=order.sl,
            tp=order.tp,
        )

    def submit_stop_order(self, order: Order) -> int:
        """Submit a pending stop order and return its ticket."""
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module not available")

        print('Preparing to submit stop order...')  ### PRINT
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
        print(request) ### PRINT

        result = mt5.order_send(request)
        last_error = mt5.last_error()
        if result is None or not hasattr(result, "order"):
            raise RuntimeError(f"Failed to submit pending stop order: {result}, error={last_error}")

        ticket = result.order
        self.track_pending_order(symbol, ticket)
        self.pending_orders_info[ticket] = order
        return ticket

    def delete_order(self, ticket: int) -> bool:
        """Delete pending order by ticket."""
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

        return ok

    def track_pending_order(self, symbol: str, ticket: int):
        self.pending_orders[symbol] = ticket

    def check_and_delete_expired_order(self, symbol: str):
        ticket = self.pending_orders.get(symbol)
        if ticket is not None:
            self.delete_order(ticket)
            # ensure cleaned
            if symbol in self.pending_orders and self.pending_orders[symbol] == ticket:
                del self.pending_orders[symbol]
            if ticket in self.pending_orders_info:
                del self.pending_orders_info[ticket]

    def is_pending_ticket_present(self, ticket: int) -> bool:
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

    def get_fill_for_ticket(self, ticket: int) -> Order:
        """Try to resolve a pending ticket to a filled Order (history -> positions)."""
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
                            if getattr(deal, "type", 0) in (mt5.TRADE_ACTION_DEAL, mt5.ORDER_TYPE_BUY)
                            else "sell"
                        )
                        filled = Order(
                            symbol=deal.symbol,
                            side=side,
                            qty=getattr(deal, "volume", 0),
                            entry_time=datetime.utcnow().isoformat(),
                            entry=float(getattr(deal, "price", 0.0)),
                            expiration=0,
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
                        expiration=0,
                        sl=0,
                        tp=0,
                    )
                    self._cleanup_pending_order(orig.symbol, ticket)
                    return filled
        except Exception:
            pass

        return None
