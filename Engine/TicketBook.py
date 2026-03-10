"""
TicketBook module.

Provides a dual-storage (in-memory + SQLite) journal for recording and
querying the full lifecycle of trade orders.

Primary responsibilities
------------------------
1. Record all orders with their tickets, expiration times, and metadata.
2. Track order state transitions: PENDING → FILLED / CANCELLED / REJECTED.
3. Expose expiry information so :class:`~Executor.MT5LiveExecutionHandler`
   can cancel orders on time.
4. Persist trade history to SQLite for analytics and crash recovery.
5. Provide a query interface for order and trade history.

Design principles
-----------------
* Called by :class:`~Executor.MT5LiveExecutionHandler` and
  :class:`~Engine.Live_Engine`; **never** by Strategy classes.
* In-memory cache for O(1) lookups during live trading.
* SQLite backend keeps history across restarts.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_LOG = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Lifecycle states an order passes through."""

    PENDING_SUBMITTED = "pending_submitted"  # Sent to broker, awaiting confirmation
    PENDING_ACTIVE = "pending_active"        # Confirmed pending order
    FILLED = "filled"                        # Order executed; position is open
    CANCELLED = "cancelled"                  # Cancelled (expired or manual)
    REJECTED = "rejected"                    # Rejected by broker
    CLOSED = "closed"                        # Position closed


@dataclass
class TicketRecord:
    """Complete record of an order / trade lifecycle.

    Required fields are set at submission time; optional fields are
    populated as the order progresses through its lifecycle.

    Attributes
    ----------
    ticket : int
        MT5 order ticket number.
    symbol : str
        Instrument ticker.
    side : str
        ``'buy'`` or ``'sell'``.
    qty : float
        Volume in lots.
    entry_price : float
        Requested entry / stop trigger price.
    sl : float
        Stop-loss price.
    tp : float
        Take-profit price.
    submission_time : datetime
        UTC time the order was submitted.
    expiration_time : datetime or None
        UTC time after which the pending order should be cancelled;
        ``None`` means GTC.
    status : str
        Current :class:`OrderStatus` value (stored as string for serialisation
        compatibility).
    strategy_name : str
        Free-text identifier of the generating strategy.
    fill_price : float or None
        Actual fill price (set when FILLED).
    fill_time : datetime or None
        UTC fill timestamp.
    commission : float or None
        Broker commission charged on fill.
    close_price : float or None
        Exit price (set when CLOSED).
    close_time : datetime or None
        UTC close timestamp.
    pnl : float or None
        Realised profit / loss in account currency.
    swap : float or None
        Overnight financing charge.
    cancel_reason : str or None
        Human-readable reason for cancellation (e.g. ``'expired'``,
        ``'manual'``, ``'broker_cancelled'``).
    """
    ticket: int
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    entry_price: float
    sl: float
    tp: float
    submission_time: datetime
    expiration_time: Optional[datetime]
    status: str  # OrderStatus enum value
    strategy_name: str
    
    # Fill details (populated when filled)
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: Optional[float] = None
    
    # Close details (populated when closed)
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl: Optional[float] = None
    swap: Optional[float] = None
    
    # Cancellation reason
    cancel_reason: Optional[str] = None


class TicketBook:
    """Centralized order and trade journal with dual storage.

    All state-change methods (``record_*``, ``update_status``) are called
    exclusively by :class:`~Executor.MT5LiveExecutionHandler`.  Strategy
    classes query read-only helpers such as :meth:`has_pending_order` and
    :meth:`has_open_position` to gate signal generation without touching MT5.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database file.  Defaults to
        ``"ticketbook.db"`` in the current working directory.
    use_memory_only : bool, optional
        When ``True`` all SQLite persistence is skipped.  Useful for
        unit testing.
    """
    
    def __init__(self, db_path: str = "ticketbook.db", use_memory_only: bool = False) -> None:
        """
        Initialize TicketBook
        
        Args:
            db_path: Path to SQLite database file
            use_memory_only: If True, skip database persistence (for testing)
        """
        self.use_memory_only = use_memory_only
        self.db_path = db_path
        
        # In-memory cache: ticket -> TicketRecord
        self._tickets: Dict[int, TicketRecord] = {}
        
        # Fast lookup indices
        self._active_pending: Dict[int, TicketRecord] = {}  # ticket -> record
        self._symbol_tickets: Dict[str, List[int]] = {}     # symbol -> [tickets]
        
        if not use_memory_only:
            self._init_database()
    
    def _init_database(self) -> None:
        """Create the SQLite schema (orders table + indexes) if not already present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                sl REAL,
                tp REAL,
                submission_time TEXT NOT NULL,
                expiration_time TEXT,
                status TEXT NOT NULL,
                strategy_name TEXT,
                fill_price REAL,
                fill_time TEXT,
                commission REAL,
                close_price REAL,
                close_time TEXT,
                pnl REAL,
                swap REAL,
                cancel_reason TEXT
            )
        """)
        
        # Index for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON orders(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON orders(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submission_time ON orders(submission_time)")
        
        conn.commit()
        conn.close()
    
    def record_order(
        self,
        ticket: int,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        sl: float,
        tp: float,
        submission_time: datetime,
        expiration_time: Optional[datetime],
        strategy_name: str,
        status: OrderStatus = OrderStatus.PENDING_ACTIVE
    ) -> TicketRecord:
        """Record a new order in the ticket book.

        Parameters
        ----------
        ticket : int
            MT5 order ticket number.
        symbol : str
            Instrument ticker.
        side : str
            ``'buy'`` or ``'sell'``.
        qty : float
            Volume in lots.
        entry_price : float
            Requested entry / stop trigger price.
        sl : float
            Stop-loss price.
        tp : float
            Take-profit price.
        submission_time : datetime
            UTC time the order was submitted.
        expiration_time : datetime or None
            UTC expiry; ``None`` means GTC.
        strategy_name : str
            Free-text strategy identifier.
        status : OrderStatus, optional
            Initial status.  Defaults to
            :attr:`~OrderStatus.PENDING_ACTIVE`.

        Returns
        -------
        TicketRecord
            The newly created record.
        """
        record = TicketRecord(
            ticket=ticket,
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            submission_time=submission_time,
            expiration_time=expiration_time,
            status=status.value,
            strategy_name=strategy_name
        )
        
        # Update in-memory cache
        self._tickets[ticket] = record
        
        # Update indices
        if status in (OrderStatus.PENDING_SUBMITTED, OrderStatus.PENDING_ACTIVE):
            self._active_pending[ticket] = record
        
        if symbol not in self._symbol_tickets:
            self._symbol_tickets[symbol] = []
        self._symbol_tickets[symbol].append(ticket)
        
        # Persist to database
        if not self.use_memory_only:
            self._save_to_db(record)
        
        return record
    
    def update_status(
        self,
        ticket: int,
        new_status: OrderStatus,
        **kwargs
    ) -> bool:
        """Update the status, and optionally other fields, of an existing record.

        Parameters
        ----------
        ticket : int
            Order ticket to update.
        new_status : OrderStatus
            The new lifecycle status.
        **kwargs
            Any :class:`TicketRecord` attribute names to update
            (e.g. ``fill_price``, ``fill_time``, ``cancel_reason``).

        Returns
        -------
        bool
            ``True`` if the record was found and updated; ``False`` if the
            ticket does not exist in the journal.
        """
        if ticket not in self._tickets:
            return False
        
        record = self._tickets[ticket]
        record.status = new_status.value
        
        # Update optional fields
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)
        
        # Update indices
        if new_status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            # Remove from active pending
            self._active_pending.pop(ticket, None)
        
        # Persist to database
        if not self.use_memory_only:
            self._update_in_db(record)
        
        return True
    
    def record_fill(
        self,
        ticket: int,
        fill_price: float,
        fill_time: datetime,
        commission: float = 0.0
    ) -> bool:
        """Record fill details for an executed order."""
        return self.update_status(
            ticket,
            OrderStatus.FILLED,
            fill_price=fill_price,
            fill_time=fill_time,
            commission=commission
        )

    def record_close(
        self,
        ticket: int,
        close_price: float,
        close_time: datetime,
        pnl: float,
        swap: float = 0.0
    ) -> bool:
        """Record close details for a position."""
        return self.update_status(
            ticket,
            OrderStatus.CLOSED,
            close_price=close_price,
            close_time=close_time,
            pnl=pnl,
            swap=swap
        )

    def record_cancellation(
        self,
        ticket: int,
        reason: str = "manual"
    ) -> bool:
        """Record that a pending order was cancelled."""
        return self.update_status(
            ticket,
            OrderStatus.CANCELLED,
            cancel_reason=reason
        )
    
    def get_expired_orders(self, current_time: datetime) -> List[int]:
        """Return ticket numbers of pending orders whose expiration has elapsed.

        Parameters
        ----------
        current_time : datetime
            Timestamp to compare against (typically the bar-close time).
            Both timezone-aware and naive datetimes are accepted; all
            comparisons are performed as naive UTC to avoid TypeError.

        Returns
        -------
        list[int]
            Tickets of active pending orders that have expired.
        """
        # Normalise to naive UTC to avoid tz-aware vs naive comparison errors
        if getattr(current_time, 'tzinfo', None) is not None:
            current_time = current_time.replace(tzinfo=None)

        expired = []
        for ticket, record in self._active_pending.items():
            if record.expiration_time is not None:
                exp = record.expiration_time
                if getattr(exp, 'tzinfo', None) is not None:
                    exp = exp.replace(tzinfo=None)
                if current_time >= exp:
                    expired.append(ticket)
        return expired
    
    def get_active_pending_orders(self, symbol: Optional[str] = None) -> List[TicketRecord]:
        """Return active pending orders, optionally filtered by symbol.

        Parameters
        ----------
        symbol : str, optional
            When provided, only orders for this ticker are returned.

        Returns
        -------
        list[TicketRecord]
        """
        if symbol is None:
            return list(self._active_pending.values())
        
        return [
            record for record in self._active_pending.values()
            if record.symbol == symbol
        ]
    
    def get_order(self, ticket: int) -> Optional[TicketRecord]:
        """Return the :class:`TicketRecord` for *ticket*, or ``None`` if not found."""
        return self._tickets.get(ticket)

    # -------------------- State query helpers (used by Strategy classes) --------------------

    def has_pending_order(self, symbol: str) -> bool:
        """Return ``True`` if there is at least one active pending order for *symbol*."""
        return any(r.symbol == symbol for r in self._active_pending.values())

    def has_open_position(self, symbol: str) -> bool:
        """Return ``True`` if there is at least one open (filled, not yet closed) position for *symbol*."""
        for ticket in self._symbol_tickets.get(symbol, []):
            record = self._tickets.get(ticket)
            if record and record.status == OrderStatus.FILLED.value:
                return True
        return False

    def get_open_positions(self, symbol: Optional[str] = None) -> List[TicketRecord]:
        """Return all records that represent open positions (status ``FILLED``).

        Parameters
        ----------
        symbol : str, optional
            When provided, only positions for this ticker are returned.

        Returns
        -------
        list[TicketRecord]
        """
        result = [
            record for record in self._tickets.values()
            if record.status == OrderStatus.FILLED.value
        ]
        if symbol is not None:
            result = [r for r in result if r.symbol == symbol]
        return result

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Query order history with optional filters.

        Parameters
        ----------
        symbol : str, optional
            Filter by instrument ticker.
        status : OrderStatus, optional
            Filter by order status.
        start_time : datetime, optional
            Include orders submitted at or after this time.
        end_time : datetime, optional
            Include orders submitted at or before this time.

        Returns
        -------
        pd.DataFrame
            One row per matching order.  Empty DataFrame when no matches.
        """
        if self.use_memory_only:
            # Query from memory
            records = list(self._tickets.values())
        else:
            # Query from database for complete history
            records = self._query_from_db(symbol, status, start_time, end_time)
        
        # Apply filters if using memory
        if self.use_memory_only:
            if symbol:
                records = [r for r in records if r.symbol == symbol]
            if status:
                records = [r for r in records if r.status == status.value]
            if start_time:
                records = [r for r in records if r.submission_time >= start_time]
            if end_time:
                records = [r for r in records if r.submission_time <= end_time]
        
        # Convert to DataFrame
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame([asdict(r) for r in records])
        return df
    
    def get_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Calculate summary trading statistics for closed positions.

        Parameters
        ----------
        symbol : str, optional
            Restrict statistics to a single instrument.

        Returns
        -------
        dict
            Keys: ``total_trades``, ``winning_trades``, ``losing_trades``,
            ``win_rate``, ``total_pnl``, ``avg_win``, ``avg_loss``,
            ``profit_factor``, ``gross_profit``, ``gross_loss``.
        """
        df = self.get_order_history(symbol=symbol, status=OrderStatus.CLOSED)
        
        if df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        total_pnl = df['pnl'].sum()
        total_trades = len(df)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        avg_win = wins['pnl'].mean() if not wins.empty else 0.0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0.0
        
        gross_profit = wins['pnl'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    # -------------------- Database helpers --------------------

    def _save_to_db(self, record: TicketRecord) -> None:
        """Insert or replace *record* in the SQLite orders table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO orders VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            record.ticket,
            record.symbol,
            record.side,
            record.qty,
            record.entry_price,
            record.sl,
            record.tp,
            record.submission_time.isoformat() if record.submission_time else None,
            record.expiration_time.isoformat() if record.expiration_time else None,
            record.status,
            record.strategy_name,
            record.fill_price,
            record.fill_time.isoformat() if record.fill_time else None,
            record.commission,
            record.close_price,
            record.close_time.isoformat() if record.close_time else None,
            record.pnl,
            record.swap,
            record.cancel_reason
        ))
        
        conn.commit()
        conn.close()
    
    def _update_in_db(self, record: TicketRecord) -> None:
        """Persist an updated *record* to the database (delegates to ``_save_to_db``)."""
        self._save_to_db(record)  # INSERT OR REPLACE handles updates
    
    def _query_from_db(
        self,
        symbol: Optional[str],
        status: Optional[OrderStatus],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[TicketRecord]:
        """Query :class:`TicketRecord` objects from the SQLite database with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM orders WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if start_time:
            query += " AND submission_time >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND submission_time <= ?"
            params.append(end_time.isoformat())
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert rows to TicketRecord objects
        records = []
        for row in rows:
            records.append(TicketRecord(
                ticket=row[0],
                symbol=row[1],
                side=row[2],
                qty=row[3],
                entry_price=row[4],
                sl=row[5],
                tp=row[6],
                submission_time=datetime.fromisoformat(row[7]) if row[7] else None,
                expiration_time=datetime.fromisoformat(row[8]) if row[8] else None,
                status=row[9],
                strategy_name=row[10],
                fill_price=row[11],
                fill_time=datetime.fromisoformat(row[12]) if row[12] else None,
                commission=row[13],
                close_price=row[14],
                close_time=datetime.fromisoformat(row[15]) if row[15] else None,
                pnl=row[16],
                swap=row[17],
                cancel_reason=row[18]
            ))
        
        return records
    
    def __repr__(self):
        return f"<TicketBook: {len(self._tickets)} total orders, {len(self._active_pending)} active pending>"
