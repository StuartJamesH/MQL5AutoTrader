"""
TicketBook: Centralized order and trade lifecycle management

Primary responsibilities:
1. Record all orders with their tickets, expiration times, and metadata
2. Track order state transitions: PENDING → FILLED/CANCELLED/REJECTED
3. Monitor and auto-cancel expired pending orders
4. Persist trade history to SQLite for analytics
5. Provide query interface for order/trade history

Design:
- Dual storage: in-memory (fast reads) + SQLite (persistence)
- Called by Engine, not by Strategy classes
- Executor notifies TicketBook of state changes
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from pathlib import Path


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING_SUBMITTED = "pending_submitted"  # Order sent, awaiting confirmation
    PENDING_ACTIVE = "pending_active"        # Confirmed pending order
    FILLED = "filled"                        # Order executed
    CANCELLED = "cancelled"                  # Order cancelled (expired or manual)
    REJECTED = "rejected"                    # Order rejected by broker
    CLOSED = "closed"                        # Position closed


@dataclass
class TicketRecord:
    """Complete record of an order/trade lifecycle"""
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
    """
    Centralized order and trade journal with dual storage:
    - In-memory cache for fast access during live trading
    - SQLite database for persistence and analytics
    """
    
    def __init__(self, db_path: str = "ticketbook.db", use_memory_only: bool = False):
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
    
    def _init_database(self):
        """Create SQLite database schema if not exists"""
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
        """
        Record a new order in the ticket book
        
        Returns:
            TicketRecord object
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
        """
        Update order status and optional fields
        
        Args:
            ticket: Order ticket number
            new_status: New OrderStatus
            **kwargs: Additional fields to update (fill_price, fill_time, etc.)
        
        Returns:
            True if update successful, False if ticket not found
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
        """Record order fill details"""
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
        """Record position close details"""
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
        """Record order cancellation"""
        return self.update_status(
            ticket,
            OrderStatus.CANCELLED,
            cancel_reason=reason
        )
    
    def get_expired_orders(self, current_time: datetime) -> List[int]:
        """
        Get list of pending order tickets that have expired
        
        Args:
            current_time: Current timestamp to compare against
        
        Returns:
            List of expired ticket numbers
        """
        expired = []
        
        for ticket, record in self._active_pending.items():
            if record.expiration_time and current_time >= record.expiration_time:
                expired.append(ticket)
        
        return expired
    
    def get_active_pending_orders(self, symbol: Optional[str] = None) -> List[TicketRecord]:
        """
        Get all active pending orders, optionally filtered by symbol
        
        Args:
            symbol: Optional symbol filter
        
        Returns:
            List of TicketRecord objects
        """
        if symbol is None:
            return list(self._active_pending.values())
        
        return [
            record for record in self._active_pending.values()
            if record.symbol == symbol
        ]
    
    def get_order(self, ticket: int) -> Optional[TicketRecord]:
        """Get order record by ticket number"""
        return self._tickets.get(ticket)
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Query order history with filters
        
        Args:
            symbol: Filter by symbol
            status: Filter by OrderStatus
            start_time: Filter by submission_time >= start_time
            end_time: Filter by submission_time <= end_time
        
        Returns:
            DataFrame of matching orders
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
        """
        Calculate trading statistics
        
        Returns:
            Dictionary with stats like total_trades, win_rate, total_pnl, etc.
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
    
    # -------------------- Database methods --------------------
    
    def _save_to_db(self, record: TicketRecord):
        """Insert new record into database"""
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
    
    def _update_in_db(self, record: TicketRecord):
        """Update existing record in database"""
        self._save_to_db(record)  # INSERT OR REPLACE handles updates
    
    def _query_from_db(
        self,
        symbol: Optional[str],
        status: Optional[OrderStatus],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[TicketRecord]:
        """Query records from database with filters"""
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
