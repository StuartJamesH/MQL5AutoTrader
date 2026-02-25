# TicketBook System Documentation

## Overview

The **TicketBook** is a centralized order and trade lifecycle management system for your MT5 trading engine. It provides comprehensive logging, automatic expiration handling, and analytics capabilities.

## Architecture

```
┌──────────────┐
│   Engine     │ ◄── Orchestrates everything
└──────┬───────┘
       │
       ├──► DataHandler ◄── Provides bars
       │
       ├──► Strategy ◄── Generates signals (no MT5 dependencies)
       │
       ├──► TicketBook ◄── Central order/trade registry
       │         │
       │         ├──► In-memory cache (fast)
       │         └──► SQL persistence (durable)
       │
       └──► Executor ◄── MT5 interface
                 │
                 └──► Notifies TicketBook on fills/rejections
```

## Key Features

### 1. **Centralized Order Tracking**
- Every order is recorded with full metadata
- Track orders from submission → fill → close
- No lost tickets or orphaned orders

### 2. **Automatic Expiration Handling**
- Timestamp-based expiration (not manual countdown)
- Engine automatically checks and cancels expired orders
- Logged with cancellation reason

### 3. **Clean Strategy Classes**
- Strategies no longer need MT5 imports
- No manual order cancellation logic
- Fully testable without MT5 connection

### 4. **Dual Storage**
- **In-memory**: Fast lookups during live trading
- **SQLite**: Persistent storage for analytics and recovery

### 5. **Trading Analytics**
- Win rate, profit factor, average win/loss
- PnL tracking with commission and swap
- Export to CSV for external analysis

## Order Lifecycle States

```
PENDING_SUBMITTED → PENDING_ACTIVE → FILLED → CLOSED
                         ↓
                    CANCELLED (expired or manual)
                         ↓
                    REJECTED (broker rejection)
```

## Usage

### Basic Setup

```python
from TicketBookk import TicketBook, OrderStatus
from Executor import MT5LiveExecutionHandler
from Strategy import TripleBarrierHiLow
from Engine import Live_Engine

# 1. Initialize TicketBook
ticketbook = TicketBook(db_path="trading_journal.db")

# 2. Initialize Executor with TicketBook
executor = MT5LiveExecutionHandler(ticketbook=ticketbook)

# 3. Initialize Strategy (no mt5_executor parameter)
strategy = TripleBarrierHiLow(
    symbol='EURUSD',
    model=model,
    model_pack=model_pack,
    patience=5,  # Orders expire after 5 minutes
    strategy_name="MyStrategy"
)

# 4. Initialize Engine with TicketBook
engine = Live_Engine(
    data_handler=data_handler,
    strategy=strategy,
    executor=executor,
    ticketbook=ticketbook
)

# 5. Run
engine.run()
```

### Querying Orders

```python
# Get all active pending orders
pending = ticketbook.get_active_pending_orders(symbol='EURUSD')

# Get filled orders
filled = ticketbook.get_order_history(status=OrderStatus.FILLED)

# Get specific order
order = ticketbook.get_order(ticket=12345678)

# Get trading statistics
stats = ticketbook.get_statistics(symbol='EURUSD')
print(f"Win rate: {stats['win_rate']:.2%}")
print(f"Profit factor: {stats['profit_factor']:.2f}")
```

### Expiration Handling

Orders now use **datetime-based expiration** instead of countdown timers:

```python
# In Strategy.on_bar()
order = Order(
    symbol=self.symbol,
    side='buy',
    entry=1.0500,
    qty=0.1,
    entry_time=datetime.now(),
    expiration=datetime.now() + timedelta(minutes=5),  # ← Timestamp
    sl=1.0450,
    tp=1.0600,
    strategy_name=self.strategy_name
)
```

The Engine automatically checks for expired orders on each bar:
```python
# In Engine.run()
expired_tickets = ticketbook.get_expired_orders(current_time)
for ticket in expired_tickets:
    executor.delete_order(ticket)
    ticketbook.record_cancellation(ticket, reason="expired")
```

## Database Schema

### Orders Table

| Column | Type | Description |
|--------|------|-------------|
| ticket | INTEGER | MT5 order ticket (primary key) |
| symbol | TEXT | Trading symbol |
| side | TEXT | 'buy' or 'sell' |
| qty | REAL | Order quantity |
| entry_price | REAL | Entry price |
| sl | REAL | Stop loss |
| tp | REAL | Take profit |
| submission_time | TEXT | When order was submitted (ISO format) |
| expiration_time | TEXT | When order expires (ISO format) |
| status | TEXT | OrderStatus value |
| strategy_name | TEXT | Which strategy created this order |
| fill_price | REAL | Actual fill price (NULL if not filled) |
| fill_time | TEXT | When filled (ISO format) |
| commission | REAL | Commission paid |
| close_price | REAL | Close price (NULL if not closed) |
| close_time | TEXT | When closed (ISO format) |
| pnl | REAL | Profit/loss |
| swap | REAL | Swap/rollover fees |
| cancel_reason | TEXT | Why cancelled ('expired', 'manual', etc) |

## Migration Guide

### Before (Old Code)

```python
# Strategy.py - OLD WAY
class TripleBarrier:
    def __init__(self, mt5_executor=None, ...):
        self.mt5_executor = mt5_executor
        self.pending_order_ticket = None
        self.countdown = patience
    
    def on_bar(self, bar):
        # Manual MT5 polling
        if self.countdown <= 0:
            if self.mt5_executor:
                mt5.orders_get(symbol=self.symbol)
                # ... manual cancellation logic
```

### After (New Code)

```python
# Strategy.py - NEW WAY
class TripleBarrier:
    def __init__(self, strategy_name="TripleBarrier", ...):
        self.strategy_name = strategy_name
        # No mt5_executor, no countdown timer!
    
    def on_bar(self, bar):
        # Just create orders with expiration timestamps
        order = Order(
            symbol=self.symbol,
            expiration=self.t[-1] + timedelta(minutes=self.patience),
            strategy_name=self.strategy_name,
            ...
        )
        return [order]
```

## Benefits

### ✅ **Cleaner Code**
- Strategies are pure signal generators
- No MT5 imports in strategy classes
- Single responsibility principle

### ✅ **Better Testing**
- Strategies work in backtest and live mode
- No need for MT5 connection to unit test
- Mock TicketBook for testing

### ✅ **Improved Reliability**
- No missed expirations (timestamp-based)
- Persistent storage survives restarts
- Central source of truth for all orders

### ✅ **Enhanced Analytics**
- Complete trade journal
- Performance attribution by strategy
- Export data for external analysis

### ✅ **Easier Debugging**
- Full order lifecycle visibility
- Track why orders were cancelled
- Trace orders from signal to close

## API Reference

### TicketBook Class

#### `__init__(db_path: str, use_memory_only: bool = False)`
Initialize TicketBook with optional SQLite persistence.

#### `record_order(...) -> TicketRecord`
Record a new order with full metadata.

#### `update_status(ticket: int, new_status: OrderStatus, **kwargs) -> bool`
Update order status and optional fields.

#### `record_fill(ticket: int, fill_price: float, fill_time: datetime, commission: float = 0.0) -> bool`
Record order fill details.

#### `record_close(ticket: int, close_price: float, close_time: datetime, pnl: float, swap: float = 0.0) -> bool`
Record position close with PnL.

#### `record_cancellation(ticket: int, reason: str = "manual") -> bool`
Record order cancellation with reason.

#### `get_expired_orders(current_time: datetime) -> List[int]`
Get list of expired pending order tickets.

#### `get_active_pending_orders(symbol: Optional[str] = None) -> List[TicketRecord]`
Get all active pending orders, optionally filtered by symbol.

#### `get_order(ticket: int) -> Optional[TicketRecord]`
Get specific order by ticket number.

#### `get_order_history(...) -> pd.DataFrame`
Query order history with filters for symbol, status, date range.

#### `get_statistics(symbol: Optional[str] = None) -> Dict[str, Any]`
Calculate trading statistics (win rate, profit factor, etc).

## Troubleshooting

### Orders not expiring
- Check that Engine is passing `ticketbook` parameter
- Verify Engine calls `get_expired_orders()` on each bar
- Ensure Order.expiration uses datetime, not None

### Statistics showing zero trades
- Orders must reach `CLOSED` status to count in stats
- Check that position closes are being recorded
- Verify MT5 is actually filling and closing orders

### Database locked errors
- Only one process can write to SQLite at a time
- Use `use_memory_only=True` for testing
- Consider PostgreSQL for production multi-process setups

## Future Enhancements

Potential improvements to consider:

1. **Event callbacks**: Notify strategies when their orders fill
2. **Risk limits**: Max position size, daily loss limits
3. **Multi-strategy attribution**: Track which strategy is most profitable
4. **Position tracking**: Link related orders (entry + exit)
5. **Broker sync**: Reconcile with MT5 order history on startup
6. **Cloud storage**: S3/GCS backup of trade database

## Support

For issues or questions:
1. Check the example script: `example_with_ticketbook.py`
2. Review strategy implementations: `Strategy.py`
3. Inspect database: `sqlite3 trading_journal.db`

---

**Note**: The TicketBook is now the single source of truth for all order and trade data. Always use it instead of direct MT5 queries for order information.
