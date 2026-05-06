# Engine — Live Trading Runtime

This directory contains the live trading engine that connects to a local MetaTrader 5
terminal, runs ML inference on each new bar, and manages the full order lifecycle from
signal generation through to position close.

---

## Table of Contents

1. [Architecture Philosophy](#1-architecture-philosophy)
2. [Component Overview](#2-component-overview)
3. [Per-Bar Execution Sequence](#3-per-bar-execution-sequence)
4. [Component Deep-Dives](#4-component-deep-dives)
   - [Launchers](#41-launchers-run_multiclasspy--hidden-run_symbolpy)
   - [Engine.py — Orchestrator](#42-enginepy--orchestrator)
   - [DataHandler.py — Market Data Feed](#43-datahandlerpy--market-data-feed)
   - [Strategy.py — Signal Generator](#44-strategypy--signal-generator)
   - [Executor.py — MT5 Interface](#45-executorpy--mt5-interface)
   - [TicketBook.py — Order Journal](#46-ticketbookpy--order-journal)
5. [Order Lifecycle](#5-order-lifecycle)
6. [Strategy Detail: TripleBarrierHiLowMulticlass](#6-strategy-detail-triplebarrierhilowmulticlass)
7. [Additional Strategies](#7-additional-strategies)
8. [The Learn Package](#8-the-learn-package)
9. [Model Packs](#9-model-packs)
10. [Configuration & Environment](#10-configuration--environment)
11. [Launching a Bot](#11-launching-a-bot)
12. [Adding a New Symbol / Instance](#12-adding-a-new-symbol--instance)
13. [Logging & Observability](#13-logging--observability)
14. [File Map](#14-file-map)

---

## 1. Architecture Philosophy

The engine is built around three hard rules:

1. **MetaTrader 5 access is centralised in `Executor.py`.** No other component
   calls the MT5 Python API. Strategies are pure signal generators; they read state
   from `TicketBook` and return `Order` objects.

2. **`TicketBook` is the single source of truth for order state.** The strategy
   gates all decisions (flat / pending / filled) through `TicketBook` queries,
   not through live MT5 lookups. This keeps inference deterministic and fast.

3. **The engine loop is one entry point.** `Live_Engine.run()` drives everything.
   Each bar follows a strict sequence: get bar → run strategy → submit orders →
   expire stale orders → detect fills → detect closes. Nothing happens out of order.

This separation means you can swap the data source (live vs. replay), swap the
strategy, or replace the executor with a paper-trading mock — all without touching
any other component.

---

## 2. Component Overview

```
Launcher  →  Live_Engine  →  MT5DataHandler  (fetches bars from MT5)
                          →  Strategy         (signal generation, order sizing)
                          →  MT5LiveExecutionHandler  (all MT5 API calls)
                                ↕
                          TicketBook  (in-memory cache + SQLite journal)
```

| Component | File | Role |
|---|---|---|
| **Launcher** | `run_multiclass.py` / `.run_*.py` | Wires components together, sets config, handles shutdown |
| **Live_Engine** | `Engine.py` | Per-bar orchestration loop |
| **MT5DataHandler** | `DataHandler.py` | Fetches/streams bars from MT5; live and replay modes |
| **TripleBarrierHiLowMulticlass** | `Strategy.py` | ML-driven signal generation and order sizing |
| **MT5LiveExecutionHandler** | `Executor.py` | Order submission, cancellation, fill/close detection |
| **TicketBook** | `TicketBook.py` | Dual-storage order journal (in-memory + SQLite) |

---

## 3. Per-Bar Execution Sequence

Every M1 bar triggers this exact sequence inside `Live_Engine.run()`:

```
1. MT5DataHandler.get_next_bar()
   └─ Yields one completed OHLCV bar (live mode: waits for bar close)

2. Strategy.on_bar(bar)
   a. Appends OHLCV values to internal ring buffers (deques)
   b. Queries TicketBook: has_pending_order() / has_open_position()
      (No MT5 calls — purely in-memory)
   c. If flat AND buffer ≥ min_bars_for_features (5,000):
      └─ Builds feature DataFrame → scales via saved RobustScaler
         → runs model inference → applies Donchian trend gate + threshold
         → sizes order (fixed-risk lot calculation)
         → returns list[Order]
   d. Returns empty list if warm-up incomplete, state not flat, or no signal

3. For each Order returned:
   └─ Executor.submit_stop_order(order)
      a. Sends TRADE_ACTION_PENDING to MT5
      b. Records new order in TicketBook (PENDING_ACTIVE)

4. Executor.process_pending_batch(bar_time)
   ├─ Pass 1 — Expiry: cancel any pending order past expiration_time via MT5
   └─ Pass 2 — Fill detection: for orders no longer in MT5 pending queue,
               search deal history → record_fill() or record_cancellation()

5. Executor.process_position_updates_batch(bar_time)
   └─ For every FILLED position: check mt5.positions_get(ticket=)
      If gone → search deal history for DEAL_ENTRY_OUT → record_close()
```

The bar_time passed to the batch methods is the bar's UTC close time, ensuring
consistent behaviour in both live and replay modes.

---

## 4. Component Deep-Dives

### 4.1 Launchers — `run_multiclass.py` / hidden `.run_*.py`

The launcher is the only place where configuration lives. It:

- Sets all constants (`SYMBOL`, `MAGIC`, `MODEL_PACK_PATH`, `RISK`, thresholds, etc.)
- Instantiates all five components and wires them together
- Registers `SIGINT`/`SIGTERM` handlers for graceful shutdown
- Calls `engine.run()` — the blocking main loop

`run_multiclass.py` is the tracked generic template. The actual production bots
(`Engine/.run_EURUSD.py`, `.run_US500.py`, `.run_XAUUSD.py`, `.run_US2000.py`) are
hidden files kept out of git — they are thin configuration wrappers around the same
wiring pattern.

**Configuration constants to set per instance:**

| Constant | Purpose |
|---|---|
| `SYMBOL` | MT5 instrument ticker (e.g. `"EURUSD"`) |
| `MAGIC` | EA magic number — **must be unique per running instance** |
| `DB_PATH` | SQLite journal path — **must be unique per symbol** |
| `MODEL_PACK_PATH` | Path to the `.pkl` model pack |
| `RISK` | Fixed-risk amount per trade in account currency |
| `TRADE_THRESHOLD` | Minimum softmax probability to act on a signal |
| `DONCHIAN_LENGTH` | Look-back for the Donchian trend gate |
| `PATIENCE` | Bars before an unfilled stop order expires |
| `MAXPOS` | Hard cap on position size in lots |

### 4.2 `Engine.py` — Orchestrator

`Live_Engine` is deliberately thin. It holds references to the three core
components and runs the bar loop. It delegates all logic to the components it
wraps. Its only non-trivial behaviour is routing orders to the right executor
method based on `strategy.order_type` (`'stop'` vs `'market'`).

`configure_logging()` is also defined here. It sets up three handlers simultaneously:
console (`stdout`), local file (`trading.log`), and an optional mirror to a Google
Drive folder specified by `CLOUD_LOG_DIR` in `.env`.

### 4.3 `DataHandler.py` — Market Data Feed

Provides two data handlers and the `Order` dataclass.

**`DataHandler`** — CSV/DataFrame backed. Used for backtesting. Exposes
`get_next_bar()` as a generator yielding `itertuples` namedtuples, making it a
drop-in for `MT5DataHandler`.

**`MT5DataHandler`** — MT5-backed. Two modes:

| Mode | Behaviour |
|---|---|
| `replay` | Fetches a historical date range on construction, replays bars in order. Useful for walk-forward validation with a live connection. |
| `live` | Polls MT5 every second with `copy_rates_from_pos(…, 7_000)`. Yields each bar exactly once at bar close. Detects terminal timezone offsets automatically on first poll. |

Live mode fetches 7,000 bars per poll cycle. This keeps the strategy's ring buffer
continuously fed with enough history for multi-timeframe indicators to remain stable
without a separate warm-up fetch.

**`Order` dataclass** — the universal carrier object for trade intent, passed
from Strategy to Executor:

```python
@dataclass
class Order:
    symbol: str
    side: str           # 'buy' | 'sell'
    entry: float        # stop trigger price (0.0 for market orders)
    qty: int            # lots
    entry_time: str
    expiration: Optional[datetime]  # UTC; None = GTC
    sl: float
    tp: float
```

### 4.4 `Strategy.py` — Signal Generator

Contains the active ML strategy class `TripleBarrierHiLowMulticlass`. See
[Section 6](#6-strategy-detail-triplebarrierhilowmulticlass) for a full deep-dive.

Key design constraint: **strategies must never call MT5 directly.** All state
queries go through `TicketBook`. All order actions are expressed as `Order` objects
returned from `on_bar()`.

### 4.5 `Executor.py` — MT5 Interface

`MT5LiveExecutionHandler` is the only component that calls the MT5 Python API.
Its responsibilities are split into three groups:

**Order submission:**
- `submit_stop_order(order)` — sends `TRADE_ACTION_PENDING` to MT5, records
  the returned ticket in `TicketBook` as `PENDING_ACTIVE`
- `execute_market_order(order)` — sends `TRADE_ACTION_DEAL` with `ORDER_FILLING_IOC`;
  immediately records as `FILLED`
- `delete_order(ticket)` — sends `TRADE_ACTION_REMOVE` to cancel a pending order

**Lifecycle batch processing (called once per bar):**
- `process_pending_batch(bar_time)` — two-pass expiry + fill detection (see
  [Section 3](#3-per-bar-execution-sequence))
- `process_position_updates_batch(bar_time)` — detects closed positions by
  checking `mt5.positions_get(ticket=)` then searching deal history for
  `DEAL_ENTRY_OUT` records

**Fill detection fallback chain:**
1. Search deal history (last 24 hours) by `order` or `ticket` field
2. Fallback: check current open positions by symbol
3. If neither finds a match: classify as `broker_cancelled`

All state changes flow back to `TicketBook` via `record_fill()`,
`record_cancellation()`, or `record_close()`.

### 4.6 `TicketBook.py` — Order Journal

`TicketBook` provides dual storage: an in-memory dict for O(1) live lookups, and
a SQLite database (`ticketbook_<symbol>.db`) for persistence across restarts.

**In-memory indexes:**
- `_tickets: dict[ticket → TicketRecord]` — full order history
- `_active_pending: dict[ticket → TicketRecord]` — only currently pending orders
- `_symbol_tickets: dict[symbol → list[ticket]]` — fast by-symbol lookup

**Key query methods (used by Strategy):**

| Method | Returns |
|---|---|
| `has_pending_order(symbol)` | `True` if any active pending order exists |
| `has_open_position(symbol)` | `True` if any FILLED (not yet CLOSED) record exists |

**Key write methods (used only by Executor):**

| Method | Transition |
|---|---|
| `record_order(...)` | → `PENDING_ACTIVE` |
| `record_fill(ticket, ...)` | → `FILLED` |
| `record_cancellation(ticket, reason)` | → `CANCELLED` |
| `record_close(ticket, ...)` | → `CLOSED` |

**`TicketRecord` fields** (all persisted to SQLite):

| Field | When populated |
|---|---|
| `ticket`, `symbol`, `side`, `qty` | On submission |
| `entry_price`, `sl`, `tp`, `expiration_time` | On submission |
| `fill_price`, `fill_time`, `commission` | On fill |
| `close_price`, `close_time`, `pnl`, `swap` | On close |
| `cancel_reason` | On cancellation |

The SQLite schema has indexes on `symbol`, `status`, and `submission_time`
for efficient post-trade analytics queries. `get_order_history()` and
`get_statistics()` provide DataFrame and dict outputs suitable for reporting.

---

## 5. Order Lifecycle

```
submit_stop_order()
        │
        ▼
  PENDING_ACTIVE ──── expiration elapsed ────► CANCELLED (reason='expired')
        │
        │ broker removes from queue
        ▼
  fill found in deal history? ─── Yes ──► FILLED ──── position gone ──► CLOSED
                                └── No ──► CANCELLED (reason='broker_cancelled')
```

`PENDING_SUBMITTED` and `REJECTED` are defined in the `OrderStatus` enum for
future use (multi-step confirmation or explicit broker rejection handling) but
are not currently set by the production code path.

---

## 6. Strategy Detail: TripleBarrierHiLowMulticlass

This is the primary production strategy. It wraps a trained three-class PyTorch
model and implements all the runtime logic required to gate, size, and submit
orders safely.

### Signal generation gates (all must pass)

| Gate | Check | Purpose |
|---|---|---|
| **Warm-up** | `len(buffer) >= 5,000 bars` | Ensures MTF indicators (5 min, 15 min, 30 min) have stabilised |
| **Flat** | `not has_pending_order() and not has_open_position()` | One trade at a time per symbol |
| **Restricted hours** | Skip if local time is 06:00–10:00 | Avoids low-liquidity/high-spread session open |
| **Donchian trend** | BUY only if trend=+1, SELL only if trend=-1 | Filters counter-trend signals |
| **Threshold** | `max_prob >= trade_threshold` | Filters low-confidence model outputs |

### Inference pipeline

```
Ring buffer (deque, maxlen=7,000)
        │ last seq_len=256 bars
        ▼
add_all_features()      ← same function used at training time
        │
RobustScaler.transform()  ← scaler loaded from model pack (never refit)
        │
model(input_tensor)     ← PyTorch forward pass, eval() mode, torch.no_grad()
        │
softmax → argmax + max_prob
        │
Donchian gate + threshold check
        │
Order sizing
```

### Order sizing

```
ATR(14) on current buffer
stop_distance = 2.5 × ATR(14)
entry  = High + 0.00001  (BUY)  or  Low - 0.00001  (SELL)
sl     = entry - stop_distance  (BUY)  or  entry + stop_distance  (SELL)
tp     = entry + stop_distance  (BUY)  or  entry - stop_distance  (SELL)
lots   = risk / (stop_distance × pip_value)   capped at maxpos
expiry = bar_time + patience minutes
```

The `2.5 × ATR` multiplier matches the `tp_mult=2.5` / `sl_mult=2.5` used at
training time. This invariant **must not be changed** without retraining the model.

### Logging

Each bar produces a row in `Engine/Learn/Trade Logs/<symbol>_<date>.csv` (when
`log=True`). The CSV records: bar time, raw OHLCV, model probabilities, predicted
class, gate results, order details, and current account equity snapshot.

---

## 7. Additional Strategies

| File | Class | Status | Description |
|---|---|---|---|
| `Strategy.py` | `TripleBarrierHiLowMulticlass` | **Active** | Single 3-class ML model; primary production strategy |
| `PriceActionStrategy.py` | `PriceActionTrader` | Experimental | Rule-based: pin bars, inside bars, engulfing, S/R levels. No ML inference. |
| `.StrategyBinary_DEPRECIATED.py` | — | Deprecated | Dual-model binary approach (separate BUY / SELL models). Superseded by multiclass. |
| `.StrategyMulticlass_DEPRECIATED.py` | — | Deprecated | Earlier multiclass variant. Do not use. |

---

## 8. The Learn Package

`Engine/Learn/` is a mirror of `ModelWorkbench/Learn/`. The live engine uses this
copy at inference time. Both trees must stay in sync — any change to features,
preprocessing, or model architecture in one tree must be applied to the other.

| Module | Role |
|---|---|
| `features.py` | `add_all_features()` — builds the same ~100-feature DataFrame used at training time |
| `preprocess.py` | `RobustScaler` routing — scales features using the scaler saved in the model pack |
| `Models.py` | LSTM, TCN, Transformer class definitions — must match the architecture used at training |
| `Loss.py` | `TradeProfitabilityLoss` — not used at inference but kept in sync for completeness |
| `labels.py` | Not used at inference; retained for reference and parity |
| `Loaders.py` | Dataset loading utilities |
| `Util.py` | Shared utilities |

> **Important:** `features.py` is the highest-risk sync point. The feature column
> list is saved in the model pack (`pack['feature_cols']`). If any column is
> renamed, reordered, or removed in `ModelWorkbench/Learn/features.py`, the same
> change must be made in `Engine/Learn/features.py`, and the model must be retrained.

---

## 9. Model Packs

A model pack is a `.pkl` file produced by `ModelWorkbench/train_prod_model_cli.py`.
It contains everything the strategy needs to run inference without any other files.

**Producing a model pack** — run from the repo root:

```powershell
.\.venv\Scripts\python.exe .\ModelWorkbench\train_prod_model_cli.py `
    --symbol EURUSD `
    --label-profile EURUSD_1m_dev `
    --model-arch LSTM `
    --model-profile EURUSD_1m_r11 `
    --loss-profile EURUSD_1m_r11 `
    --patience 12 `
    --epochs 30
```

Each completed run writes up to four files to `Engine/Model Packs/`:

| File | Contents |
|---|---|
| `*_model.pkl` | Full model pack at the best val-loss epoch |
| `*_best_pnl_model.pkl` | Pack at the epoch with highest val-set PnL (written only if different from above) |
| `*_summary.json` | Training config, metrics, and per-epoch curves (human-readable, no unpickling needed) |
| `*_plots.png` | Val loss, precision/recall, confusion matrix, and PnL curves |

**Pack contents:**

| Key | Contents |
|---|---|
| `model` | Trained PyTorch `state_dict` |
| `model_class` | Architecture class reference (for re-instantiation) |
| `model_params` | Constructor kwargs to recreate the architecture |
| `model_info` | Metadata: `model_type`, `seq_len`, symbol, best epoch, val metrics |
| `scaler` | Fitted `RobustScaler` instance (never refit at inference) |
| `features` | Ordered list of feature column names expected by the model |
| `feature_function` | Reference to the per-symbol feature engineering function |
| `preprocess_function` | Reference to `preprocess_ohlcv` |
| `preprocess_args` | Kwargs passed to preprocessing at inference |
| `label_params` | Label parameters used at training (for reference) |
| `loss_params` | Loss profile used at training (for reference) |
| `data_split` | Train/val row counts and timestamps |
| `val_metrics` | Per-epoch precision, recall, F1, PnL, and prediction count curves |

Model packs live in `Engine/Model Packs/`. The active pack for each bot is set
by `MODEL_PACK_PATH` in the launcher.

---

## 10. Configuration & Environment

**`.env` file** (repo root — not committed):

| Variable | Purpose |
|---|---|
| `CLOUD_LOG_DIR` | Optional path to a Google Drive folder for log mirroring |

**Per-launcher constants** (set at the top of each launcher file):

See [Section 4.1](#41-launchers--run_multiclasspy--hidden-run_symbolpy) for the
full table.

---

## 11. Launching a Bot

```powershell
# From the repo root, using the venv
.\.venv\Scripts\python.exe .\Engine\.run_EURUSD.py
.\.venv\Scripts\python.exe .\Engine\.run_US500.py
.\.venv\Scripts\python.exe .\Engine\.run_XAUUSD.py
.\.venv\Scripts\python.exe .\Engine\.run_US2000.py

# Generic template (edit first)
.\.venv\Scripts\python.exe .\Engine\run_multiclass.py
```

**Prerequisites:**
- MetaTrader 5 terminal is running and a trading account is logged in
- The target symbol is visible in MT5 Market Watch
- The model pack `.pkl` exists at the path specified in `MODEL_PACK_PATH`
- `pip install MetaTrader5 torch pandas ta-lib` (or activate the existing venv)

**Graceful shutdown:** press `Ctrl+C` or send `SIGTERM`. The signal handler calls
`mt5.shutdown()` and exits cleanly.

---

## 12. Adding a New Symbol / Instance

1. Copy an existing hidden launcher (e.g. `Engine/.run_EURUSD.py`) to
   `Engine/.run_NEWSYMBOL.py`
2. Replace every field marked `# <-- REPLACE`
3. Set a **unique `MAGIC`** number — MT5 uses this to distinguish EAs
4. Set **`DB_PATH`** to a unique filename (e.g. `ticketbook_NEWSYMBOL.db`)
5. Place a trained model pack in `Engine/Model Packs/` and update `MODEL_PACK_PATH`
6. Run: `.\.venv\Scripts\python.exe .\Engine\.run_NEWSYMBOL.py`

**Two values that must never be shared between running instances:**
- `MAGIC` — shared magic numbers cause conflicting order operations
- `DB_PATH` — shared databases cause corrupted TicketBook state

Hidden per-symbol launchers are **not** tracked in git. They are treated as
local runtime configuration.

---

## 13. Logging & Observability

| Output | Location | Contents |
|---|---|---|
| Console | `stdout` | All `logging.INFO`+ events |
| Local log | `trading.log` (repo root) | Same as console, persisted |
| Cloud log | `CLOUD_LOG_DIR/trading.log` | Optional mirror (Google Drive) |
| Trade log CSV | `Engine/Learn/Trade Logs/<symbol>_<date>.csv` | Per-bar inference details, order events |
| TicketBook DB | `ticketbook_<symbol>.db` | Full order lifecycle history, queryable via SQLite |

The TicketBook SQLite database survives restarts. On relaunch, the in-memory
cache is populated from the database (via `_query_from_db`), so pending order
state is restored correctly after a crash or scheduled restart.

To query trade history programmatically:

```python
from TicketBook import TicketBook, OrderStatus

tb = TicketBook(db_path="ticketbook_EURUSD.db")
df = tb.get_order_history(symbol="EURUSD", status=OrderStatus.CLOSED)
stats = tb.get_statistics(symbol="EURUSD")
```

---

## 14. File Map

```
Engine/
├── run_multiclass.py               Tracked generic launcher template
├── .run_EURUSD.py                  Hidden production launcher (not in git)
├── .run_US500.py                   Hidden production launcher (not in git)
├── .run_XAUUSD.py                  Hidden production launcher (not in git)
├── .run_US2000.py                  Hidden production launcher (not in git)
│
├── Engine.py                       Live_Engine + configure_logging()
├── DataHandler.py                  Order dataclass, DataHandler, MT5DataHandler
├── Executor.py                     MT5LiveExecutionHandler
├── TicketBook.py                   TicketBook + TicketRecord + OrderStatus
├── Strategy.py                     TripleBarrierHiLowMulticlass (active)
├── PriceActionStrategy.py          PriceActionTrader (experimental)
├── ARCHITECTURE.md                 Mermaid component diagram + lifecycle state machine
│
├── Learn/
│   ├── features.py                 add_all_features() — inference feature builder
│   ├── preprocess.py               RobustScaler routing
│   ├── Models.py                   LSTM / TCN / Transformer definitions
│   ├── Loss.py                     TradeProfitabilityLoss (training use only)
│   ├── labels.py                   Label functions (reference only at inference)
│   ├── Loaders.py                  Dataset loading utilities
│   ├── Util.py                     Shared utilities
│   └── Trade Logs/                 Per-run CSV prediction and action logs
│
└── Model Packs/                    Serialised .pkl files (model + scaler + metadata)
```
