# Trading System Architecture

## Overview

This system is a live algorithmic trading engine that connects to a local MetaTrader 5 terminal. It is built around a strict separation of concerns: the strategy generates signals only, the executor owns all MT5 interaction, and the TicketBook is the single source of truth for order and position state.

---

## Component Map

```mermaid
flowchart TD
    subgraph Launcher["run_binary.py (Launcher)"]
        CFG["Config constants\nSYMBOL · TIMEFRAME · RISK\nPATIENCE · THRESHOLDS · MAGIC"]
    end

    subgraph Core["Core Engine Components"]
        ENG["Engine\nLive_Engine.run()"]
        DH["DataHandler\nMT5DataHandler"]
        STR["Strategy\nTripleBarrierHiLowBinary\n(pure signal generator)"]
        EXE["Executor\nMT5LiveExecutionHandler"]
        TB["TicketBook\n(in-memory + SQLite journal)"]
    end

    subgraph MT5["MetaTrader 5 Terminal"]
        MT5API["mt5 Python API"]
        MT5ORDERS["Pending Orders Queue"]
        MT5POSITIONS["Open Positions"]
        MT5DEALS["Deal History"]
        MT5FEED["Price Feed"]
    end

    subgraph Storage["Persistent Storage"]
        DB[("ticketbook_SYMBOL.db\nSQLite")]
        CSV["Trade Log CSV\nEngine/Learn/Trade Logs/"]
        TRADELOG["trading.log"]
    end

    %% Launcher wires everything together
    CFG -->|"creates"| TB
    CFG -->|"creates(ticket_book=)"| EXE
    CFG -->|"creates(ticket_book=)"| STR
    CFG -->|"creates"| DH
    CFG -->|"creates"| ENG

    %% Per-bar data flow
    DH -->|"bar (OHLCV)"| ENG
    ENG -->|"on_bar(bar)"| STR
    STR -->|"list[Order]"| ENG
    ENG -->|"submit_stop_order(order)\nor execute_market_order(order)"| EXE

    %% Lifecycle batch calls (once per bar, after order submission)
    ENG -->|"process_pending_batch(bar_time)"| EXE
    ENG -->|"process_position_updates_batch(bar_time)"| EXE

    %% Strategy reads state from TicketBook (no MT5 calls)
    TB -->|"has_pending_order(symbol)"| STR
    TB -->|"has_open_position(symbol)"| STR

    %% Executor writes state to TicketBook
    EXE -->|"record_order() · record_fill()\nrecord_cancellation() · record_close()"| TB

    %% Executor talks to MT5
    EXE -->|"order_send()\norders_get()\npositions_get()\nhistory_deals_get()"| MT5API
    DH -->|"copy_rates_range()\ncopy_rates_from()"| MT5API
    MT5API --- MT5ORDERS
    MT5API --- MT5POSITIONS
    MT5API --- MT5DEALS
    MT5API --- MT5FEED

    %% Persistence
    TB -->|"INSERT OR REPLACE"| DB
    STR -->|"_log_row()"| CSV
```

---

## Per-Bar Execution Sequence

Each bar triggers the following sequence inside `Live_Engine.run()`:

```
1. DataHandler.get_next_bar()        → yields one OHLCV bar
2. Strategy.on_bar(bar)
   a. Appends bar to internal price buffer
   b. Queries TicketBook for pending/position state (no MT5 call)
   c. Runs model inference if buffer is full and state is flat
   d. Returns list[Order] (empty if no signal, or state is blocked)
3. For each Order in list:
   └─ Executor.submit_stop_order(order)
      a. Sends TRADE_ACTION_PENDING to MT5
      b. Records the new order in TicketBook
4. Executor.process_pending_batch(bar_time)
   ├─ Pass 1 — Expiry: cancel any pending orders past their expiration_time
   └─ Pass 2 — Fill detection: for orders no longer in MT5 queue,
               search deal history → record_fill() or record_cancellation()
5. Executor.process_position_updates_batch(bar_time)
   └─ For every FILLED position: check mt5.positions_get(ticket=)
      If gone → search deal history for DEAL_ENTRY_OUT → record_close()
```

---

## Order Lifecycle (TicketBook States)

```mermaid
stateDiagram-v2
    [*] --> PENDING_ACTIVE : submit_stop_order() accepted by MT5
    PENDING_ACTIVE --> CANCELLED : expiration_time elapsed\n(process_pending_batch pass 1)
    PENDING_ACTIVE --> CANCELLED : broker rejected / removed\n(process_pending_batch pass 2)
    PENDING_ACTIVE --> FILLED : stop price triggered\n(process_pending_batch pass 2)
    FILLED --> CLOSED : SL / TP hit or manual close\n(process_position_updates_batch)
```

> `PENDING_SUBMITTED` and `REJECTED` are defined in the enum for future use  
> (e.g. multi-step order confirmation or explicit broker rejection handling).

---

## Component Responsibilities

| Component | Owns | Does NOT own |
|---|---|---|
| **run_binary.py** | Wiring, config, logging setup, graceful shutdown | Any trading logic |
| **MT5DataHandler** | Fetching bars from MT5 (live or replay) | Strategy state |
| **Live_Engine** | Per-bar orchestration loop | Order logic, MT5 calls |
| **Strategy** | Signal generation, order sizing, trade logging | MT5 calls, state mutation |
| **MT5LiveExecutionHandler** | All MT5 API calls, order submission, lifecycle batches | Signal generation |
| **TicketBook** | In-memory order state, SQLite persistence, query interface | MT5 calls, strategy logic |

---

## Creating a New Strategy Instance

1. Copy `run_binary.py` to e.g. `run_xauusd.py`
2. Replace all values marked `# <-- REPLACE`
3. Set a unique `MAGIC` number — MT5 uses this to distinguish different EAs
4. Set `DB_PATH` to a unique filename so journals don't collide
5. Run with `python Engine/run_xauusd.py`

Key things that **must** be unique per running instance:
- `MAGIC` — prevents conflicting order operations between instances
- `DB_PATH` — prevents TicketBook state from being shared across symbols

---

## File Map

```
Engine/
├── run_binary.py               Launcher / template — copy per strategy instance
├── Engine.py                   Live_Engine: per-bar orchestration loop
├── DataHandler.py              MT5DataHandler + Order dataclass
├── Executor.py                 MT5 order submission + lifecycle batch processing
├── TicketBook.py               Dual-storage order journal (in-memory + SQLite)
├── StrategyBinary.py           TripleBarrierHiLowBinary: dual-model signal generator
├── Strategy.py                 TripleBarrier / TripleBarrierHiLow / _XAUUSD variants
├── Learn/
│   ├── Models.py               LSTM / TCN / Transformer model definitions
│   ├── features.py             Feature engineering
│   ├── preprocess.py           Data preprocessing / scaling
│   ├── train.py                Model training loop
│   └── Trade Logs/             Per-run CSV prediction + action logs
└── Model Packs/                Serialised model weights + metadata (.pkl)
```
