# MQL5AutoTrader

An end-to-end automated trading system for MetaTrader 5, combining offline ML model training with a live execution engine. The system is split into two main workspaces: **ModelWorkbench** for training and research, and **Engine** for live trading.

---

## How It Works

The core idea: on every 1-minute bar, does a stop-entry trade have a high probability of hitting its take-profit before its stop-loss? A neural network (LSTM or TCN) is trained to answer this question from observable price and indicator features, then deployed to trade live via MT5.

**Key design principles:**
- **Precision over recall** — a few high-confidence signals beats many marginal ones
- **Trend-aware entries** — pullbacks within confirmed trends only; range markets are skipped
- **No lookahead** — all features are computed causally from past data
- **Training/inference consistency** — label SL/TP distances, scaler, and feature list are all serialised into the model pack and reloaded at inference

---

## Repository Structure

```
MLQ5-Production/
├── ModelWorkbench/       # Offline training workspace
│   ├── Learn/            # Reusable modules: features, labels, models, loss, preprocessing
│   ├── data/             # OHLCV CSVs (not tracked in git — regenerate via fetch_datasets_bulk.py)
│   ├── params/           # Saved label parameter configs
│   └── *.ipynb / *.py    # Notebooks and training scripts
│
├── Engine/               # Live trading runtime
│   ├── Learn/            # Mirror of ModelWorkbench/Learn/ (must stay in sync)
│   ├── Model Packs/      # Trained .pkl model packs (output of training)
│   ├── Engine.py         # Per-bar orchestration loop
│   ├── DataHandler.py    # MT5 market data feed
│   ├── Strategy.py       # ML signal generation
│   ├── Executor.py       # All MT5 API calls
│   └── TicketBook.py     # Order journal (in-memory + SQLite)
│
├── MQL5/                 # Custom MetaTrader 5 indicators (MQL5 source files)
│   └── Indicators/       # Drop into MT5 terminal's MQL5/Indicators/ folder
│
└── launch_bots.ps1       # Launches all configured live bots
```

---

## ModelWorkbench

The offline training workspace. The full workflow is:

```
1. Fetch data         →  fetch_datasets_bulk.py  (or 1_0 Get Historical Data.ipynb)
2. Design labels      →  1_1 Signal Lab.ipynb
3. Select features    →  1_2 Feature Lab.ipynb
4. Verify parity      →  1_3 Feature Parity Check.ipynb
5. Train model        →  .train_<symbol>_<arch>.py  (production)
                         train_sweep_tcn/lstm/loss.py  (hyperparameter sweeps)
6. Evaluate offline   →  3_0 Review Model - Multiclass.ipynb
7. Review live trades →  4_0 Production Trade Report.ipynb
```

Training produces a `.pkl` model pack written directly to `Engine/Model Packs/` — there is no separate deployment step.

**Model architectures:**
- **LSTM** — Bidirectional LSTM (3 layers, 256 hidden) → Squeeze-Excite → Multi-head attention
- **TCN** — Dilated causal TCN (6 layers, 256 channels, kernel=3, receptive field ≈ 127 bars) → Squeeze-Excite → Multi-head attention

See [`ModelWorkbench/README.md`](ModelWorkbench/README.md) for full details.

---

## Engine

The live trading runtime. On every M1 bar close:

```
1. MT5DataHandler.get_next_bar()       — fetch completed bar from MT5
2. Strategy.on_bar(bar)                — run ML inference, return Order objects
3. Executor.submit_stop_order(order)   — place pending stop orders in MT5
4. Executor.process_pending_batch()    — expire stale orders, detect fills
5. Executor.process_position_updates_batch() — detect closed positions
```

**Key components:**

| Component | Role |
|---|---|
| `Engine.py` | Per-bar orchestration loop |
| `DataHandler.py` | Streams M1 bars from MT5 (live or replay mode) |
| `Strategy.py` | ML inference, Donchian trend gate, order sizing |
| `Executor.py` | The only component that calls the MT5 Python API |
| `TicketBook.py` | Dual-storage order journal (in-memory + SQLite) |

Strategies require **5,000 bars of warm-up** before inference begins. Live entries are **stop orders**, sized by fixed-risk lot calculation, with a configurable expiry window.

See [`Engine/README.md`](Engine/README.md) for full details.

---

## Quick Start

```powershell
# Launch all live bots
.\launch_bots.ps1

# Launch a specific bot
.\.venv\Scripts\python.exe .\Engine\.run_EURUSD.py

# Train a production model
.\.venv\Scripts\python.exe .\ModelWorkbench\.train_EURUSD_TCN.py

# Refresh OHLCV datasets
.\.venv\Scripts\python.exe .\ModelWorkbench\fetch_datasets_bulk.py

# Run a hyperparameter sweep (must cd first — dataset paths are relative)
Set-Location .\ModelWorkbench
..\.venv\Scripts\python.exe .\train_sweep_tcn.py
```

---

## Configuration

Live bots are configured via hidden per-symbol launcher scripts (`Engine/.run_<symbol>.py`). Each instance requires a unique `MAGIC` number and its own `ticketbook_<symbol>.db`. Environment variables (e.g., `CLOUD_LOG_DIR`) are loaded from `.env` in the repo root.

Runtime logs are written to `trading.log` and optionally mirrored to a Google Drive folder.
