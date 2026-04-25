# Trade Outcome Calculation

## Design

`calculate_trade_outcomes_all_candles` (in `ModelWorkbench/Learn/labels.py`) assigns a binary trade result to every candle in the dataset by simulating a stop-order entry:

| Result | Value | Condition |
|--------|-------|-----------|
| TP hit | `+1` | A future bar's High (BUY) or Low (SELL) reaches the TP level first |
| SL hit | `-1` | A future bar's Low (BUY) or High (SELL) reaches the SL level first |
| Unresolved | `NaN` | No TP or SL trigger before the end of the dataset |

- Entry levels are `entry_buy = High[t0]` and `entry_sell = Low[t0]`.
- TP and SL distances are `tp_mult × ATR` and `sl_mult × ATR`, where `ATR` is the 14-bar ATR computed by TA-Lib.
- Bars where `ATR == 0` or `ATR is NaN` (first ~14 bars of any series) are skipped entirely.
- There is **no time horizon cap** — the function scans forward to the last available bar.
- Tie-break: when TP and SL would both be triggered on the same future bar, **SL wins** (strict `<` comparison).
- `NaN` outcomes are `fillna(0.0)` downstream, contributing zero to both the gradient and P&L display.

---

## Bugs Found (historical)

Three bugs existed before this fix was applied:

### 1. TP returned `+2` instead of `+1`
Every caller post-processed outcomes with:
```python
for col in ['buy_outcome', 'sell_outcome']:
    outcomes[col] = outcomes[col].clip(upper=None).where(outcomes[col] <= 0, outcomes[col] * 2)
```
This doubled all positive values, making TP = +2 and SL = -1 — an asymmetric R-ratio that inflated the P&L simulation.

### 2. Timeouts contributed fractional P&L
The old function accepted a `max_horizon` parameter (default 1000 bars). When neither TP nor SL was triggered within that window, it computed a fractional value proportional to how close price was to TP or SL at the horizon bar. These fractions (e.g., `+0.4`, `-0.2`) appeared in the P&L sum instead of zero.

### 3. Commission was proportional to outcome magnitude
Because TP = +2 and SL = -1, the commission calculation `COMMISSION × |outcome|` charged 2× more per winning trade than per losing trade, misrepresenting the flat per-trade cost.

---

## Fixes Applied

| File | Change |
|------|--------|
| `ModelWorkbench/Learn/labels.py` | Removed `max_horizon` param, removed fractional timeout code, now returns only `1`, `-1`, or `NaN` |
| `ModelWorkbench/2_0_a Train LSTM - Multiclass.ipynb` (cell 10) | Removed `max_horizon=1000` override and the doubling block; updated print stats from `== 2` to `== 1` |
| `ModelWorkbench/2_0_b Train TCN - Multiclass.ipynb` (cell 10) | Same changes |
| `ModelWorkbench/train_sweep_lstm.py` | Same changes |
| `ModelWorkbench/train_sweep_tcn.py` | Same changes |
| `ModelWorkbench/train_sweep_loss.py` | Same changes |
| `ModelWorkbench/train_prod_model.py` | Removed `max_horizon=1000` from `outcome_params`; removed doubling block in `add_outcomes()` |

---

## Checklist for New Training Files

When adding a new training script or notebook that uses `calculate_trade_outcomes_all_candles`, verify:

1. **No `max_horizon` argument is passed.** The parameter no longer exists.
2. **No post-processing doubles TP values.** Remove any block matching:
   ```python
   outcomes[col] = outcomes[col].clip(upper=None).where(outcomes[col] <= 0, outcomes[col] * 2)
   ```
3. **Print stats check `== 1`, not `== 2`:**
   ```python
   n_tp = (outcomes['buy_outcome'] == 1).sum()
   n_sl = (outcomes['buy_outcome'] == -1).sum()
   ```
4. **Commission line is flat-rate per trade** — `COMMISSION * |outcome|` is correct now that `|outcome|` is always 1 for resolved trades.

---

## Unit Tests

`ModelWorkbench/tests/test_trade_outcomes.py` validates the contract with 8 cases:

1. BUY TP hit — pre-spike entries resolve to `+1`
2. BUY SL hit — all resolved entries are `-1`
3. No-horizon: TP beyond old 1000-bar limit still resolves to `+1`
4. SELL TP hit — pre-spike entries resolve to `+1`
5. SELL SL hit — all resolved entries are `-1`
6. Last-bar entries with no resolution are `NaN`
7. Same-bar TP+SL tie → SL wins (`-1`)
8. P&L sum: 3 TP + 2 SL on predicted bars = net `+1` before commission

Run with:
```powershell
Set-Location ModelWorkbench
..\.venv\Scripts\python.exe tests/test_trade_outcomes.py
```
