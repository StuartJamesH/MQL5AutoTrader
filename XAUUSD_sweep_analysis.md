# XAUUSD Sweep Analysis — v1 (April 25, 2026)

**Source:** `Engine/Model Packs/XAUUSD_M1_520weeks_LSTM_Multiclass_256seq_20260425_v1_sweep_summary.json`  
**Reference prod runs:** `XAUUSD_*_20260410_prod_summary.json`, `Engine/train_multiclass_prod.log` lines 341–393

---

## Executive Summary

The v1 sweep used the correct 20% holdout split (195K val sequences) and the US500 v3 winner architecture. However, **recall for SELL is critically low and unstable** — collapsing to near-zero in multiple epochs despite a recall floor constraint. The model predicts so few trades (0.2% SELL rate) that validation metrics are still statistically noisy. The training was cut short at 14 of 30 epochs and has not converged. Commission of 0.0 will produce overconfident live PnL estimates. These issues must be addressed before production training.

---

## 1. Data & Label Diagnosis

### Dataset
- **Training data:** `data/XAUUSD_M1_520weeks.csv`
- **Data split:** `test_start_date=2026-04-01`, `holdout_size=0.20`
- **Rows used:** train=781,109 | val=195,342 | total=976,451
- **⚠️ Critical anomaly:** Older production runs (Apr 10–18) show `total_rows=3,512,302` for the same dataset. The v1 sweep used only ~28% of the available data. This likely occurs because `test_start_date=2026-04-01` combined with `holdout_size=0.20` is windowing the data to only the most recent portion rather than using the full historical dataset with the last 20% held out.
- **Impact:** Training on 781K rows instead of ~2.8M significantly under-utilises available history and increases overfitting risk for a 5.1M parameter model.

### Label distribution
- From April 17 log (same dataset): SELL=2.63%, FLAT=94.72%, BUY=2.65% (train); SELL=2.53%, FLAT=94.59%, BUY=2.88% (val)
- At v1 best epoch (8): model predicted only 407 SELL and 1,086 BUY from 195,086 sequences → effective SELL prediction rate = **0.21%**, BUY = **0.56%**
- True SELL rate ≈ 2.6% → model is predicting only ~8% of actual SELL labels as SELL (recall=0.056 confirms this)
- **tp_mult=3.0 with max_horizon=90** generates sparse, high-quality signals but very few of them. 3×ATR within 90 minutes on a volatile instrument like XAUUSD is a demanding criterion.

### Label stability
- BUY PnL dominates (135 vs 61 at best epoch). XAUUSD has been in a sustained uptrend during this data period, which biases BUY signal quality. SELL signals are harder to label consistently in a trending market.

---

## 2. Feature Assessment

- **Input dim:** 82 features (consistent with April 17 run)
- Architecture (LSTMAttentionSEClassifier) doesn't change feature input
- No feature changes made between April 10 prod and v1 sweep — no new feature issues identified
- **Lookahead risk:** Already assessed in code review; issue #10 (alignment bug) is still open

---

## 3. Architecture Assessment

### Model
- `LSTMAttentionSEClassifier`: hidden_dim=256, num_layers=3, bidirectional=True
- `attn_heads=8`, `se_context_window=32`, `use_learned_query=True`, `dropout=0.1`, `dropout_out=0.35`
- **param_count=5,098,563** (~5.1M parameters)
- Training on 781K sequences → 5.1M params / 781K seqs = **6.5 parameters per training sample** — borderline overfit territory especially given the sparse signal classes

### LSTM receptive field
- A 3-layer bidirectional LSTM with seq_len=256 processes 256 bars = ~4.3 hours of XAUUSD 1-minute data per sequence
- For a commodity that trends strongly over intraday sessions, 256 bars is adequate but the bidirectional nature is appropriate for training (non-causal is fine offline)

### SE context window
- `se_context_window=32` on `seq_len=256` — SE aggregates the final 32 of 256 steps for channel recalibration. This is correctly sized at 12.5% of sequence length (US500 v3 winner used same ratio)

### Assessment
Architecture is healthy and already matches the US500 v3 winner config. No architecture changes recommended.

---

## 4. Loss & Training Config Assessment

### Alpha / class weights
```
class_weights = [0.545, 1.909, 0.545]  # [SELL, FLAT, BUY]
```
- FLAT is up-weighted by ~3.5× relative to SELL/BUY. In a 95%/2.5%/2.5% split, inverse-frequency weights would be ~0.033/1.0/0.033 (normalised). The current weighting under-penalises FLAT predictions, contributing to the model defaulting to FLAT.
- **Recommendation:** Use proper inverse-class-frequency weights. With SELL=2.6%, FLAT=94.7%, BUY=2.65%, approximate inverse-freq: SELL≈36×, FLAT≈1×, BUY≈36×. After normalising to mean=1: SELL≈2.5, FLAT≈0.07, BUY≈2.5. Or clamp to [1.0, 0.5, 1.0] style to avoid extreme gradients. The current [0.545, 1.909, 0.545] effectively **boosts FLAT** which is the opposite of what's needed.

### Recall floor
```
recall_floor=0.25, rec_floor_weight=15.0
```
- Recall floor at 0.25 means a penalty activates only when recall drops below 25%. The best-epoch recall_sell=0.056 (5.6%) is far below 0.25, yet the penalty at weight 15 was **insufficient to prevent this collapse**.
- Epoch 2 recall_sell collapsed to **0.001** (only 16 SELL predictions from 195K sequences). The penalty at that point was: `15 × (0.25 - 0.001)² = 15 × 0.062 = 0.93` — added to a total val_loss of ~0.023. This is a **4× multiplier** on val_loss, yet the model still chose near-zero recall in the next epoch.
- **Root cause:** The recall floor hinge is applied to the *loss function* during training, but its effect is diluted across 780K training sequences. The effective per-sequence penalty is 0.93 / 780,853 ≈ negligible.
- **Recommendation:** Increase `rec_floor_weight` from 15 → **60**. This matches the US500 recommendation and creates a much stronger corrective signal.

### Precision penalty
```
pr_weight=15.0
```
- At best epoch: precision_sell=0.366, precision_buy=0.347 — these are acceptable (above the 0.35 target).
- Given SELL recall is the primary problem, reducing `pr_weight` slightly (15 → 10) would relax precision pressure and allow the model to make more SELL predictions even at slightly lower precision, improving recall.

### Direction penalty
```
direction_penalty=1.5
```
- With SELL recall=0.056, the model is assigning most SELL bar probability to FLAT (not to BUY — this is not a direction confusion problem, it's a FLAT collapse problem).
- Direction penalty is not the primary lever here. Keep at 1.5.

### Gamma
```
gamma=2.5
```
- With FLAT dominating at 95%+ of predictions, the focal loss should strongly down-weight FLAT easy examples. gamma=2.5 is appropriate; no change needed.

### Learning rate
```
base_lr=0.0001
```
- LR is already at the recommended US500 level. Val loss shows a mild downward trend (0.0243→0.0225 at ep8). LR appears correctly calibrated.
- However, the PnL at epoch 8 is 196, then collapses to 17 at epoch 13 despite val_loss not diverging much (0.0225→0.0236). This suggests the **recall instability** is the cause of PnL collapse, not LR.

### Epoch budget
- Only **14 of 30 epochs completed**. The training log shows no XAUUSD entry after April 18. Training was either interrupted or timed out.
- Train_loss at epoch 13: 9.41 vs epoch 0: 13.48 — only ~30% converged. Model needs the full 30 epochs.

### Commission
```
commission=0.0
```
- XAUUSD round-trip spread is approximately $0.30–0.50/oz (Pepperstone, IC Markets typical for XAU/USD). At 0.0 commission, the model never learns to avoid marginal trades.
- **Recommendation:** Set `commission=0.40` (USD per oz, equivalent to ~0.3–0.5 pip XAUUSD spread).

### Data window anomaly
- If the 976K row count is confirmed to be a data windowing issue (rather than expected), production training should be run with the full 3.5M row dataset using `holdout_size=0.20` but without an unintended truncation from `test_start_date`.
- Verify whether removing `test_start_date` or setting it correctly causes the full dataset to be used.

---

## 5. Epoch-by-Epoch Summary

| Epoch | val_loss | PnL | SELL | BUY | recall_sell | recall_buy | n_sell | n_buy |
|-------|----------|-----|------|-----|-------------|------------|--------|-------|
| 0 | 0.02431 | -318 | -318 | — | 0.112 | 0.065 | 1282 | 593 |
| 1 | 0.02347 | -53 | — | — | 0.232 | 0.052 | 2177 | 387 |
| 2 | 0.02353 | 50 | — | — | **0.001** | 0.049 | **16** | 375 |
| 3 | 0.02376 | 124 | — | — | 0.084 | 0.010 | 628 | 67 |
| 4 | 0.02453 | 30 | — | — | 0.022 | 0.009 | 171 | 51 |
| 5 | 0.02346 | 111 | — | — | 0.048 | 0.037 | 371 | 247 |
| 6 | 0.02274 | 169 | — | — | 0.066 | 0.051 | 490 | 364 |
| 7 | 0.02339 | 38 | — | — | 0.040 | 0.003 | 292 | 21 |
| **8** | **0.02249** | **196** | **61** | **135** | **0.056** | **0.134** | **407** | **1086** |
| 9 | 0.02254 | 104 | — | — | 0.033 | 0.078 | 255 | 607 |
| 10 | 0.02280 | 86 | — | — | 0.033 | 0.073 | 256 | 585 |
| 11 | 0.02376 | 86 | — | — | 0.024 | 0.037 | 183 | 301 |
| 12 | 0.02336 | 71 | — | — | 0.006 | 0.051 | 41 | 395 |
| 13 | 0.02365 | 17 | — | — | 0.032 | 0.003 | 262 | 18 |

**Observation:** Epoch 8 is the clear checkpoint winner. Val_loss and PnL align here (unusual — typically val_loss misses the PnL peak by a few epochs). After epoch 8, both recall dimensions collapse simultaneously, causing PnL to fall despite precision remaining decent.

---

## 6. Comparison to Previous Production Runs

| Run | val_bars | best_pnl | recall_sell | recall_buy | note |
|-----|----------|----------|-------------|------------|------|
| Apr 10 prod | 10,000 | 197 (ep4) | 0.455 | 0.534 | statistically unreliable (9.5K seqs) |
| Apr 14 prod | 10,000 | 70 (ep3) | ~0.23 | ~0.37 | interrupted at ep16 |
| Apr 18 prod | 10,000 | 43 (ep15) | ~0.17 | ~0.13 | best val_loss=6.845 |
| **v1 sweep** | **195,342** | **196 (ep8)** | **0.056** | **0.134** | **most statistically valid** |

The April 10 run's recall_sell=0.455 was statistical noise from 10K val bars (≈419 SELL bars). With 195K val bars, the true recall is 0.056. The v1 sweep is the only honest measurement of model quality.

---

## 7. Code Change Plan

### Change 1: Increase `rec_floor_weight` — 15 → 60
**Priority:** High  
**File(s):** `.train_XAUUSD_LSTM.py` (production launcher — local config, not tracked)  
**Why:** At `rec_floor_weight=15`, recall_sell collapsed to 0.001 (16 SELL predictions / 195K sequences) in epoch 2. The hinge penalty was insufficient to reverse this. By epoch 8, the best checkpoint, recall_sell is still only 0.056 — well below the 0.25 recall_floor target. Increasing to 60 (4× multiplier) will create a strong corrective gradient whenever recall drops below the floor.  
**What to change:** In the loss_params block, set `rec_floor_weight=60.0`  
**Expected impact:** Recall_sell should stabilise above 0.15; PnL should increase due to higher SELL trade volume with positive avg_pnl_per_sell=0.150

---

### Change 2: Reduce `tp_mult` — 3.0 → 2.5
**Priority:** High  
**File(s):** `.train_XAUUSD_LSTM.py` (label_params — local config)  
**Why:** With `tp_mult=3.0, sl_mult=2.0, max_horizon=90`, the model predicts trades on only 0.21% of val sequences (407 SELL / 195K). The true label rate is ~2.6%. Even at perfect recall, there would only be ~5,000 SELL signals in 195K sequences — already sparse. Reducing to 2.5 increases the labelled SELL/BUY frequency, giving the model more training signal and making recall metrics more statistically stable per epoch.  
**What to change:** In label_params, set `tp_mult=2.5`  
**Expected impact:** SELL label rate should approximately double (from ~2.6% to ~4–5%), increasing trade volume at each epoch checkpoint and reducing per-epoch recall oscillation

---

### Change 3: Set realistic commission — 0.0 → 0.40
**Priority:** High  
**File(s):** `.train_XAUUSD_LSTM.py` (training_config — local config)  
**Why:** XAUUSD round-trip spread is approximately $0.30–0.50/oz. At commission=0.0, the model's PnL metric accepts all marginal trades that barely cover spread. In production, these trades lose money. All 1,086 BUY trades at best epoch have avg_pnl_per_buy=0.124 — after a $0.40 commission, this drops to –$0.276 per trade meaning the BUY component would be *net negative* in production. Commission must be included in training.  
**What to change:** In training_config, set `commission=0.40`  
**Expected impact:** Val PnL will be lower and more honest. Model will learn to avoid low-quality trades that don't cover the spread, improving live performance. BUY trade count will decrease but avg_pnl_per_buy should increase significantly.

---

### Change 4: Recalibrate `class_weights` to correctly down-weight FLAT
**Priority:** High  
**File(s):** `.train_XAUUSD_LSTM.py` (class_weights in training config)  
**Why:** Current `class_weights=[0.545, 1.909, 0.545]` **up-weights FLAT** (majority class) by 3.5×, which is the opposite of inverse-frequency weighting. With FLAT at ~95% of labels, the loss function is dominated by FLAT examples even before focal re-weighting. The correct direction is to up-weight minority classes (SELL, BUY). This is a likely contributor to the model's preference for predicting FLAT.  
**What to change:** Set `class_weights=[2.0, 0.2, 2.0]` — this gives SELL/BUY 10× the weight of FLAT, appropriate for a ~95%/2.5%/2.5% distribution. The exact values can be tuned but the direction must be reversed from the current config.  
**Expected impact:** Model should increase SELL/BUY predictions and reduce the FLAT collapse tendency, improving recall from the current critically-low 0.056/0.134.

---

### Change 5: Reduce `pr_weight` — 15 → 8
**Priority:** Medium  
**File(s):** `.train_XAUUSD_LSTM.py` (loss_params)  
**Why:** Precision at best epoch is 0.366 (SELL) and 0.347 (BUY) — both acceptable. The primary problem is recall, not precision. A high `pr_weight=15` penalises the model for making SELL/BUY predictions that don't hit TP, discouraging it from predicting trades at all. Reducing precision pressure will allow the recall floor (Change 1) to pull recall up without being simultaneously suppressed by the precision penalty.  
**What to change:** In loss_params, set `pr_weight=8.0`  
**Expected impact:** Precision may drop slightly (0.35 → 0.30), but recall should increase meaningfully. With more trades predicted, avg_pnl_per_trade quality becomes the key metric.

---

### Change 6: Ensure full dataset is used (verify data window)
**Priority:** High  
**File(s):** `.train_XAUUSD_LSTM.py` (training_config)  
**Why:** The v1 sweep used only 976K total rows vs the expected ~3.5M in the full XAUUSD dataset. If `test_start_date="2026-04-01"` combined with `holdout_size=0.20` is unintentionally restricting the data window to only the most recent ~1.2M rows, the model is being trained on 3.6× less data than available. For a 5.1M parameter model, this increases overfitting risk.  
**What to change:** Either remove `test_start_date` entirely and rely solely on `holdout_size=0.20` for splitting, or verify that the split uses all data before 2026-04-01 for training (not just 80% of a recent window). Confirm the training logs show ~2.8M train sequences before proceeding.  
**Expected impact:** 3.6× more training data should substantially improve model generalisation and reduce the recall instability from epoch to epoch.

---

### Change 7: Ensure full 30 epochs run uninterrupted
**Priority:** Medium  
**File(s):** Runtime / scheduler — no code change required  
**Why:** Training ran only 14 of 30 epochs. Train_loss at epoch 13 was 9.41 vs 13.48 at epoch 0 — the model has only converged ~30% of its potential learning. The val_loss trend is still mildly downward at epoch 13 (0.02365), indicating additional learning capacity.  
**What to change:** Allow training to run the full 30 epochs without interruption. Monitor for overfitting (train/val loss divergence) but expect best PnL to appear between epoch 15–25 if recall instability is resolved by Changes 1 and 4.  
**Expected impact:** PnL should peak higher and hold more stable across epochs once recall is stabilised.

---

## Recommended Production Config (full diff from v1 sweep)

```python
# label_params
tp_mult = 2.5          # was 3.0 — increase signal frequency
sl_mult = 2.0          # unchanged

# training_config
commission = 0.40      # was 0.0 — realistic XAUUSD round-trip spread
# remove test_start_date or verify it uses full dataset

# class_weights
class_weights = [2.0, 0.2, 2.0]  # was [0.545, 1.909, 0.545] — reverse the direction

# loss_params
rec_floor_weight = 60.0   # was 15.0 — enforce recall floor strongly
pr_weight = 8.0           # was 15.0 — relax precision to allow more trades
# all other loss params unchanged (gamma=2.5, recall_floor=0.25, direction_penalty=1.5)
```

Architecture (hidden_dim=256, num_layers=3, bidirectional=True, se_context_window=32, attn_heads=8, dropout=0.1, dropout_out=0.35) is **unchanged** — it matches the US500 v3 winner config and should be kept as-is.

---

_To implement these changes, update the hidden production launcher `.train_XAUUSD_LSTM.py` with the parameters above, verify the data split produces ~2.8M train sequences, and allow training to run the full 30-epoch budget uninterrupted._
