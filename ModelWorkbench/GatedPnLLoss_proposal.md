# GatedVolumeFocalLoss — Design Proposal

**File:** `ModelWorkbench/GatedPnLLoss_proposal.md`
**Date:** 2025-07
**Status:** Proposal — not yet implemented
**Scope:** US500 1-minute TCN multiclass classifier (0=SELL, 1=FLAT, 2=BUY)

---

## Executive Summary

The current `TradeProfitabilityLoss` contains a **continuous precision penalty** (`pr_weight × (1 − soft_prec)`) that dominated 79–92% of validation loss across every US500 run (r17–r20). This structurally crowds out the focal CE signal and drives prediction volume from 2,126 raw preds at epoch 0 (gated_pnl = **81**) down to 243 by epoch 4 (gated_pnl = **38**) — a 9× volume collapse yielding 2.2× less gated profit despite 3.9× better per-trade quality. Empirical correlation across 73 epochs and 6 runs confirms the cause: `Corr(gated_volume, gated_pnl) = +0.924` vs `Corr(precision, gated_pnl) = −0.596`. The proposed `GatedVolumeFocalLoss` inverts the design: focal CE is the dominant training signal; precision and volume are guarded by **inactive hinges** (fire only below floor) rather than continuous penalties. The result is a feasibility region — a precision/volume space within which CE optimises freely — instead of a loss landscape that relentlessly sacrifices volume for precision.

---

## 1. Problem Statement and Diagnosis

### 1.1 The Structural Failure of `TradeProfitabilityLoss` for US500

The core failure is a **sign inversion**: the loss was designed to optimise per-trade quality (precision → ppt), but the objective is `gated_pnl = gated_trades × gated_ppt`, which is a product of volume and quality. At the current US500 operating point, the volume term has much larger marginal returns.

#### Quantitative proof from r20

The r20 run, epoch 0 vs epoch 4, provides the clearest natural experiment:

| Epoch | Raw S+B | Gate survival | Gated trades | gated_ppt | gated_pnl | prec_S | prec_B |
|-------|---------|---------------|-------------|-----------|-----------|--------|--------|
| ep0   | 2,126   | 37.9%         | **806**     | 0.100     | **81.0**  | 0.298  | 0.377  |
| ep4   | 243     | 39.9%         | 97          | 0.392     | 38.0      | 0.437  | 0.500  |
| ep5   | 100     | 42.0%         | 42          | 0.167     | 7.0       | 0.405  | 0.538  |

Re-deriving from `gated_pnl = raw_preds × gate_survival × ppt`:

```
ep0:  2,126 × 0.379 × 0.100 = 80.6   ≈ 81  ✓
ep4:    243 × 0.399 × 0.392 = 38.0       ✓
ep5:    100 × 0.420 × 0.167 =  7.0       ✓
```

The precision/volume tradeoff at these operating points:

```
Volume ratio (ep0 / ep4):       2126 / 243   =  8.75×  more volume at ep0
Quality ratio (ep4 / ep0 ppt): 0.392 / 0.100 =  3.92×  more quality at ep4

Net gated_pnl ratio:            81.0 / 38.0  =  2.13×  more PnL at ep0
```

**Conclusion:** 8.75× more volume × 2.5× lower ppt = **2.2× more gated PnL**. The current loss is optimising the wrong variable. Every precision improvement bought by `pr_weight` between ep0 and ep4 destroyed over twice as much gated PnL as it created.

The even more extreme case (ep5: gated_pnl = 7) shows what full volume collapse looks like despite prec_S = 0.405 and prec_B = 0.538 — the best precisions of the run.

### 1.2 Why the Continuous Precision Penalty Is Structurally Misaligned

The current precision penalty is:

```
L_prec = (w_sell × (1 − sp_sell) + w_buy × (1 − sp_buy)) / 2
```

where `w_sell`, `w_buy` ∈ [5, 10] and `sp_d = Σ(p_d × 1[y=d]) / (Σ p_d + ε)`.

**Key properties that make it structurally wrong for this problem:**

1. **Always active.** At healthy operating points (prec = 0.40), `L_prec = w × 0.60` per direction. This contributes 79–92% of total validation loss regardless of whether precision is in danger (0.20) or already adequate (0.45). The gradient toward FLAT-prediction never switches off.

2. **Reward-function inversion.** Lowering `pr_weight` increases volume (more S/B predictions accepted), but the penalty always fires even at `prec = 0.45` — well above the 0.38 floor. The model cannot "rest" at any precision level; it always faces gradient pressure to push FLAT probability higher.

3. **CE crowded out.** With `pr_component / val_loss = 79–92%`, the focal CE signal that actually teaches the model to distinguish S/B from FLAT contributes only 8–17% of the gradient budget. The model is not learning to classify; it is learning to produce few, confident S/B predictions.

4. **Gate-blind.** The breakout gate (the dominant gate) discards ~60% of signals at a fixed rate regardless of precision. A model that improves precision from 0.30 to 0.45 by halving volume passes only slightly better signals through the gate — but the gate discards the same fraction either way. The net effect on gated PnL is strongly negative.

### 1.3 The Optimal Operating Point

From r20 epoch-by-epoch data and cross-run empirical correlations:

- **Target volume:** 800–2,000 raw S+B preds per val epoch (corresponding to 300–800 gated trades at 38–42% gate survival)
- **Minimum precision:** 0.28–0.32 per direction (both achieved simultaneously at ep0 r20 with 2,126 preds)
- **Target gated_pnl:** 60–90 per epoch at the target operating point

The key insight: **ep0's operating point (prec_S=0.298, vol=2,126) is simultaneously above the precision floor AND at the volume target.** These floors are achievable together; there is no inherent precision–volume conflict at the target operating point. The conflict is entirely created by the continuous `pr_weight` penalty dragging the system away from this point.

---

## 2. Design Principles for `GatedVolumeFocalLoss`

### Principle 1 — Inversion

The primary training signal changes from "maximise precision" to "maintain minimum precision, maximise volume." Precision is no longer a target; it is a lower-bounded constraint.

### Principle 2 — Hinge Guards vs. Continuous Signals

All three guards (precision, volume, recall) are implemented as **quadratic hinges**: active only below their respective floors, zero above. This creates a **feasibility region** in the (precision, volume, recall) space. Within the region, the model trains freely on focal CE. Hinges activate only to prevent collapse.

The analogy to the current loss:
- **Current:** Continuous `(1 − prec)` penalty → always active, 80%+ of gradient
- **Proposed:** `max(0, floor − prec)²` hinge → dormant when healthy, fires only in danger zone

This is exactly how the recall hinge already works (and why `recall_floor=0.05` behaves well in r17–r20 — it fires in only 12–26% of epochs). The proposal extends the same philosophy to precision and volume.

### Principle 3 — CE as Primary Signal

With precision and volume guarded by dormant hinges, the focal cross-entropy term contributes ~80–90% of the gradient at healthy operating points (versus 8–17% under the current design). This is a 5–10× increase in the CE gradient budget, allowing the model to genuinely learn to discriminate SELL/BUY from FLAT.

### Principle 4 — Feasibility Region

The loss defines a three-dimensional feasibility region:

```
F = {model weights | sp_sell ≥ prec_floor  ∧
                     sp_buy  ≥ prec_floor  ∧
                     pred_rate ≥ vol_floor ∧
                     sr_sell ≥ recall_floor ∧
                     sr_buy  ≥ recall_floor}
```

Within `F`: `L_total = L_fce` (CE optimises freely)
On the boundary of `F`: hinge forces activate proportionally to the violation magnitude

The three constraints are:
1. **Precision floor** — prevents garbage signals (SELL on clearly-FLAT bars)
2. **Volume floor** — prevents the "refuse to trade" collapse seen at ep5 r20
3. **Recall floor** — prevents one direction from vanishing entirely (existing guard)

---

## 3. New Loss Class: `GatedVolumeFocalLoss`

### 3.1 Complete Formula

The total loss is the sum of four independent terms:

```
L_total = L_fce + L_vol + L_prec + L_rec
```

Each term defined below.

### 3.2 Term 1 — Focal Cross-Entropy (Primary Signal)

Identical to the existing implementation:

```
probs  = softmax(logits)                            # shape (B, 3)
pt     = probs[i, y_i]                              # true-class probability
ce_i   = CrossEntropy(logits_i, y_i, alpha)         # alpha = per-class weight
focal_i = (1 − pt_i)^γ × ce_i                      # focal down-weight
L_fce  = mean(focal_i)                              # scalar
```

**Alpha computation:** The trainer auto-computes alpha from training label frequencies using power-softened inverse weighting (see Section 3.8). For US500 (SELL=2.44%, FLAT=94.94%, BUY=2.62%), the effective alpha is approximately `[1.45, 0.16, 1.39]` (SELL, FLAT, BUY). See full derivation in Section 3.8.

**Expected magnitude:** With the above alpha and typical predictions, `L_fce ≈ 0.25–0.50` at healthy operating points. This is now the **dominant component**, contributing ~80% of total loss when all hinge guards are satisfied.

### 3.3 Term 2 — Volume Floor Hinge

**Motivation:** Prevents prediction collapse (ep5 r20: raw=100 preds → gated_pnl=7). Formulated to be dormant at the target operating volume and corrective only when volume collapses.

**Soft volume rate computation:**

For each sample `i` in the batch (size `B`), compute the total probability mass assigned to trade directions:

```
pred_rate = (1/B) × Σ_i [p_sell_i + p_buy_i]
          = mean_i(P(SELL|x_i) + P(BUY|x_i))
```

This is the mean fraction of probability mass on non-FLAT classes per bar. It differs from the hard prediction rate (argmax count / B) because it includes partial probability mass from bars predicted as FLAT but with some SELL/BUY probability.

**Relationship to hard prediction count (val set calibration):**

Using the r20 anchor points (val set ≈ 48,000 bars):

| Epoch | Hard preds (S+B) | Hard rate | Estimated soft pred_rate |
|-------|-----------------|-----------|--------------------------|
| ep0   | 2,126           | 4.34%     | ~0.070                   |
| ep4   | 243             | 0.50%     | ~0.037                   |
| ep5   | 100             | 0.20%     | ~0.031                   |

Fitting a linear model to these three points:
```
soft_pred_rate ≈ 0.029 + 0.942 × hard_rate
```

Solving for the lower target volume (400 hard preds, the minimum acceptable):
```
hard_rate = 400/48000 = 0.00833
soft_pred_rate = 0.029 + 0.942 × 0.00833 = 0.037
```

Target floor at 500 hard preds:
```
hard_rate = 500/48000 = 0.01042
soft_pred_rate = 0.029 + 0.942 × 0.01042 = 0.039
```

**Recommended `vol_floor = 0.040`** — corresponds to approximately 510 hard predictions on the ~48k val set, the lower bound of the target range (400–1,000).

**Volume hinge formula:**

```
L_vol = w_vol × max(0, vol_floor − pred_rate)²
```

**Calibration of `vol_floor_weight`:**

At ep5 r20 (soft pred_rate ≈ 0.031, violation = 0.009):
```
L_vol = w_vol × (0.040 − 0.031)² = w_vol × 8.1e−5
```

Targeting `L_vol ≈ 0.012` (≈4% of `L_fce ≈ 0.30`) when volume is at ep5 collapse level:
```
w_vol = 0.012 / 8.1e−5 ≈ 148  →  round to 120–150
```

At ep0 r20 (soft pred_rate ≈ 0.070):
```
violation = max(0, 0.040 − 0.070) = 0  →  L_vol = 0  (dormant ✓)
```

**Recommended `vol_floor_weight = 120`** — provides ~4% of CE when at ep5 collapse severity; dormant at healthy volume.

Gradient of the volume hinge with respect to `p_sell_i`:
```
∂L_vol/∂p_sell_i = −2 × w_vol × max(0, vol_floor − pred_rate) / B
                 = −2 × 120 × 0.009 / 512
                 = −0.00422  (per unit change in p_sell_i)
```
This gradient pushes all `p_sell` and `p_buy` probabilities upward, correcting the collapse.

### 3.4 Term 3 — Precision Floor Hinge

**Motivation:** Prevents garbage signals (random SELL/BUY predictions with precision < 0.28). Dormant at the target operating point (prec ≈ 0.30–0.45); active only in the danger zone (prec < 0.28).

**Soft precision computation (identical to current implementation):**

For each direction `d` ∈ {SELL=0, BUY=2}:

```
TP_soft_d = Σ_i [p_d_i × 1(y_i = d)]    # probability mass on true positives
FP_soft_d = Σ_i [p_d_i × 1(y_i ≠ d)]    # probability mass on false positives

soft_prec_d = TP_soft_d / (TP_soft_d + FP_soft_d + ε)
            = (Σ_i p_d_i × 1(y_i = d)) / (Σ_i p_d_i + ε)
```

This is the probability-weighted fraction of SELL/BUY predictions that land on true SELL/BUY bars. It is always in [0, 1] and correlates tightly with hard precision (argmax-based), but is differentiable throughout.

**Relationship to hard precision:**
- At ep0 r20: hard prec_S = 0.298, soft prec_S typically ≈ 0.30–0.32 (soft precision ≥ hard precision because it spreads the penalty over partial probability mass)
- The hinge on soft precision fires approximately when hard precision would fall below `prec_floor − 0.02`

**Precision hinge formula:**

```
L_prec = w_prec × [max(0, prec_floor − sp_sell)² + max(0, prec_floor − sp_buy)²]
```

**Recommended `prec_floor = 0.28`:**

Calibration against ep0 r20 (prec_S=0.298):
- `violation_S = max(0, 0.28 − 0.298) = 0` → hinge dormant ✓
- Ep0 is above the floor; no penalty applied to the best-ever gated_pnl epoch

At the danger zone (prec_S = 0.20, prec_B = 0.22):
```
L_prec = w_prec × [(0.28−0.20)² + (0.28−0.22)²]
       = w_prec × [0.0064 + 0.0036]
       = w_prec × 0.01
```

Targeting `L_prec ≈ 0.25` (≈80% of CE) in the danger zone to provide strong correction:
```
w_prec = 0.25 / 0.01 = 25
```

**Recommended `prec_floor_weight = 25`:**

Verification at near-floor precision (prec_S=0.26, prec_B=0.29):
```
L_prec = 25 × [(0.28−0.26)² + max(0, 0.28−0.29)²]
       = 25 × [0.0004 + 0]
       = 0.010 ≈ 3.3% of CE
```
Gentle correction, not domination. The hinge activates softly as precision approaches the floor, strongly only in the danger zone.

### 3.5 Term 4 — Recall Floor Hinge (Unchanged)

The existing recall hinge is kept **exactly as calibrated in r17–r20** (`recall_floor=0.05`, `rec_floor_weight=15.0`). No changes are needed.

**Why no changes:**
1. The recall hinge already behaves as a collapse guard: fired in 12–26% of epochs in r17–r20, well below the 30% alert threshold.
2. `recall_floor=0.05` correctly targets true probability collapse rather than the soft/hard recall mismatch (in FLAT-dominated outputs, soft recall ≈ 0.05–0.10 structurally even when hard recall is healthy at 0.30+).
3. The firing rate is not expected to change materially when CE becomes dominant — recall collapses are caused by the model over-predicting FLAT, which CE now actively opposes.

**Recall hinge formula (unchanged):**

```
sr_sell = (Σ_i p_sell_i × 1(y_i=SELL)) / (Σ_i 1(y_i=SELL) + ε)
sr_buy  = (Σ_i p_buy_i  × 1(y_i=BUY))  / (Σ_i 1(y_i=BUY)  + ε)

L_rec = w_rec × [max(0, recall_floor − sr_sell)² + max(0, recall_floor − sr_buy)²]
```

With `recall_floor = 0.05` and `w_rec = 15.0`.

### 3.6 Complete Loss Formula

```
L_total = L_fce + L_vol + L_prec + L_rec

where:

L_fce  = mean_i((1 − p_y_i)^γ × CE(logits_i, y_i, alpha))

L_vol  = w_vol  × max(0, r_vol  − pred_rate)²
         where pred_rate = mean_i(p_sell_i + p_buy_i)

L_prec = w_prec × [max(0, r_prec − sp_sell)² + max(0, r_prec − sp_buy)²]
         where sp_d = (Σ_i p_d_i × 1(y_i=d)) / (Σ_i p_d_i + ε)

L_rec  = w_rec  × [max(0, r_rec  − sr_sell)² + max(0, r_rec  − sr_buy)²]
         where sr_d = (Σ_i p_d_i × 1(y_i=d)) / (Σ_i 1(y_i=d) + ε)
```

**No direction confusion penalty** (set to 0.0, consistent with r17–r20 where `direction_penalty=0.0`). The precision hinge implicitly penalises direction confusion because BUY predictions on SELL bars inflate FP_soft and reduce sp_sell.

**No profit term** (set to 0.0, consistent with r20's finding that `profit_weight=5.0` with negative ungated PnL contributed positive-valued loss components that opposed S/B predictions).

### 3.7 Expected Loss Component Magnitudes at Healthy Operating Point

At the target steady state (prec ≈ 0.30–0.35, vol ≈ 600–1,000 raw preds):

| Component | Expected magnitude | % of total |
|-----------|--------------------|------------|
| `L_fce`   | 0.25–0.45          | **75–85%** |
| `L_vol`   | 0 (dormant)        | 0%         |
| `L_prec`  | 0 (dormant)        | 0%         |
| `L_rec`   | 0 (dormant)        | 0%         |

Compare to current TradeProfitabilityLoss at the same operating point:

| Component | Observed magnitude (r18–r20) | % of total |
|-----------|------------------------------|------------|
| `L_prec`  | 3.0–4.2                      | 79–92%     |
| `L_fce`   | 0.3–0.6                      | 8–17%      |
| `L_rec`   | 0–0.05                       | 0–3%       |

The proposed loss inverts this ratio: CE goes from 8–17% to 75–85% of total gradient.

### 3.8 Alpha (Class Weights) Recommendation

The training script (`train_prod_model_cli.py:869–873`) auto-computes alpha via power-softened inverse frequency:

```python
raw_weights = N_train / (n_classes × counts_per_class)   # inverse frequency
weights     = raw_weights ** 0.6                          # power softening
weights[1]  = min(weights[1], (weights[0] + weights[2]) / 2.0)  # cap FLAT
weights     = weights / weights.mean()                    # normalise to mean=1
```

For US500 (SELL=2.44%, FLAT=94.94%, BUY=2.62%):

```
raw_weights = [1/(3×0.0244), 1/(3×0.9494), 1/(3×0.0262)]
            = [13.66, 0.351, 12.72]

power-0.6   = [13.66^0.6, 0.351^0.6, 12.72^0.6]
            = [4.80, 0.534, 4.60]

FLAT cap    = min(0.534, (4.80+4.60)/2) = min(0.534, 4.70) = 0.534 (no cap)

mean        = (4.80 + 0.534 + 4.60) / 3 = 3.311

alpha       = [1.450, 0.161, 1.390]   (SELL, FLAT, BUY)
```

These alpha values correctly down-weight FLAT (94.94% of training data) and up-weight the rare S/B classes by approximately 9×. The `power=0.6` softening prevents the extreme 40× inverse-frequency weights from dominating the loss.

**Alpha is auto-computed and does not need to be specified in the JSON profile.** No change needed.

### 3.9 Parameter Table

| Parameter | Symbol | Recommended value | Tuning range | Controls |
|-----------|--------|-------------------|--------------|---------|
| `gamma` | γ | **2.0** | 1.5–2.5 | Focal exponent. 2.0 validated in r17–r20; higher values destabilise sparse labels. |
| `vol_floor` | r_vol | **0.040** | 0.025–0.060 | Lower bound on soft pred rate. 0.040 ≈ 510 hard preds on 48k val set. Increase if vol still collapses; decrease if random predictions dominate. |
| `vol_floor_weight` | w_vol | **120** | 80–250 | Volume hinge strength. 120 → L_vol ≈ 0.012 at ep5 collapse severity (4% of CE). Increase if collapse persists despite floor. |
| `prec_floor` | r_prec | **0.28** | 0.22–0.32 | Minimum acceptable soft precision. 0.28 is below ep0 r20's prec_S=0.298 (hinge dormant at ep0). Increase to 0.30 if many signals are random noise; decrease to 0.25 if floor opposes volume recovery. |
| `prec_floor_weight` | w_prec | **25** | 15–40 | Precision hinge strength. At prec=0.20 danger zone: L_prec=0.25 (80% of CE). Strong corrective signal without being continuous. |
| `recall_floor` | r_rec | **0.05** | 0.04–0.08 | Minimum soft recall per direction (collapse guard only). Unchanged from r17+. |
| `rec_floor_weight` | w_rec | **15.0** | 10–20 | Recall hinge strength. Unchanged from r17+; gentle correction when floor fires. |
| `eps` | ε | **1e-6** | — | Numerical stability. Unchanged. |

---

## 4. Expected Training Dynamics

### 4.1 Initialisation (epoch −1, random weights)

At random initialisation:
- `softmax(logits) ≈ [0.333, 0.333, 0.333]` for each sample
- `pred_rate = (0.333 + 0.333) = 0.667` → **well above `vol_floor=0.040`** → L_vol = 0
- `sp_sell = (0.333 × 0.0244) / 0.333 ≈ 0.0244` → **below `prec_floor=0.28`** → L_prec fires
- `sp_buy ≈ 0.0262` → **below floor** → L_prec fires
- `sr_sell ≈ 0.333 > 0.05` → L_rec = 0

At random weights, `L_prec` fires strongly (both directions ~0.252 below floor):
```
L_prec = 25 × [(0.28−0.0244)² + (0.28−0.0262)²]
       = 25 × [0.0654 + 0.0645]
       = 3.24
```
`L_fce ≈ 0.3–0.5` (CE with uniform predictions and alpha=[1.45, 0.16, 1.39]).

The precision hinge dominates at initialisation, pushing the model to concentrate probability mass on true S/B bars. This is appropriate and analogous to the early-epoch behaviour of the current loss.

### 4.2 Early Epochs (ep0–3)

As the model learns basic FLAT/S/B structure from focal CE:
- `L_fce` begins descending as easy FLAT bars are correctly classified
- `sp_sell` rises toward 0.25–0.30 as model focuses on higher-quality SELL signals
- `L_prec` transitions from strongly active → weakly active → dormant as precision crosses 0.28

During this transition:
- `pred_rate` falls from ~0.667 (random) toward ~0.05–0.10 as FLAT probability rises
- Volume hinge may fire briefly if pred_rate dips below 0.040 during the rapid FLAT-learning phase
- Expected: `L_vol` fires for ≤3 epochs then goes dormant

**Contrast with current loss:** Under `TradeProfitabilityLoss`, `pr_weight` fires continuously, creating a permanent gradient toward fewer S/B predictions. This is what drives the relentless volume decline observed across ep0–ep7 in r17–r20. Under `GatedVolumeFocalLoss`, once the model crosses `prec_floor=0.28`, the precision gradient switches off completely.

### 4.3 Expected Settled State (ep5–20)

At the target operating point:
- `prec_S ≈ 0.28–0.40`, `prec_B ≈ 0.30–0.42` → both above floor → all hinges dormant
- `pred_rate ≈ 0.040–0.080` → above vol_floor → volume hinge dormant
- `sr_sell, sr_buy ≈ 0.06–0.15` → above recall_floor=0.05 → recall hinge dormant
- `L_total ≈ L_fce ≈ 0.25–0.45` → CE is the dominant and near-exclusive signal
- `pr%loss` drops from the observed 79–92% to **<20%** (only during brief hinge-firing events)

Predicted val_loss range: **0.25–0.55** (vs 3.4–4.5 observed in r18–r20 with dominated pr_loss). The substantially lower val_loss reflects CE dominance, not improvement in precision — the early stopping criterion needs to be understood in this context.

Expected gated metrics at settled state:
- Raw S+B preds: **600–1,500** per epoch (8× improvement over r20's ep4–7 average of ~160)
- Gated trades: **230–600** (at 38–40% gate survival)
- gated_pnl: **40–90** (based on ep0 r20 as calibration anchor)

### 4.4 Comparison to Current Loss at Equivalent Epochs

| Metric | Current TradeProfitabilityLoss (r20 ep1–6 avg) | Proposed GatedVolumeFocalLoss (projected) |
|--------|----------------------------------------------|------------------------------------------|
| L_total | 3.5–4.0 | 0.30–0.50 |
| pr% of loss | 79–92% | 0% (dormant) |
| CE% of loss | 8–17% | 80–90% |
| Raw S+B preds | 100–400 | 500–1,500 |
| Gated trades | 40–155 | 200–600 |
| gated_pnl | 7–38 | 40–90 (projected) |

---

## 5. Implementation Notes

### 5.1 PyTorch Implementation Sketch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedVolumeFocalLoss(nn.Module):
    """Gated-PnL-optimised loss for US500 3-class trade entry classification.

    Replaces the continuous precision penalty of TradeProfitabilityLoss with
    hinge-based guards on precision, volume, and recall. Focal CE is the
    dominant training signal when all guards are satisfied.

    Loss = L_fce + L_vol + L_prec + L_rec

    - L_fce:  Focal cross-entropy (primary signal, ~80-90% of total at health)
    - L_vol:  Quadratic hinge: active only when soft pred rate < vol_floor
    - L_prec: Quadratic hinge: active only when soft precision < prec_floor
    - L_rec:  Quadratic hinge: active only when soft recall < recall_floor
              (identical to TradeProfitabilityLoss; no changes needed)
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        trade_classes=(0, 2),
        # Volume hinge
        vol_floor: float = 0.040,
        vol_floor_weight: float = 120.0,
        # Precision hinge
        prec_floor: float = 0.28,
        prec_floor_weight: float = 25.0,
        # Recall hinge (unchanged from TradeProfitabilityLoss)
        recall_floor: float = 0.05,
        rec_floor_weight: float = 15.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha             = alpha
        self.gamma             = float(gamma)
        self.sell_cls          = int(trade_classes[0])
        self.buy_cls           = int(trade_classes[1])
        self.vol_floor         = float(vol_floor)
        self.vol_floor_weight  = float(vol_floor_weight)
        self.prec_floor        = float(prec_floor)
        self.prec_floor_weight = float(prec_floor_weight)
        self.recall_floor      = float(recall_floor)
        self.rec_floor_weight  = float(rec_floor_weight)
        self.eps               = float(eps)

    def forward(self, logits, targets, trade_outcomes=None,
                return_components: bool = False):
        """
        Args:
            logits:          (B, 3) raw model outputs
            targets:         (B,)   integer class labels (0=SELL, 1=FLAT, 2=BUY)
            trade_outcomes:  ignored (retained for API compatibility)
            return_components: if True, return dict of component scalars
        """
        probs    = torch.softmax(logits, dim=1)        # (B, 3)
        p_sell   = probs[:, self.sell_cls]             # (B,)
        p_buy    = probs[:, self.buy_cls]              # (B,)

        true_sell = (targets == self.sell_cls).float()  # (B,)
        true_buy  = (targets == self.buy_cls).float()   # (B,)

        # ── 1. Focal cross-entropy ────────────────────────────────────────────
        ce        = F.cross_entropy(logits, targets, weight=self.alpha,
                                    reduction='none')
        pt        = probs[torch.arange(len(targets)), targets]
        focal     = ((1.0 - pt) ** self.gamma) * ce
        L_fce     = focal.mean()

        # ── 2. Volume floor hinge ─────────────────────────────────────────────
        # pred_rate = mean soft probability mass on SELL + BUY classes
        pred_rate  = (p_sell + p_buy).mean()           # scalar in [0, 1]
        vol_viol   = F.relu(self.vol_floor - pred_rate)
        L_vol      = self.vol_floor_weight * (vol_viol ** 2)

        # ── 3. Precision floor hinge ──────────────────────────────────────────
        # soft_prec_d = TP_soft_d / (TP_soft_d + FP_soft_d + eps)
        #             = weighted fraction of predicted d that are true d
        sp_sell    = (p_sell * true_sell).sum() / (p_sell.sum() + self.eps)
        sp_buy     = (p_buy  * true_buy ).sum() / (p_buy.sum()  + self.eps)
        prec_viol_s = F.relu(self.prec_floor - sp_sell)
        prec_viol_b = F.relu(self.prec_floor - sp_buy)
        L_prec     = self.prec_floor_weight * (prec_viol_s**2 + prec_viol_b**2)

        # ── 4. Recall floor hinge (unchanged from TradeProfitabilityLoss) ─────
        sr_sell    = (p_sell * true_sell).sum() / (true_sell.sum() + self.eps)
        sr_buy     = (p_buy  * true_buy ).sum() / (true_buy.sum()  + self.eps)
        rec_viol_s = F.relu(self.recall_floor - sr_sell)
        rec_viol_b = F.relu(self.recall_floor - sr_buy)
        L_rec      = self.rec_floor_weight * (rec_viol_s**2 + rec_viol_b**2)

        # ── Total ─────────────────────────────────────────────────────────────
        total = L_fce + L_vol + L_prec + L_rec

        if return_components:
            return {
                "total":           total,
                "focal_ce":        float(L_fce.item()),
                "precision_loss":  float(L_prec.item()),   # named for compatibility
                "recall_loss":     float(L_rec.item()),
                "confusion_loss":  0.0,                    # not used; kept for compat
                "profit_loss":     0.0,                    # not used; kept for compat
                # New fields (if the training loop is updated to log them):
                "vol_loss":        float(L_vol.item()),
            }
        return total
```

**Implementation notes:**
- The `return_components` dict uses the same key names as `TradeProfitabilityLoss` for compatibility with the training loop's loss-component logger (`loss[fce=... pr=... rec=... dir=... pft=...]`). The `vol_loss` key is new and requires a minor update to the epoch log line in `train_prod_model_cli.py` (see Section 5.2).
- `trade_outcomes` parameter is accepted but ignored, ensuring the same `criterion(logits, targets, outcomes)` call signature works without changes.
- No `label_smoothing` parameter — label smoothing conflicted with the precision penalty in r3 and is irrelevant when CE is not fighting a precision penalty.
- No `direction_penalty` — the precision hinge implicitly penalises direction confusion. Separate direction penalty was 0.0 in r17–r20 and contributed near-zero loss.

### 5.2 Integration with `train_prod_model_cli.py`

Two changes required in `train_prod_model_cli.py`:

**Change A — Add `GatedVolumeFocalLoss` to the criterion factory** (wherever `TradeProfitabilityLoss` is instantiated from the loss profile):

```python
# In the criterion construction block (approx. line 880-910):
from Learn.Loss import TradeProfitabilityLoss, GatedVolumeFocalLoss

loss_class_map = {
    "TradeProfitabilityLoss": TradeProfitabilityLoss,
    "GatedVolumeFocalLoss":   GatedVolumeFocalLoss,
}
loss_class_name = loss_params.pop("loss_class", "TradeProfitabilityLoss")
LossClass = loss_class_map[loss_class_name]
criterion = LossClass(alpha=class_weights, **loss_params)
```

This requires adding `"loss_class": "GatedVolumeFocalLoss"` to the new JSON profile. Existing profiles without this key default to `TradeProfitabilityLoss` (backward compatible).

**Change B — Update the epoch log line** to include `vol_loss` in the `loss[...]` block (optional but recommended for diagnostics):

```python
# Current log line (approx line 1037):
"loss[fce=%.3f pr=%.3f rec=%.3f dir=%.3f pft=%.3f]"
# becomes:
"loss[fce=%.3f vol=%.3f pr=%.3f rec=%.3f]"
```

If Change B is not implemented, set `confusion_loss=0.0` and `profit_loss=0.0` in `return_components` (already done in the pseudocode above), so the existing format logs them as `dir=0.000 pft=0.000` and `pr=L_prec` as before.

**Mirror:** Both changes apply to `Engine/Learn/Loss.py` (new class) and `Engine/train_multiclass_prod.py` is the Engine-side trainer — confirm whether it shares the same criterion factory or has a separate one.

### 5.3 JSON Profile Schema

The new loss profile in `ModelWorkbench/params/loss_params_multiclass.json`:

```json
"US500_gvfl_r1": {
  "_comment": "From US500_1m_r21 (pr_weight=4.0, vol still collapsing). New loss class GatedVolumeFocalLoss: replaces continuous pr_weight penalty with volume + precision hinge guards. Key motivation: r20 ep0 (gated_pnl=81, 806 gated trades, prec_S=0.298) was blocked and then destroyed by pr_weight driving vol from 2126 to 243 preds by ep4 (gated_pnl=38). At pr_weight=4.0 (r21), pr still 79-92% of val_loss. This design makes CE ~80-90% of loss when both guards satisfied. vol_floor=0.040 (≈510 hard preds on 48k val; vol hinge fires at ep5 r20 collapse state). prec_floor=0.28 (dormant at ep0 r20 prec_S=0.298). recall_floor=0.05 unchanged from r17+.",
  "loss_class":         "GatedVolumeFocalLoss",
  "gamma":              2.0,
  "trade_classes":      [0, 2],
  "vol_floor":          0.040,
  "vol_floor_weight":   120.0,
  "prec_floor":         0.28,
  "prec_floor_weight":  25.0,
  "recall_floor":       0.05,
  "rec_floor_weight":   15.0,
  "eps":                1e-6
}
```

**Parameter semantics vs. `TradeProfitabilityLoss`:**

| Old parameter | New parameter | Change |
|---------------|---------------|--------|
| `pr_weight` (continuous) | `prec_floor` + `prec_floor_weight` (hinge) | Fundamental: continuous → hinge guard |
| *(not present)* | `vol_floor` + `vol_floor_weight` | New: volume hinge guard |
| `recall_floor` | `recall_floor` | Unchanged: same value (0.05) |
| `rec_floor_weight` | `rec_floor_weight` | Unchanged: same value (15.0) |
| `direction_penalty` | *(removed)* | Was 0.0 in r17–r20; implicit via precision hinge |
| `profit_weight` | *(removed)* | Was 0.0 in r20; removed to simplify |
| `gamma` | `gamma` | Unchanged: 2.0 |

---

## 6. Migration and Validation Plan

### 6.1 First Validation Run

**Profile name:** `US500_gvfl_r1`

**CLI invocation:**
```bash
python ModelWorkbench/train_prod_model_cli.py \
  --symbol US500 \
  --label-profile US500_1m_dev_r2 \
  --model-profile US500_1m_r2 \
  --loss-profile US500_gvfl_r1 \
  --n-rows 3000000 \
  --epochs 30 \
  --patience 8 \
  --model-version gvfl_r1
```

Use `--model-profile US500_1m_r2` (TCN, wd=0.008) — the architecture validated in r18 (best run to date: gated_pnl=48). The loss change is the only variable; architecture is held fixed.

### 6.2 Success Criteria

**Primary success** (at best_gated epoch):
- `gated_pnl ≥ 60` at best_gated_epoch (vs r18 best of 48, r20 ep0 blocked 81)
- `gated_trades ≥ 200` at best_gated_epoch
- `prec_S ≥ 0.28 AND prec_B ≥ 0.28` (both above precision floor and `_GATED_MIN_PRECISION=0.30`)

**Secondary health checks:**
- Raw S+B preds: ≥ 500 through ep8 (no early volume collapse)
- `pr% of loss` ≤ 20% in any epoch (confirms CE is dominant)
- `vol_loss > 0` in ≤ 5 of 30 epochs (volume hinge is a guard, not a constant signal)
- `prec_loss > 0` in ≤ 5 of 30 epochs (precision hinge is a guard)
- `gnorm` stable or slowly rising (not the 29→149 escalation seen in r17)
- `val_loss` in range 0.25–0.60 (much lower than r18's 3.6–4.5, reflecting CE dominance)

**Checkpoint alert:** With `_GATED_MIN_PRECISION=0.30` (updated post-r20), the gated model should now be saved whenever prec_S ≥ 0.30 AND prec_B ≥ 0.30, allowing ep0-type epochs to be checkpointed.

### 6.3 Epoch-by-Epoch Monitoring

Read the log via:
```python
with open('Engine/train_multiclass_prod.log') as f:
    lines = f.readlines()
start_idx = max(i for i, l in enumerate(lines) if 'Training start' in l)
for l in lines[start_idx:]:
    if 'Epoch' in l:
        print(l.strip())
```

Monitor these specific flags at each epoch:

| Epoch | Check |
|-------|-------|
| ep0 | `preds[S+B] > 1000`? If < 500: vol hinge may need `vol_floor_weight` increase. |
| ep1–3 | `pr% of loss < 30%`? If > 50%: precision hinge still firing — check prec_floor is not too high. |
| ep5 | `preds[S+B] > 400`? If < 200: vol hinge miscalibrated; increase `vol_floor_weight` to 200. |
| ep8 | `gated_trades > 150`? If < 100: volume collapse starting; lower `vol_floor` to 0.030 in r2. |

### 6.4 Failure Modes and Fallback

**Failure Mode 1: Volume collapse despite vol hinge**
- Symptom: Raw S+B preds < 200 by ep5, `vol_loss > 0.050` every epoch
- Diagnosis: `vol_floor_weight` too low; hinge fires but provides insufficient gradient
- Fallback: `US500_gvfl_r2` with `vol_floor_weight = 200` (and optionally `vol_floor = 0.035`)

**Failure Mode 2: Precision below 0.28 persists despite prec hinge**
- Symptom: `prec_loss > 0.10` every epoch, prec_S < 0.25 at ep5+
- Diagnosis: `prec_floor_weight` too low for CE-dominant gradient regime
- Fallback: `US500_gvfl_r2` with `prec_floor_weight = 40`

**Failure Mode 3: Oscillation between high-vol/low-prec and low-vol/high-prec**
- Symptom: preds[S+B] swings 1000→100→800 epoch-to-epoch
- Diagnosis: `vol_floor` and `prec_floor` are in opposition — the model oscillates between satisfying one floor or the other
- Mitigation: This is unlikely given that ep0 r20 simultaneously had prec_S=0.298 AND raw_preds=2126 (both floors would be satisfied). If it occurs, lower `prec_floor` to 0.24 in r3.

**Failure Mode 4: val_loss doesn't decrease (CE not learning)**
- Symptom: `val_loss` flat at 0.45–0.55 with no trend across 15+ epochs
- Diagnosis: Focal CE alone is insufficient; model may need architecture change
- Fallback: Revert to `US500_1m_r21` (pr_weight=4.0) — the conservative `TradeProfitabilityLoss` extension

**Fallback chain:** `US500_gvfl_r1` → `US500_gvfl_r2` (weight adjustments) → `US500_gvfl_r3` (floor adjustments) → `US500_1m_r21` (safe fallback to TradeProfitabilityLoss)

---

## 7. Risks and Mitigations

### Risk 1 — Volume Floor Too High → Random Signal Flood

**Description:** If `vol_floor = 0.040` is set too high relative to what the model can achieve while maintaining reasonable precision (≥ 0.28), the volume hinge pushes the model to produce random S/B predictions to meet the floor. This would drive precision below 0.28, activating the precision hinge simultaneously, creating oscillation.

**Probability:** Low. Ep0 r20 demonstrated `pred_rate ≈ 0.07 AND prec_S = 0.298 AND prec_B = 0.377` simultaneously. The target operating point satisfies both floors simultaneously; they are not in inherent opposition.

**Mitigation:** 
- `prec_floor = 0.28` provides a counterbalancing force. If random signals flood in (precision → 0.20), precision hinge fires with `L_prec = 25 × [(0.08)² + (0.06)²] = 0.25`, strongly opposing the random-signal state.
- If oscillation is observed despite this, reduce `vol_floor` to 0.030 (corresponding to ~250 hard preds — still far above the ep5 collapse state).

### Risk 2 — Volume Floor Too Low → Same Collapse as Current Loss

**Description:** If `vol_floor = 0.040` is too low (i.e., the model can satisfy it while still collapsing to 200 hard preds), the hinge remains dormant and volume collapses identically to current behaviour.

**Probability:** Medium. The vol_floor calibration assumes `soft_pred_rate ≈ 0.029 + 0.942 × hard_rate`, which is a linear extrapolation from two data points (ep0 and ep5 r20). The true relationship may differ during training.

**Mitigation:** 
- Monitor `vol_loss` in the training log. If `vol_loss = 0.000` every epoch while `preds[S+B] < 300`, the floor is miscalibrated (too low).
- In that case, raise `vol_floor` to 0.050 in `US500_gvfl_r2`.
- Add explicit logging of `pred_rate` in the epoch log line for direct monitoring.

### Risk 3 — CE-Dominant Gradient Produces FLAT-Correlated Precision

**Description:** With CE as the primary signal, the model may learn to improve CE by correctly classifying the majority (FLAT) class without improving S/B precision. Since 94.94% of bars are FLAT, a model that predicts FLAT on 99% of bars achieves low CE but produces only ~50 raw preds and zero gated_pnl.

**Probability:** Low but non-zero. The focal weighting `(1 − pt)^γ` with `gamma=2.0` and alpha `[1.45, 0.16, 1.39]` already down-weights easy FLAT examples. The volume hinge provides additional gradient pressure against FLAT-only predictions.

**Mitigation:**
- Alpha up-weighting (1.45× for SELL, 1.39× for BUY vs 0.16× for FLAT) means a correctly-predicted FLAT bar contributes ~9× less CE gradient than a correctly-predicted S/B bar.
- If FLAT-collapse occurs anyway, increase `alpha_SELL` and `alpha_BUY` by 1.5× (or equivalently, lower `alpha_FLAT` to 0.05). This is a one-line change in the trainer.
- Volume hinge fires as a backstop if this collapse drives pred_rate below 0.040.

### Risk 4 — gnorm Escalation Without pr_weight Damping

**Description:** In r17–r20, gnorm escalated monotonically (29→149 in r17, 21→139 in r18). This was attributed partly to the precision penalty creating competing gradients. With CE dominant, gradient norms may be more stable — or alternatively, without the precision penalty as a damper, the CE gradient may escalate faster.

**Probability:** Low. CE is a well-behaved convex signal for each bar. The oscillation in previous runs was caused by the precision penalty fighting the CE signal. Removing this competition should reduce gradient variance.

**Mitigation:**
- Monitor `gnorm` per epoch. If gnorm exceeds 300 by ep10, consider increasing weight decay (`--wd 0.010`) in `US500_gvfl_r2`.
- The existing gradient clipping (`clip_grad_norm_`) in the training loop provides a hard ceiling.

### Risk 5 — Lower val_loss Scale Distorts Early Stopping

**Description:** `GatedVolumeFocalLoss` val_loss will be in the range 0.25–0.55 (CE-dominant) vs 3.4–4.5 (pr-dominant) for `TradeProfitabilityLoss`. The early stopping criterion (`patience=8`) compares val_loss across epochs. If val_loss plateaus at a low value early (e.g., 0.30 at ep3), patience might trigger early stopping before gated_pnl has peaked.

**Probability:** Medium. The val_loss plateau depends entirely on CE convergence, which may be faster than the 8-epoch patience.

**Mitigation:**
- Use `--patience 10` for the first run to give the CE-dominant loss more time to converge.
- If early stopping triggers before ep8, extend to `--patience 12` in the second run.
- Consider adding a separate `gated_pnl_patience` parameter to the trainer (future enhancement).

---

## Appendix A — Soft vs Hard Recall: Why vol_floor Targets Soft pred_rate

The existing `recall_floor` guard operates on **soft recall** (mean `P(class)` on true-label bars), not hard recall (argmax-based). As documented in the r17 `_comment`: soft recall sits structurally at 0.05–0.10 in FLAT-dominated outputs even when hard recall is healthy (0.30–0.45). This is because the mean probability on true SELL bars is dominated by the ~30 true SELL bars in a batch of 512, each with P(SELL) ≈ 0.20–0.40 (partial probability, not argmax).

The same structural effect applies to `pred_rate`: the soft prediction rate is dominated by partial probability mass from FLAT-predicted bars, and is always higher than the hard prediction rate. This is why `vol_floor = 0.040` (soft) maps to approximately 510 hard predictions — the soft floor fires at a much higher volume than a literal fraction-of-bars interpretation would suggest.

**Key implication:** When monitoring `vol_loss` in training logs, a value of `0.000` does NOT mean the model has 40% of bars predicted as S/B. It means the mean softmax probability on S+B classes is above 0.040 — which corresponds to approximately 510+ hard predictions on the val set. This is healthy operating behaviour.

---

## Appendix B — Cross-Reference: r20 Epoch Data Detailed

Full r20 epoch table with loss component analysis:

| Ep | val_loss | S+B raw | gated | gated_pnl | ppt   | prec_S | prec_B | rec_S | rec_B | pr%  | vol hinge (est.) | prec hinge (est.) |
|----|---------|---------|-------|-----------|-------|--------|--------|-------|-------|------|-----------------|------------------|
| 0  | 3.7428  | 2,126   | 806   | 81.0      | 0.100 | 0.298  | 0.377  | 0.372 | 0.130 | 92%  | 0 (pred_rate≈0.070) | 0 (both ≥0.28) |
| 1  | 3.6545  | 287     | 115   | 18.0      | 0.157 | 0.363  | 0.393  | 0.035 | 0.043 | 84%  | ~0.003          | 0                |
| 2  | 3.5044  | 1,042   | 395   | 48.0      | 0.122 | 0.374  | 0.421  | 0.195 | 0.104 | 88%  | 0                | 0                |
| 3  | 3.4893  | 311     | 124   | 25.0      | 0.202 | 0.386  | 0.523  | 0.073 | 0.023 | 81%  | ~0.001           | 0                |
| 4  | 3.3857  | 243     | 97    | 38.0      | 0.392 | 0.437  | 0.500  | 0.035 | 0.048 | 81%  | ~0.002           | 0                |
| 5  | 3.4716  | 100     | 42    | 7.0       | 0.167 | 0.405  | 0.538  | 0.023 | 0.010 | 79%  | **~0.012**       | 0                |
| 6  | 3.4573  | 176     | 68    | 18.0      | 0.265 | 0.424  | 0.510  | 0.041 | 0.018 | 81%  | ~0.005           | 0                |

Note: "vol hinge (est.)" is the estimated `L_vol` under `GatedVolumeFocalLoss` with `vol_floor=0.040, w_vol=120`. Under the current `TradeProfitabilityLoss`, the pr component at ep5 was `3.4716 × 0.79 ≈ 2.74` — 228× larger than the proposed vol hinge. This illustrates the fundamental difference: the current loss applies massive gradient pressure at ep5 because precision is "only" 0.405/0.538; the proposed loss applies a tiny corrective signal at ep5 because volume has collapsed.

---

## Appendix C — JSON Profile Complete Schema

The new loss class requires the following keys in `loss_params_multiclass.json`. Keys not present default to their constructor defaults.

```json
{
  "US500_gvfl_r1": {
    "_comment": "GatedVolumeFocalLoss first validation run. Replaces continuous pr_weight with vol+prec hinge guards. Motivation: r20 ep0 gated_pnl=81 (806 gated) destroyed by pr_weight=5.0 driving vol to 243 preds by ep4 (gated_pnl=38). Corr(vol,gpnl)=+0.924 vs Corr(prec,gpnl)=-0.596 across 73 epochs. CE is now ~80-90% of loss when both guards satisfied (vs 8-17% in r17-r20).",
    "loss_class":         "GatedVolumeFocalLoss",
    "gamma":              2.0,
    "trade_classes":      [0, 2],
    "vol_floor":          0.040,
    "vol_floor_weight":   120.0,
    "prec_floor":         0.28,
    "prec_floor_weight":  25.0,
    "recall_floor":       0.05,
    "rec_floor_weight":   15.0,
    "eps":                1e-6
  },
  "US500_gvfl_r2": {
    "_comment": "Fallback: increase vol_floor_weight if r1 shows vol collapse despite hinge. All other params from r1.",
    "loss_class":         "GatedVolumeFocalLoss",
    "gamma":              2.0,
    "trade_classes":      [0, 2],
    "vol_floor":          0.040,
    "vol_floor_weight":   200.0,
    "prec_floor":         0.28,
    "prec_floor_weight":  25.0,
    "recall_floor":       0.05,
    "rec_floor_weight":   15.0,
    "eps":                1e-6
  },
  "US500_gvfl_r3": {
    "_comment": "Fallback: lower vol_floor if r1/r2 show oscillation between vol and prec hinges. vol_floor=0.030 ≈ 250 hard preds (minimum acceptable). prec_floor raised to 0.30 for extra quality guard.",
    "loss_class":         "GatedVolumeFocalLoss",
    "gamma":              2.0,
    "trade_classes":      [0, 2],
    "vol_floor":          0.030,
    "vol_floor_weight":   200.0,
    "prec_floor":         0.30,
    "prec_floor_weight":  25.0,
    "recall_floor":       0.05,
    "rec_floor_weight":   15.0,
    "eps":                1e-6
  }
}
```

---

*End of proposal. Document version: 1.0 (July 2025)*
