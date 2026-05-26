# ML Label Learnability Framework

## Goal

Replace `ModelWorkbench\1_1 Signal Lab.ipynb` with a new notebook that does two jobs in one place:

1. continue the current label-design workflow
2. add lightweight learnability analysis before deep-learning training

This document is an execution-ready implementation spec only. It updates the old generic framework so it matches the current repository layout and conventions.

---

## Repository-Aligned Scope

The new notebook will live at:

```text
ModelWorkbench\1_1 Signal Lab.ipynb
```

It will replace the current notebook at the same path.

Reusable notebook-specific code should live in a new folder:

```text
ModelWorkbench\Lables\
```

Use the folder name exactly as above for this uplift.

The new notebook must fit the existing `ModelWorkbench` workflow:

```text
1_0 Get Historical Data.ipynb
→ 1_1 Signal Lab.ipynb   (new label + learnability notebook)
→ 1_2 Feature Lab.ipynb
→ 1_3 Feature Parity Check.ipynb
→ training notebooks / train_prod_model_cli.py / sweep scripts
```

The notebook is a research and orchestration layer. Existing core production logic remains in `ModelWorkbench\Learn\`.

---

## Current Repo Dependencies To Reuse

Do not duplicate existing production logic that already exists in `Learn`.

### Existing modules to call directly

```python
from Learn.labels import (
    causal_market_regime,
    causal_triple_barrier_hilow_trend_labeler,
    calculate_trade_outcomes_all_candles,
)

from Learn.features import (
    add_all_features,
    _add_features_EURUSD,
    _add_features_US500,
    _add_features_US2000,
    _add_features_XAUUSD,
    _add_features_SpotCrude,
)

from Learn.preprocess import preprocess_ohlcv
```

### Existing repo assets to align with

- datasets under `data\`
- label profiles in `ModelWorkbench\params\label_params.json`
- feature engineering conventions in `ModelWorkbench\Learn\features.py`
- label logic in `ModelWorkbench\Learn\labels.py`
- quick probe precedent in `ModelWorkbench\label_quality_probe.py`
- downstream training output in `Engine\Model Packs\`

The uplift should extend the research workflow without breaking compatibility with the existing training pipeline.

---

## Target Structure

```text
ModelWorkbench\
│
├── 1_1 Signal Lab.ipynb
├── Lables\
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── labels.py
│   ├── features.py
│   ├── baselines.py
│   ├── mutual_information.py
│   ├── separability.py
│   ├── nearest_neighbors.py
│   ├── regime_analysis.py
│   ├── scoring.py
│   └── visualisation.py
│
├── Learn\
│   ├── labels.py
│   ├── features.py
│   └── preprocess.py
│
├── params\
│   └── label_params.json
│
├── label_quality_probe.py
└── docs\
    └── ml_label_learnability_framework.md
```

### Design intent

- `1_1 Signal Lab.ipynb` becomes the interactive control surface
- `Lables\` contains notebook helper modules and learnability analysis code
- `Learn\` stays the source of truth for production label and feature logic

`Lables\` should wrap and coordinate existing `Learn` code, not replace it.

---

## What The New Notebook Must Do

The replacement notebook should cover the full pre-training label research loop:

1. load an OHLCV dataset already used in the repo
2. load or define a candidate label configuration
3. generate regime-aware labels using the production labeller
4. convert event labels into bar-level research targets where needed
5. build causal features using the current repo feature stack
6. run lightweight learnability diagnostics
7. compare multiple label configs side by side
8. surface a recommendation for which configs are worth deep-training
9. optionally save research outputs for later review

This notebook is not a deep-learning trainer. It is a gatekeeper before expensive training runs.

---

## Notebook Sections

## 1. Imports

The notebook should import pandas, numpy, matplotlib, and the new `Lables` helpers plus the existing `Learn` APIs they rely on.

At a minimum, the flow should support:

```python
from Lables.config import build_experiment_config
from Lables.data import load_ohlcv_dataset, load_label_profiles
from Lables.labels import generate_label_events, events_to_bar_labels
from Lables.features import build_research_features
from Lables.baselines import evaluate_baseline_models
from Lables.mutual_information import evaluate_mutual_information
from Lables.separability import evaluate_class_separability
from Lables.nearest_neighbors import evaluate_neighbor_consistency
from Lables.regime_analysis import evaluate_regime_stability
from Lables.scoring import aggregate_scores, rank_label_profiles
from Lables.visualisation import (
    plot_label_distribution,
    plot_signal_chart,
    plot_regime_performance,
    plot_feature_importance,
    plot_projection,
)
```

## 2. Experiment Configuration

The notebook should expose a compact configuration cell for:

- symbol
- dataset path
- one or more label profile names from `params\label_params.json`
- optional inline label overrides
- feature function selection
- train/validation/test split boundaries
- probe tier / depth of learnability analysis
- output save toggle

Use current repo naming and symbols such as:

```python
SYMBOL = "EURUSD"
LABEL_PROFILES = ["EURUSD_1m_dev"]
DATASET_PATH = "..\\data\\EURUSD_M1_520weeks.csv"
```

## 3. Data Loading

Load existing OHLCV CSVs from the repo data layout and normalize them to the current codebase format:

```python
["Time", "Open", "High", "Low", "Close", "Volume"]
```

Requirements:

- sort chronologically
- preserve temporal order
- validate required columns
- handle missing rows explicitly
- do not randomise or shuffle

## 4. Label Generation

The new notebook must continue to support the current Signal Lab purpose:

- inspect regime filters
- inspect candidate density
- inspect final BUY / FLAT / SELL balance
- visualise signals on price
- compare parameter profiles

Label generation must call the existing production labeller:

```python
Learn.labels.causal_triple_barrier_hilow_trend_labeler(...)
```

Regime analysis must call:

```python
Learn.labels.causal_market_regime(...)
```

If learnability probes need bar-level classes, add a helper in `Lables\labels.py` that mirrors the repo’s existing event-to-bar conversion approach used by `label_quality_probe.py`.

## 5. Feature Engineering

The notebook should reuse the existing feature stack instead of inventing a second one.

Feature creation should support:

- symbol-specific feature builders such as `_add_features_EURUSD`
- fallback to `add_all_features`
- optional inclusion/exclusion of MTF features
- preprocessing through `Learn.preprocess.preprocess_ohlcv`

The `Lables\features.py` wrapper should return:

```python
X, y, feature_names, feature_frame
```

where `feature_frame` is kept for diagnostics and plots.

## 6. Learnability Evaluation

The notebook should add structured research probes that are cheap compared with PyTorch training.

Recommended evaluation blocks:

1. label statistics
2. baseline classifier probe
3. mutual information analysis
4. class separability analysis
5. nearest-neighbor consistency
6. regime stability
7. aggregate learnability score

`label_quality_probe.py` is the nearest existing precedent and should inform the implementation shape, but the notebook should provide richer visual and side-by-side analysis.

## 7. Ranking And Recommendation

The notebook should produce a comparison table across candidate label profiles and rank them by an aggregate score.

The final question is:

> Are these labels statistically learnable from the current causal feature set, and are they good enough to justify deep-learning training?

---

## Module Responsibilities In `Lables\`

## `config.py`

Purpose:

- centralise notebook defaults
- merge selected profile config with notebook overrides
- define evaluation weights and split rules

Suggested outputs:

- experiment config dict
- resolved label config table for display

## `data.py`

Purpose:

- load OHLCV CSVs from `..\data\...`
- load label profiles from `ModelWorkbench\params\label_params.json`
- provide symbol-to-feature-function lookup

This module should reflect the repo’s existing symbol coverage:

- `EURUSD`
- `US500`
- `XAUUSD`
- `US2000`
- `NAS100`
- `SpotCrude`

## `labels.py`

Purpose:

- wrap current `Learn.labels` functions for notebook use
- generate events for one or many profiles
- convert events to bar-level targets for research models
- summarise label density and TP/SL/timeout mix

This module should preserve the current repo semantics:

- regime-aware filtering first
- triple-barrier event resolution second
- winning events mapped to BUY or SELL
- non-winning bars mapped to FLAT when bar labels are required

## `features.py`

Purpose:

- call the appropriate existing feature builder
- align features with generated bar labels
- drop rows that are not yet feature-complete
- run current preprocessing consistently

This must stay compatible with downstream training expectations.

## `baselines.py`

Purpose:

- run fast conventional ML probes with strict time ordering

Recommended models:

- LogisticRegression
- RandomForestClassifier
- LightGBM if already available in the environment
- XGBoost only if already available; otherwise optional

Metrics:

- balanced accuracy
- macro F1
- per-class precision / recall
- ROC-AUC or PR-AUC where shape permits

The goal is not production modelling. It is only to detect whether structure exists.

## `mutual_information.py`

Purpose:

- estimate how much predictive information features contain about the label
- rank top features
- test whether signal is diffuse or concentrated in a few features

## `separability.py`

Purpose:

- measure how well classes separate in feature space
- provide 2D projections for visual inspection

Recommended metrics:

- PCA projection
- optional UMAP if already installed
- silhouette score
- Fisher-style class distance measures

## `nearest_neighbors.py`

Purpose:

- test whether similar historical feature states produce similar outcomes

Recommended outputs:

- neighbor agreement
- local entropy
- class purity

## `regime_analysis.py`

Purpose:

- break results down by regime
- identify whether a label profile only works in narrow market conditions

Use existing regime concepts already embedded in the repo:

- trend / range state
- volatility state
- session or time bucket where available

## `scoring.py`

Purpose:

- combine evaluation outputs into a configurable learnability score
- rank label profiles
- generate a final recommendation table

Example weighting direction:

```python
learnability_score = (
    0.35 * baseline_score +
    0.20 * mutual_information_score +
    0.20 * neighbor_consistency_score +
    0.15 * separability_score +
    0.10 * regime_stability_score
)
```

Use configurable weights rather than hardcoding final values deep in the notebook.

## `visualisation.py`

Purpose:

- keep plotting code out of the notebook
- produce notebook-ready matplotlib figures

Use matplotlib only unless the repo already depends on something else for the same task.

---

## Required Outputs

The completed notebook should produce the following outputs for one or more label profiles.

## 1. Label quality summary

Example columns:

| label_profile | regime_coverage | candidate_density | label_density | tp_rate | sl_rate | timeout_rate |
|---|---:|---:|---:|---:|---:|---:|

## 2. Learnability summary

Example columns:

| label_profile | baseline_score | mi_score | separability_score | neighbor_score | regime_score | learnability_score |
|---|---:|---:|---:|---:|---:|---:|

## 3. Visual diagnostics

- price chart with labelled signals
- label distribution chart
- feature-importance chart
- PCA projection
- regime breakdown chart

## 4. Final recommendation

A compact conclusion that states:

- which label profile ranks best
- whether the best profile is strong enough for deep-learning training
- whether the bottleneck is labels, features, or regime concentration

---

## Validation Rules

All learnability analysis must follow the same causal standards as the rest of the repo.

### Must do

- preserve chronological ordering
- avoid future leakage
- fit scalers on train only when applicable
- use walk-forward or strict temporal holdout evaluation
- use only past-computable features

### Must not do

- random shuffling
- future-normalised features
- leakage from future bars into label or feature construction
- notebook-only logic that silently diverges from `Learn`

---

## Compatibility Rules

The uplift should not change the downstream training contract.

The notebook must remain compatible with:

- `ModelWorkbench\1_2 Feature Lab.ipynb`
- training notebooks
- `train_prod_model_cli.py`
- sweep scripts
- `params\label_params.json`

If the notebook saves a chosen profile or an updated parameter block, the shape should remain compatible with the existing label profile format already used by the repo.

---

## Explicit Non-Goals

This uplift does **not** implement:

- deep-learning model training inside `1_1`
- changes to the production label algorithm in `Learn\labels.py`
- changes to live trading code in `Engine\`
- migration of existing training notebooks

It only prepares a stronger pre-training research notebook and the helper modules it will rely on.

---

## Recommended Build Order

1. replace the narrative and section structure of `1_1 Signal Lab.ipynb`
2. create the `ModelWorkbench\Lables\` package
3. add config and data-loading helpers
4. add wrappers around existing `Learn.labels` logic
5. add feature preparation wrappers around existing `Learn.features` and `Learn.preprocess`
6. add learnability evaluation modules
7. add plotting helpers
8. wire notebook sections to the new helpers
9. confirm output tables and saved artifacts are useful for choosing training candidates

---

## Final Objective

After this uplift, `ModelWorkbench\1_1 Signal Lab.ipynb` should answer both of these questions in one workflow:

1. **Are these label parameters producing clean, selective, regime-aware signals?**
2. **Are those labels learnable enough from the current causal feature stack to justify a full training run?**

That becomes the decision gate before spending time on `1_2`, sweep scripts, or PyTorch production training.
