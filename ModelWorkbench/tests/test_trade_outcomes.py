"""
Unit tests for calculate_trade_outcomes_all_candles (ModelWorkbench/Learn/labels.py).

Tests verify the post-fix contract:
  - TP hit first  →  outcome =  1
  - SL hit first  →  outcome = -1
  - No resolution before end of data  →  outcome = NaN
  - No fractional / timeout values — only {1, -1, NaN}

Run from the ModelWorkbench directory:
    python -m pytest tests/test_trade_outcomes.py -v
or:
    python tests/test_trade_outcomes.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Learn.labels import calculate_trade_outcomes_all_candles


# ─── Helpers ─────────────────────────────────────────────────────────────────

SPREAD = 1.5   # each bar: High = price+SPREAD, Low = price-SPREAD → ATR ≈ 2*SPREAD = 3.0
PRICE  = 100.0


def _make_df(n: int, price: float = PRICE, spread: float = SPREAD) -> pd.DataFrame:
    """
    Create a DataFrame where every bar has a fixed spread so ATR is stable
    after the 14-bar warm-up.  All bars are identical until a specific bar
    is overridden in the test.
    """
    return pd.DataFrame({
        'Open':   [price] * n,
        'High':   [price + spread] * n,
        'Low':    [price - spread] * n,
        'Close':  [price] * n,
        'Volume': [1000] * n,
    })


def _run(df: pd.DataFrame, tp_mult: float = 3.0, sl_mult: float = 1.5):
    return calculate_trade_outcomes_all_candles(
        df, atr_window=14, tp_mult=tp_mult, sl_mult=sl_mult
    )


def _assert_binary(outcomes: pd.DataFrame):
    """All resolved outcomes must be exactly 1.0 or -1.0 (no fractions)."""
    for col in ['buy_outcome', 'sell_outcome']:
        resolved = outcomes[col].dropna()
        bad = resolved[~resolved.isin([1.0, -1.0])]
        assert bad.empty, f"{col} contains non-binary values: {bad.values[:5]}"


# With SPREAD=1.5 and atr_window=14, stable ATR ≈ 2*SPREAD = 3.0 after warm-up.
# BUY entry = High[t0] = 101.5
# tp_mult=3, sl_mult=1.5  →  tp_buy = 101.5 + 9 = 110.5  |  sl_buy = 101.5 - 4.5 = 97.0
# The normal spread bars have High=101.5 and Low=98.5, so they never accidentally
# trigger tp_buy (110.5) or sl_buy (97.0).
TP_TRIGGER_HIGH = 112.0   # well above tp_buy (110.5)
SL_TRIGGER_LOW  = 95.5    # well below sl_buy (97.0)

# SELL entry = Low[t0] = 98.5
# tp_sell = 98.5 - 9 = 89.5  |  sl_sell = 98.5 + 4.5 = 103.0
SELL_TP_TRIGGER_LOW  = 87.0    # well below tp_sell (89.5)
SELL_SL_TRIGGER_HIGH = 105.0   # well above sl_sell (103.0)


# ─── Test 1: BUY — TP clearly hit ────────────────────────────────────────────

def test_buy_tp_hit():
    n = 100
    df = _make_df(n)
    df.at[50, 'High'] = TP_TRIGGER_HIGH   # triggers BUY TP for all entry bars before bar 50

    outcomes = _run(df)
    _assert_binary(outcomes)

    # Bars 0-49 (whose future includes bar 50) should all resolve as TP=1.
    resolved_pre = outcomes['buy_outcome'].iloc[:50].dropna()
    assert not resolved_pre.empty, "No resolved BUY outcomes before spike bar"
    assert (resolved_pre == 1).all(), (
        f"Expected all pre-spike entries to be TP=1, got: {resolved_pre.value_counts().to_dict()}"
    )
    print("PASS test_buy_tp_hit")


# ─── Test 2: BUY — SL clearly hit ────────────────────────────────────────────

def test_buy_sl_hit():
    n = 100
    df = _make_df(n)
    df.at[50, 'Low'] = SL_TRIGGER_LOW    # triggers BUY SL
    # Keep TP impossible by using a very high tp_mult override
    outcomes = _run(df, tp_mult=50.0, sl_mult=1.5)

    _assert_binary(outcomes)

    resolved = outcomes['buy_outcome'].dropna()
    assert not resolved.empty, "No resolved BUY outcomes"
    assert (resolved == -1).any(), f"Expected SL hit, got: {resolved.value_counts().to_dict()}"
    assert not (resolved == 1).any(), "Unexpected TP hit in SL test"
    print("PASS test_buy_sl_hit")


# ─── Test 3: No-horizon validation ───────────────────────────────────────────

def test_no_horizon_reaches_tp_beyond_old_limit():
    """
    Under the old code, max_horizon=1000 would cause any trade that hadn't resolved
    by bar 1000 to return a fractional timeout value.  With the fix, the function
    looks all the way to the end of the dataset, so a TP touched at bar 1100 must
    return 1, not a fraction.
    """
    n = 1200
    df = _make_df(n)
    # Only bar 1100 reaches TP; no bar reaches SL (normal Low=98.5 > sl_buy=97.0)
    df.at[1100, 'High'] = TP_TRIGGER_HIGH

    outcomes = _run(df)
    _assert_binary(outcomes)

    # Bars whose future includes bar 1100 should return 1.
    # At minimum, bar 0 (after ATR warm-up) should resolve as TP=1.
    resolved = outcomes['buy_outcome'].dropna()
    assert (resolved == 1).any(), (
        "Expected TP=1 for bars whose future includes bar 1100, "
        f"got: {resolved.value_counts().to_dict()}"
    )
    print("PASS test_no_horizon_reaches_tp_beyond_old_limit")


# ─── Test 4: SELL — TP hit ────────────────────────────────────────────────────

def test_sell_tp_hit():
    n = 100
    df = _make_df(n)
    df.at[50, 'Low'] = SELL_TP_TRIGGER_LOW   # triggers SELL TP for entry bars before bar 50

    outcomes = _run(df)
    _assert_binary(outcomes)

    # Bars 0-49 (whose future includes bar 50) should all resolve as TP=1.
    resolved_pre = outcomes['sell_outcome'].iloc[:50].dropna()
    assert not resolved_pre.empty, "No resolved SELL outcomes before spike bar"
    assert (resolved_pre == 1).all(), (
        f"Expected all pre-spike entries to be TP=1, got: {resolved_pre.value_counts().to_dict()}"
    )
    print("PASS test_sell_tp_hit")


# ─── Test 5: SELL — SL hit ────────────────────────────────────────────────────

def test_sell_sl_hit():
    n = 100
    df = _make_df(n)
    df.at[50, 'High'] = SELL_SL_TRIGGER_HIGH   # triggers SELL SL
    # Keep SELL TP impossible with very high tp_mult
    outcomes = _run(df, tp_mult=50.0, sl_mult=1.5)

    _assert_binary(outcomes)

    resolved = outcomes['sell_outcome'].dropna()
    assert not resolved.empty, "No resolved SELL outcomes"
    assert (resolved == -1).any(), f"Expected SELL SL hit, got: {resolved.value_counts().to_dict()}"
    assert not (resolved == 1).any(), "Unexpected TP hit in SELL SL test"
    print("PASS test_sell_sl_hit")


# ─── Test 6: End-of-data bars remain NaN ─────────────────────────────────────

def test_unresolved_end_bars_are_nan():
    """
    With impossible TP/SL levels (tp_mult=sl_mult=100) nothing can trigger,
    so every bar should return NaN — not a fractional timeout value.
    """
    n = 50
    df = _make_df(n)
    outcomes = _run(df, tp_mult=100.0, sl_mult=100.0)

    _assert_binary(outcomes)   # no fractions allowed
    assert outcomes['buy_outcome'].isna().all(),  "Expected all NaN for impossible TP/SL (buy)"
    assert outcomes['sell_outcome'].isna().all(), "Expected all NaN for impossible TP/SL (sell)"
    print("PASS test_unresolved_end_bars_are_nan")


# ─── Test 7: Tie-break — SL wins when TP and SL on the same bar ───────────────

def test_simultaneous_tp_sl_sl_wins():
    """
    When a future bar's High reaches TP *and* its Low reaches SL simultaneously
    (same bar index), the code uses strict < on the index, so SL wins when both
    occur at the same future bar.
    """
    n = 100
    df = _make_df(n)
    # With tp_mult=2, sl_mult=2 and ATR≈3:
    #   tp_buy ≈ 101.5 + 6 = 107.5   sl_buy ≈ 101.5 - 6 = 95.5
    # Bar 30: High=110 (≥107.5) AND Low=94 (≤95.5) → both triggered same bar → SL wins
    df.at[30, 'High'] = 110.0
    df.at[30, 'Low']  = 94.0

    outcomes = _run(df, tp_mult=2.0, sl_mult=2.0)
    _assert_binary(outcomes)

    # All resolved BUY outcomes should be -1 (SL wins tie-break)
    resolved = outcomes['buy_outcome'].dropna()
    assert not resolved.empty
    assert (resolved == -1).all(), (
        f"Expected SL to win tie-break, got: {resolved.value_counts().to_dict()}"
    )
    print("PASS test_simultaneous_tp_sl_sl_wins")


# ─── Test 8: P&L sum — 3 TP + 2 SL = net +1 ─────────────────────────────────

def test_pnl_sum():
    """
    Simulate model predicting BUY on 5 bars: 3 TP (+1 each) and 2 SL (-1 each).
    Expected net P&L = 3 - 2 = +1 (before commission).
    """
    outcomes_2d = np.array([
        [0.0,  1.0],   # bar 0: BUY predicted → TP
        [0.0, -1.0],   # bar 1: BUY predicted → SL
        [0.0,  1.0],   # bar 2: BUY predicted → TP
        [0.0, -1.0],   # bar 3: BUY predicted → SL
        [0.0,  1.0],   # bar 4: BUY predicted → TP
        [0.0,  0.0],   # bar 5: FLAT predicted → not counted
    ])
    preds = np.array([2, 2, 2, 2, 2, 1])  # 2=BUY, 1=FLAT

    buy_mask  = (preds == 2)
    sell_mask = (preds == 0)

    buy_net   = outcomes_2d[:, 1]
    sell_net  = outcomes_2d[:, 0]
    profit    = float(buy_net[buy_mask].sum()) + float(sell_net[sell_mask].sum())

    assert profit == 1.0, f"Expected P&L = 1.0, got {profit}"
    print("PASS test_pnl_sum")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        test_buy_tp_hit,
        test_buy_sl_hit,
        test_no_horizon_reaches_tp_beyond_old_limit,
        test_sell_tp_hit,
        test_sell_sl_hit,
        test_unresolved_end_bars_are_nan,
        test_simultaneous_tp_sl_sl_wins,
        test_pnl_sum,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
