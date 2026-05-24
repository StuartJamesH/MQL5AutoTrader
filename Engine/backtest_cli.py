"""
backtest_cli.py — CLI runner for TripleBarrierHiLowMulticlass stepwise backtests.

Runs a backtest against a local OHLCV CSV dataset using
:class:`~Engine.Backtest_Engine` and :class:`~Executor.BacktestExecutionHandler`.
No MetaTrader 5 connection is required.

Data source
-----------
Each ``--symbol`` maps to a canonical CSV at ``data/{SYMBOL}_M1_520weeks.csv``
in the repository root.  The mapping is defined in ``SYMBOL_CSV_MAP`` below.

Results
-------
After the run, a JSON results file is written to ``Engine/backtest_results/``.
Use ``--output`` to specify a custom path.

Usage
-----
Run from the repository root::

    python Engine/backtest_cli.py \\
        --symbol US500 \\
        --model-pack "US500_TCN_prod_model.pkl" \\
        --n-rows 50000 \\
        --logging INFO

Run ``python Engine/backtest_cli.py --help`` for the full argument list.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure Engine/ is on sys.path so local imports resolve correctly when the
# script is run as `python Engine/backtest_cli.py` from the repo root.
_ENGINE_DIR = Path(__file__).resolve().parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from Engine import Backtest_Engine, configure_backtest_logging  # noqa: E402
from DataHandler import DataHandler                              # noqa: E402
from Executor import BacktestExecutionHandler                    # noqa: E402
from TicketBook import TicketBook                                # noqa: E402
from Strategy import TripleBarrierHiLowMulticlass               # noqa: E402

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_REPO_ROOT = _ENGINE_DIR.parent
_DATA_DIR = _REPO_ROOT / "data"
_MODEL_PACK_DIR = _ENGINE_DIR / "Model Packs"
_RESULTS_DIR = _ENGINE_DIR / "backtest_results"

# Symbol → CSV filename in _DATA_DIR
SYMBOL_CSV_MAP: dict[str, str] = {
    "US500":     "US500_M1_520weeks.csv",
    "EURUSD":    "EURUSD_M1_520weeks.csv",
    "XAUUSD":    "XAUUSD_M1_520weeks.csv",
    "US2000":    "US2000_M1_520weeks.csv",
    "NAS100":    "NAS100_M1_520weeks.csv",
    "SpotCrude": "SpotCrude_M1_520weeks.csv",
}

# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def _resolve_csv(symbol: str) -> Path:
    """Return the absolute path to the OHLCV CSV for *symbol*.

    Looks up ``SYMBOL_CSV_MAP`` first, then falls back to a filename of the
    form ``{SYMBOL}_M1_520weeks.csv`` in the data directory.

    Raises
    ------
    FileNotFoundError
        If no matching CSV is found.
    """
    csv_name = SYMBOL_CSV_MAP.get(symbol, f"{symbol}_M1_520weeks.csv")
    path = _DATA_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found for symbol '{symbol}'.\n"
            f"  Expected: {path}\n"
            f"  Known symbols: {sorted(SYMBOL_CSV_MAP)}"
        )
    return path


def _resolve_model_pack(name: str) -> Path:
    """Resolve *name* to an existing ``.pkl`` path.

    Resolution order:

    1. As-is (absolute or relative path from cwd).
    2. Relative to ``Engine/Model Packs/``.

    Raises
    ------
    FileNotFoundError
        If neither candidate exists.
    """
    direct = Path(name)
    if direct.exists():
        return direct.resolve()

    candidate = _MODEL_PACK_DIR / name
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Model pack not found: '{name}'\n"
        f"  Tried: {direct.resolve()}\n"
        f"       : {candidate}"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtest_cli",
        description=(
            "Run a TripleBarrierHiLowMulticlass stepwise backtest on CSV data.\n\n"
            f"Data directory : {_DATA_DIR}\n"
            f"Model pack dir : {_MODEL_PACK_DIR}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "--symbol",
        required=True,
        metavar="SYMBOL",
        help=(
            "Symbol name, e.g. 'US500' or 'EURUSD'.  "
            "Hardcodes to data/{SYMBOL}_M1_520weeks.csv."
        ),
    )
    req.add_argument(
        "--model-pack",
        required=True,
        dest="model_pack",
        metavar="NAME_OR_PATH",
        help=(
            "Model pack filename relative to 'Engine/Model Packs/', "
            "e.g. 'US500_TCN_prod_model.pkl', or a full path to the .pkl file."
        ),
    )

    # ------------------------------------------------------------------
    # Data / run control
    # ------------------------------------------------------------------
    data = parser.add_argument_group("data options")
    data.add_argument(
        "--n-rows",
        type=int,
        default=None,
        dest="n_rows",
        metavar="N",
        help=(
            "Use only the most-recent N rows from the CSV.  "
            "Omit to use all rows."
        ),
    )
    data.add_argument(
        "--logging",
        default="INFO",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logging level.  (default: INFO)",
    )

    # ------------------------------------------------------------------
    # Strategy tuning
    # ------------------------------------------------------------------
    strat = parser.add_argument_group("strategy parameters")
    strat.add_argument(
        "--patience",
        type=int,
        default=1,
        help="Bars before an unfilled stop order expires.  (default: 1)",
    )
    strat.add_argument(
        "--risk",
        type=float,
        default=20.0,
        help="Fixed-risk amount per trade in account currency.  (default: 20.0)",
    )
    strat.add_argument(
        "--maxpos",
        type=float,
        default=5.0,
        help="Maximum position size cap in lots.  (default: 5.0)",
    )
    strat.add_argument(
        "--trade-threshold",
        type=float,
        default=0.6,
        dest="trade_threshold",
        metavar="PROB",
        help=(
            "Minimum softmax probability required to place a trade.  "
            "(default: 0.6)"
        ),
    )
    strat.add_argument(
        "--donchian-length",
        type=int,
        default=60,
        dest="donchian_length",
        help="Donchian channel look-back for the trend gate.  (default: 60)",
    )
    strat.add_argument(
        "--maxlen",
        type=int,
        default=7_000,
        help=(
            "Ring-buffer size (bars) for the strategy's price history.  "
            "Also the warm-up period before predictions begin.  (default: 7000)"
        ),
    )
    strat.add_argument(
        "--point-value",
        type=float,
        default=1.0,
        dest="point_value",
        metavar="PV",
        help=(
            "Dollar value per 1.0 price-unit move per lot.  "
            "Used for P&L simulation.  (default: 1.0)"
        ),
    )
    strat.add_argument(
        "--no-log-trades",
        action="store_true",
        dest="no_log_trades",
        help="Disable per-bar CSV trade logging inside the strategy.",
    )
    strat.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose per-bar debug output from the strategy.",
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    out = parser.add_argument_group("output options")
    out.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Path for the JSON results file.  "
            "Defaults to Engine/backtest_results/backtest_{symbol}_{timestamp}.json"
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Configure logging before anything else ---
    configure_backtest_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    log = logging.getLogger(__name__)

    # --- Resolve paths ---
    try:
        csv_path = _resolve_csv(args.symbol)
        pkl_path = _resolve_model_pack(args.model_pack)
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return  # unreachable; satisfies type checkers

    log.info("Symbol      : %s", args.symbol)
    log.info("CSV source  : %s", csv_path)
    log.info("Model pack  : %s", pkl_path)

    # --- Load CSV ---
    log.info("Loading CSV data (%s)...", csv_path.name)
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    # Parse the time column so bar.Time is a datetime, not a string.
    # Without this, _bar_time_as_utc falls back to datetime.utcnow() and
    # every order expires immediately because the wall-clock time is far
    # ahead of historical bar timestamps.
    time_col = "Time" if "Time" in df.columns else "Date"
    df[time_col] = pd.to_datetime(df[time_col])

    if args.n_rows is not None:
        df = df.iloc[-args.n_rows:].reset_index(drop=True)
        log.info("Using most-recent %d of %d rows", len(df), total_rows)
    else:
        log.info("Loaded %d rows", total_rows)

    # --- Load model pack ---
    log.info("Loading model pack...")
    with open(pkl_path, "rb") as fh:
        model_pack = pickle.load(fh)
    log.info("Model pack loaded.")

    # --- Wire components ---
    ticket_book = TicketBook(use_memory_only=True)

    executor = BacktestExecutionHandler(
        point_value=args.point_value,
        ticket_book=ticket_book,
    )
    executor.set_point_value(args.symbol, args.point_value)

    data_handler = DataHandler(df)

    strategy = TripleBarrierHiLowMulticlass(
        symbol=args.symbol,
        model_pack=model_pack,
        patience=args.patience,
        maxlen=args.maxlen,
        risk=args.risk,
        maxpos=args.maxpos,
        trade_threshold=args.trade_threshold,
        donchian_length=args.donchian_length,
        mt5_executor=executor,
        data_handler=data_handler,
        debug=args.debug,
        log=not args.no_log_trades,
        ticket_book=ticket_book,
    )

    engine = Backtest_Engine(data_handler, strategy, executor)

    # --- Run ---
    log.info(
        "Backtest starting — patience=%d risk=%.1f threshold=%.2f "
        "donchian=%d maxlen=%d point_value=%.2f",
        args.patience,
        args.risk,
        args.trade_threshold,
        args.donchian_length,
        args.maxlen,
        args.point_value,
    )

    summary = engine.run()

    # --- Extract model metadata from pack ---
    raw_model_info  = model_pack.get("model_info",  {})
    raw_val_metrics = model_pack.get("val_metrics", {})

    # Serialise model_info (values may be non-JSON-native types)
    model_info_out = {
        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
        for k, v in raw_model_info.items()
    }

    # Keep only scalar val_metrics (drop per-epoch curves)
    val_metrics_out = {
        k: (float(v) if not isinstance(v, str) else v)
        for k, v in raw_val_metrics.items()
        if not k.endswith("_curve")
    }

    # --- Build full results record ---
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "run_timestamp": run_ts,
        "symbol":        args.symbol,
        "csv_source":    str(csv_path),
        "model_pack":    str(pkl_path),
        "n_rows":        len(df),
        "strategy_params": {
            "patience":         args.patience,
            "risk":             args.risk,
            "maxpos":           args.maxpos,
            "trade_threshold":  args.trade_threshold,
            "donchian_length":  args.donchian_length,
            "maxlen":           args.maxlen,
            "point_value":      args.point_value,
        },
        "model_info":    model_info_out,
        "val_metrics":   val_metrics_out,
        "backtest_performance": summary,
    }

    # --- Save JSON results file ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _RESULTS_DIR / f"backtest_{args.symbol}_{run_ts}.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info("Results saved: %s", output_path)

    # --- Console summary ---
    s = summary
    mi = model_info_out
    vm = val_metrics_out
    W = 60
    div = "=" * W

    def _row(label: str, value: str) -> str:
        return f"  {label:<28}{value}"

    print(f"\n{div}")
    print("  BACKTEST RESULTS")
    print(div)
    print(_row("Symbol",          args.symbol))
    print(_row("CSV source",      csv_path.name))
    print(_row("Model pack",      pkl_path.name))
    print(_row("Rows processed",  f"{len(df):,}"))
    print(_row("Run timestamp",   run_ts))

    print(f"\n  {'— Strategy Parameters —':^{W-4}}")
    print(_row("Patience",        f"{args.patience} bar(s)"))
    print(_row("Risk / trade",    f"{args.risk:.2f}"))
    print(_row("Trade threshold", f"{args.trade_threshold:.2f}"))
    print(_row("Point value",     f"{args.point_value:.2f}"))

    if mi:
        print(f"\n  {'— Model Info —':^{W-4}}")
        for key in ("model_type", "model_version", "date_trained", "seq_len",
                    "n_epochs", "best_epoch", "best_val_loss",
                    "label_profile", "model_profile", "loss_profile"):
            if key in mi:
                print(_row(key.replace("_", " ").title(), str(mi[key])))

    if vm:
        print(f"\n  {'— Validation Metrics (training) —':^{W-4}}")
        for key in ("best_f1_buy", "best_f1_sell",
                    "best_precision_buy", "best_precision_sell",
                    "final_f1_buy", "final_f1_sell",
                    "final_profit"):
            if key in vm:
                label = key.replace("_", " ").title()
                val   = vm[key]
                fmt   = f"{val:.4f}" if isinstance(val, float) else str(val)
                print(_row(label, fmt))

    print(f"\n  {'— Backtest Performance —':^{W-4}}")
    print(_row("Total trades",          f"{s['trades']}"))
    print(_row("  Buy trades",          f"{s['buy_trades']}"))
    print(_row("  Sell trades",         f"{s['sell_trades']}"))
    print(_row("Total P&L",             f"{s['total_pnl']:.2f}"))
    print(_row("Avg P&L / trade",       f"{s['avg_pnl']:.2f}"))
    print(_row("Overall win rate",      f"{s['win_rate'] * 100:.1f}%"))
    print(_row("  Buy win rate",        f"{s['buy_win_rate'] * 100:.1f}%"))
    print(_row("  Sell win rate",       f"{s['sell_win_rate'] * 100:.1f}%"))
    print(_row("Gross profit",          f"{s['gross_profit']:.2f}"))
    print(_row("Gross loss",            f"{s['gross_loss']:.2f}"))
    print(_row("Profit factor",         f"{s['profit_factor']:.4f}"))
    print(_row("Max single win",        f"{s['max_win']:.2f}"))
    print(_row("Max single loss",       f"{s['max_loss']:.2f}"))
    print(_row("Max consecutive wins",  f"{s['max_consecutive_wins']}"))
    print(_row("Max consecutive losses",f"{s['max_consecutive_losses']}"))
    print(div)
    print(f"  Results file: {output_path}")
    print(div)


if __name__ == "__main__":
    main()
