## Run command example:
##   python Engine\\run_binary.py --source mt5 --mt5-mode live --symbol EURUSD.a \
##     --buy-pack "Engine/Model Packs/EURUSD_1minute_LSTM_BUY_256seq_model.pkl" \
##     --sell-pack "Engine/Model Packs/EURUSD_1minute_LSTM_SELL_256seq_model.pkl"

import argparse
import os
import pickle

import torch

from Engine import Live_Engine
from DataHandler import MT5DataHandler
from Executor import MT5LiveExecutionHandler

from StrategyBinary import TripleBarrierHiLowBinary


def _init_model_from_pack(model_pack: dict) -> torch.nn.Module:
    model_info = model_pack.get("model_info", {})
    model_params = model_pack.get("model_params", {})
    model_type = str(model_info.get("model_type", ""))

    # Import locally to keep startup light / match existing patterns
    from Learn.Models import (
        LSTMAttentionSEClassifier,
        LSTMClassifier,
        TCNAttentionSEClassifier,
        TransformerClassifier,
        TransformerSEClassifier,
        HybridLSTMTransformer,
    )

    if "TCN" in model_type:
        model_cls = TCNAttentionSEClassifier
    elif "TransformerSE" in model_type:
        model_cls = TransformerSEClassifier
    elif "Hybrid" in model_type:
        model_cls = HybridLSTMTransformer
    elif "Transformer" in model_type:
        model_cls = TransformerClassifier
    elif "LSTM" in model_type or model_type in ["LSTM_TripleBarrier", "LSTM_TripleBarrier_HiLow"]:
        # Maintain parity with existing run.py behaviour
        model_cls = LSTMAttentionSEClassifier
    else:
        # Fallback to plain LSTM if model_type is empty/unexpected
        model_cls = LSTMClassifier

    model = model_cls(**model_params)
    model.load_state_dict(state_dict=model_pack["model"])
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", choices=["csv", "mt5"], default="mt5", help="Data source (csv or mt5)")
    parser.add_argument("--mt5-mode", choices=["replay", "live"], default="replay", help="MT5 mode")
    parser.add_argument("--symbol", default="EURUSD.a", help="Symbol to use for MT5 mode")
    parser.add_argument("--start", default="2025-12-20", help="Start date for replay mode (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default="1min", help="Timeframe (e.g., 1min, 5min, 15min)")

    parser.add_argument("--buy-pack", required=True, help="Path to BUY binary model pack .pkl")
    parser.add_argument("--sell-pack", required=True, help="Path to SELL binary model pack .pkl")

    parser.add_argument("--buy-threshold", type=float, default=0.5, help="BUY trade probability threshold")
    parser.add_argument("--sell-threshold", type=float, default=0.5, help="SELL trade probability threshold")

    parser.add_argument("--risk", type=float, default=20, help="Risk budget used for position sizing")
    parser.add_argument("--patience", type=int, default=1, help="Bars to wait before cancelling pending orders")
    parser.add_argument("--maxpos", type=float, default=1.5, help="Maximum position size cap")
    parser.add_argument("--debug", action="store_true", help="Enable verbose strategy debug output")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")

    args = parser.parse_args()

    if args.source != "mt5":
        raise NotImplementedError("run_binary.py currently supports --source=mt5 only")

    # Load model packs
    if not os.path.exists(args.buy_pack):
        raise FileNotFoundError(f"BUY model pack not found: {args.buy_pack}")
    if not os.path.exists(args.sell_pack):
        raise FileNotFoundError(f"SELL model pack not found: {args.sell_pack}")

    print("Unpacking BUY model...")
    with open(args.buy_pack, "rb") as f:
        buy_pack = pickle.load(f)

    print("Unpacking SELL model...")
    with open(args.sell_pack, "rb") as f:
        sell_pack = pickle.load(f)

    buy_model = _init_model_from_pack(buy_pack)
    sell_model = _init_model_from_pack(sell_pack)

    print("Model packs loaded successfully.")

    print(f"Starting MT5 {args.mt5_mode} mode for {args.symbol} from {args.start}")

    data = MT5DataHandler(symbol=args.symbol, timeframe=args.timeframe, mode=args.mt5_mode, start=args.start)
    executor = MT5LiveExecutionHandler(deviation=10)

    strategy = TripleBarrierHiLowBinary(
        args.symbol,
        buy_model=buy_model,
        buy_model_pack=buy_pack,
        sell_model=sell_model,
        sell_model_pack=sell_pack,
        risk=args.risk,
        patience=args.patience,
        mt5_executor=executor,
        data_handler=data,
        maxpos=args.maxpos,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        debug=args.debug,
        log=not args.no_log,
    )

    engine = Live_Engine(data, strategy, executor)
    engine.run()
