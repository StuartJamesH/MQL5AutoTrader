## Run command python 
import pandas as pd
import pickle

import argparse
from Strategy import TripleBarrierHiLow_XAUUSD
from Engine import Live_Engine

from DataHandler import MT5DataHandler
from Executor import MT5LiveExecutionHandler

from Learn.features import add_all_features, add_selected_features
from Learn.preprocess import preprocess_ohlcv


if __name__ == "__main__":

    # Import model pack
    print('Unpacking model...')
    with open('Engine/Model Packs/XAUUSD_1minute_LSTM_TripleBarrier_HiLow_256seq_20260112__DEV_model.pkl', 'rb') as f:
        model_pack = pickle.load(f)

    # Get model parameters from model pack
    model_info = model_pack['model_info']
    model_params = model_pack['model_params']

    # Init model
    if model_info['model_type'] in ['LSTM_TripleBarrier', 'LSTM_TripleBarrier_HiLow']:
        from Learn.Models import LSTMAttentionSEClassifier
        model = LSTMAttentionSEClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_info['model_type']}")

    model.load_state_dict(state_dict=model_pack['model'])
    model.eval()
    print(f'Model pack loaded successfully.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['csv', 'mt5'], default='csv', help='Data source (csv or mt5)')
    parser.add_argument('--mt5-mode', choices=['replay', 'live'], default='replay', help='MT5 mode when --source=mt5')
    parser.add_argument('--symbol', default='XAUUSD.a', help='Symbol to use for MT5 mode')
    parser.add_argument('--start', default='2025-12-20', help='Start date for replay mode (YYYY-MM-DD)')
    args = parser.parse_args()

    # Init connection to MT5
    if MT5DataHandler is None or MT5LiveExecutionHandler is None:
        raise RuntimeError('MT5 adapters not available. Ensure MetaTrader5 is installed and engine/mt5_execution.py is present.')
    print(f"Starting MT5 {args.mt5_mode} mode for {args.symbol} from {args.start}")
    
    timeframe = '1min'

    # Configure engine components and run
    data = MT5DataHandler(symbol=args.symbol, timeframe=timeframe, mode=args.mt5_mode, start=args.start)
    executor = MT5LiveExecutionHandler(deviation=10)
    strategy = TripleBarrierHiLow_XAUUSD(args.symbol, model=model, model_pack=model_pack, volume=0.1, patience=1, mt5_executor=executor, data_handler=data, debug=True, log=True)
    engine = Live_Engine(data, strategy, executor)
    engine.run()