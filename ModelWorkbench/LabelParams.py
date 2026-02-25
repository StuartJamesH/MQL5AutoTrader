params = {
    'EURUSD': {
        '1m': {
            'max_horizon': 60,
            'regime_params': {
                'ma_period': 50,
                'slope_smoothness': 30,
                'regime_min_duration': 0,
                'atr_window': 60,
                'atr_lookback': 60*24,
                'atr_percentile': 1,
                'slope_threshold': 3e-6
                },
            'label_params': {
                'z_window': 14,
                'z_thresh': 1,
                'z_limit': 5,
                'atr_window': 14,
                'tp_mult': 4,
                'sl_mult': 2,
                'max_horizon': 60,
                'trend_pullback_thresh': 0,
                'skip_range': True
            }
        }
    },
    'US500': {
        '1m': {
            'max_horizon': 60,
            'regime_params': {
                'ma_period': 50,
                'slope_smoothness': 30,
                'regime_min_duration': 0,
                'atr_window': 60,
                'atr_lookback': 60*24,
                'atr_percentile': 1,
                'slope_threshold': 0.02
            },
            'label_params': {
                'z_window': 14,
                'z_thresh': 1,
                'z_limit': 5,
                'atr_window': 14,
                'tp_mult': 4,
                'sl_mult': 2,
                'max_horizon': 60,
                'trend_pullback_thresh': 0, # Increase this to make it more selective
                'skip_range': True
            }
        }
    }
}