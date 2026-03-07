"""
Example script showing how to use the TicketBook system with the trading engine.

This demonstrates the complete integration:
1. TicketBook initialization
2. Executor integration
3. Strategy setup
4. Engine orchestration
"""

import sys
sys.path.append('Engine')

from Engine.TicketBook import TicketBook, OrderStatus
from Executor import MT5LiveExecutionHandler
from Strategy import TripleBarrierHiLow
from Learn.Models import LSTMClassifier
from DataHandler import MT5DataHandler
from Engine import Live_Engine
import torch
import pickle

# Example: Load your trained model
# model = torch.load('path/to/model.pth')
# model_pack = torch.load('path/to/model_pack.pkl')

def main():
    # 1. Initialize TicketBook with SQLite persistence
    ticketbook = TicketBook(db_path="trading_journal.db", use_memory_only=False)
    print(f"TicketBook initialized: {ticketbook}")
    
    # 2. Initialize Executor with TicketBook integration
    executor = MT5LiveExecutionHandler(
        deviation=10,
        magic=234000,
        ticketbook=ticketbook  # Pass TicketBook to Executor
    )
    print("Executor initialized with TicketBook integration")
    
    # 3. Initialize Data Handler for live trading
    data_handler = MT5DataHandler(
        symbol='EURUSD.a',
        timeframe='M1',
        mode='live'
    )
    print("Data handler initialized for live trading")
    
    # 4. Initialize Strategy
    # Option A: Pass TicketBook directly to Strategy

    # Import model pack
    print('Unpacking model...')
    with open('Engine/Model Packs/EURUSD_1minute_LSTM_TripleBarrier_HiLow_256seq_20251208__precision_only_model.pkl', 'rb') as f:
        model_pack = pickle.load(f)

    # Get model parameters from model pack
    model_info = model_pack['model_info']
    model_params = model_pack['model_params']

    # Init model
    model = LSTMClassifier(**model_params)
    model.load_state_dict(state_dict=model_pack['model'])
    model.eval()

    strategy = TripleBarrierHiLow(
        symbol='EURUSD.a',
        model=model,  # Replace with your actual model
        model_pack=model_pack,  # Replace with your actual model_pack
        patience=5,  # Orders expire after 5 minutes
        risk=50,
        data_handler=data_handler,
        strategy_name="TripleBarrierHiLow_EURUSD_1M",
        ticketbook=ticketbook  # Option A: Pass TicketBook here
    )
    # Option B: Let Engine pass TicketBook to Strategy (omit ticketbook parameter above)
    # The Engine will automatically set strategy.ticketbook if not already set
    print(f"Strategy initialized: {strategy.strategy_name}")
    
    # 5. Initialize Engine with TicketBook
    engine = Live_Engine(
        data_handler=data_handler,
        strategy=strategy,
        executor=executor,
        ticketbook=ticketbook  # Pass TicketBook to Engine
    )
    print("Engine initialized with TicketBook")
    
    # 6. Run the trading engine
    print("\n" + "="*50)
    print("Starting live trading engine...")
    print("="*50 + "\n")
    
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("Trading stopped by user")
        print("="*50)
    finally:
        # 7. Print trading statistics
        print("\n" + "="*50)
        print("TRADING STATISTICS")
        print("="*50)
        stats = ticketbook.get_statistics(symbol='EURUSD')
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 8. Export trade history
        print("\n" + "="*50)
        print("EXPORTING TRADE HISTORY")
        print("="*50)
        history = ticketbook.get_order_history()
        if not history.empty:
            history.to_csv('trade_history.csv', index=False)
            print(f"Exported {len(history)} orders to trade_history.csv")
        else:
            print("No trades to export")


def query_ticketbook_examples():
    """
    Example queries you can run on the TicketBook
    """
    ticketbook = TicketBook(db_path="trading_journal.db")
    
    # Get all active pending orders
    pending = ticketbook.get_active_pending_orders()
    print(f"Active pending orders: {len(pending)}")
    
    # Get all filled orders for EURUSD
    filled = ticketbook.get_order_history(symbol='EURUSD', status=OrderStatus.FILLED)
    print(f"Filled orders for EURUSD: {len(filled)}")
    
    # Get closed trades (completed trades with PnL)
    closed_trades = ticketbook.get_order_history(status=OrderStatus.CLOSED)
    print(f"Closed trades: {len(closed_trades)}")
    
    # Get a specific order by ticket
    order = ticketbook.get_order(ticket=12345678)
    if order:
        print(f"Order {order.ticket}: {order.symbol} {order.side} {order.status}")
    
    # Calculate statistics
    stats = ticketbook.get_statistics()
    print(f"\nTrading Statistics:")
    print(f"Total trades: {stats['total_trades']}")
    print(f"Win rate: {stats['win_rate']:.2%}")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Profit factor: {stats['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
    
    # Uncomment to run query examples:
    # query_ticketbook_examples()
