class Live_Engine:
    def __init__(self, data_handler, strategy, executor):
        self.data_handler = data_handler
        self.strategy = strategy
        self.executor = executor
        self.order_type = strategy.order_type

    def run(self):
        for bar in self.data_handler.get_next_bar():
            orders = self.strategy.on_bar(bar)
            for order in orders:
                if self.order_type == 'market':
                    self.executor.submit_market_order(order)
                elif self.order_type == 'stop':
                    self.executor.submit_stop_order(order)