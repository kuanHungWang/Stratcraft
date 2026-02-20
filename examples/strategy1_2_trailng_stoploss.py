from stratcraft import Strategy, Direction, TrailingStopLoss
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any
from decorators import broadcast, rolling
from indicators import sma, rsi
from util import access_case_insensitive
from metrics import Metrics



class CustomStrategy(Strategy):
    """
    
    This strategy uses a SMA crossover system with trailing stop loss:
    - Trade only one stock
    - Buy when price crosses above the SMA(20)
    - Sell when price crosses below the SMA(20)
    - Uses trailing stop loss and take profit, where:
        initial_stop_loss = entry price - 1.0,
        start updating stoploss when close price higher than entry price + 1.0
        for each update, set stoploss = close price - 1.0
    - Use regular take profit of 10%
    - Invest 10% of available cash each time
    
    """
    
    def initialize(self):
        """Initialize the strategy with data and indicators"""
        if not self.param:
            self.param = {
                'symbol': 'AAPL',
                'sma_period': 20,
                'stop_loss_percent': 5.0,
                'take_profit_percent': 10.0,
                'investment_percentage': 0.10
            }
        # Load price data fields
        self.data = pd.read_csv(f"{self.param['symbol']}.csv", index_col=0, parse_dates=True)
        # Calculate SMA indicator and store in self.data
        self.data['sma'] = sma(self.data['close'], period=self.param['sma_period'])
        
    def step_forward(self, data_handler):
        """
        Make trading decisions based on the latest data
        
        Args:
            data_handler: DataHandler instance containing the latest market data
                - Access latest data with data_handler['field_name']
        """


        # Access the latest data points using the DataHandler's convenient syntax
        current_close = data_handler['close']
        current_sma = data_handler['sma']
        previous_close = data_handler[('close', -2)]  # in run(), set data_length=2, since we want to access the previous two data points
        previous_sma = data_handler[('sma', -2)]
        
        # Check for buy signal: price crosses above SMA
        if previous_close < previous_sma and current_close > current_sma:
            # Calculate position size (invest 10% of portfolio)
            position_value = self.portfolio.cash * self.param['investment_percentage']
            
            # Buy with stop loss and take profit
            trade = self.buy(
                symbol=self.param['symbol'],
                value=position_value,
                take_profit_percent=self.param['take_profit_percent']
            )
            if trade:
                # Add trailing stop loss
                trailing_stop = TrailingStopLoss(
                    price=trade.entry_price - 1.0,
                    distance=1.0,
                    threshold=trade.entry_price + 1.0,
                    direction=Direction.LONG
                )
                # Note: use dot notation to set stop loss, not add_stop_loss()
                trade.stop_loss = trailing_stop
    
    

if __name__ == "__main__":
    # Create and run the strategy
    initial_capital = 100000  # Default initial capital
    strategy = CustomStrategy(initial_capital=initial_capital)
    # If you get warning about mismatch index, 
    # it doesn't necessary means there is issue with data, 
    # for example, daily price data and quartely fundamental data.
    strategy.run(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31),
        data_length=2 # Set 2 since we have data_handler[('close', -2)]
    )
    # get rowdata of backtest results.
    trade_history_df = strategy.portfolio.trade_history()
    if trade_history_df.empty:
        print("No trades executed during the backtest period.")
 
    pl_history_df = strategy.portfolio.pl_history()
    if pl_history_df.empty:
        print("No profit/loss data available.")

    # calculate metrics with backtest rowdata
    if not pl_history_df.empty and not trade_history_df.empty:
        metrics = Metrics(trade_history_df, pl_history_df)
        result = metrics.metrics(concise=True)
        print('Backtest finished, results: ')
        Metrics.pretty_print(result)