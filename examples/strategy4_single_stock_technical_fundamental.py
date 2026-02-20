from stratcraft import Strategy, Direction, DataHandler
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
from decorators import broadcast, rolling, available
from indicators import sma, rsi
from util import access_case_insensitive
from metrics import Metrics
START_DATE='2022-01-01'
END_DATE='2023-12-31'
class CustomStrategy(Strategy):
    """
    Single stock with one technical indicator and one fundamental indicator
    - Trade only AAPL
    - Buy when RSI(14) is below 35 and EBITDA quarterly growth is positive
    - Sell when RSI(14) is above 65 only when hold position(No short selling)
    - Use 5% stop loss and 10% take profit


    """
    
    def initialize(self):
        """Initialize the strategy with data and indicators"""
        # Define parameters
        if not self.param:
            self.param = {
                'symbol': 'AAPL',
                'rsi_period': 14,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'stop_loss_percent': 5.0,
                'take_profit_percent': 10.0,
                'position_size_percent': 20.0
            }
        
        # Load price data for our symbol
        price_data = pd.read_csv(f"{self.param['symbol']}.csv", index_col=0, parse_dates=True)

        # Store the price data in self.data
        for item in price_data.keys():
            self.data[item] = price_data[item]
        
        # Calculate technical indicators and store in self.data
        self.data['rsi'] = rsi(self.data['close'], period=self.param['rsi_period'])
        
        # Load fundamental data - EBITDA
        # Get fundamental data and filing dates
        fundamental_data = get_fundamental(self.param['symbol'], ['is_ebitda', 'fillingDate'])
       # Important: Do not store the fundamental data in self.data as it has different index, may case warning and error.
        # Just use it as intermediate variable to calculate indicators, or use broadcast decorator to align index to daily price data.
 
        # Create a function to check if EBITDA growth is positive
        @available(self.data['close'].index, length=2)
        def is_ebitda_growth_positive(ebitda_data):
            return bool(ebitda_data.iloc[-1] > ebitda_data.iloc[-2])
        
        # Calculate positive EBITDA growth indicator
        self.data['ebitda_growth_positive'] = is_ebitda_growth_positive(
            fundamental_data['is_ebitda'][self.param['symbol']],
            available_date=fundamental_data['fillingDate'][self.param['symbol']]
        )
        
        
    def step_forward(self, data_handler):
        """
        Make trading decisions based on the latest data
        
        Args:
            data_handler: DataHandler instance containing the latest market data
        """


        # Access the latest data points using DataHandler


        # Get fundamental data if available
        ebitda_growth_positive = bool(data_handler['ebitda_growth_positive'])

        # Check if we already have a position
        has_position = len(self.portfolio.live_trades()) > 0
        
        # Check for buy signal
        # Buy when RSI is oversold and EBITDA is growing
        buy_signal = data_handler['rsi'] < self.param['rsi_oversold'] and ebitda_growth_positive
        
        if buy_signal and not has_position:
            # Calculate position size
            position_value = self.portfolio.cash * (self.param['position_size_percent'] / 100.0)
            
            # Buy with stop loss and take profit
            self.buy(
                symbol=self.param['symbol'],
                value=position_value,
                stop_loss_percent=self.param['stop_loss_percent'],
                take_profit_percent=self.param['take_profit_percent']
            )
        
        

                
        # Check for sell signal: RSI is overbought
        elif data_handler['rsi'] > self.param['rsi_overbought'] and has_position:
            # Close any existing positions for this symbol
            for trade in self.portfolio.live_trades():
                if trade.symbol == self.param['symbol'] and trade.direction == Direction.LONG:
                    # Create a sell trade to close the position
                    self.sell(symbol=self.param['symbol'], quantity=trade.quantity)
             
             
    
if __name__ == "__main__":
    # Create and run the strategy
    strategy = CustomStrategy()

    # If you get warning about mismatch index, 
    # it doesn't necessary means there is issue with data, 
    # for example, daily price data and quartely fundamental data.
    # Run backtest from 2022-01-01 to 2023-12-31
    strategy.run(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31),
        data_length=2  # Need at least 2 data points to compare current and previous
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
        
  
  