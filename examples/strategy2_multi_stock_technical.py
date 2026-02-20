import pandas as pd
import numpy as np
from stratcraft import Strategy
from datetime import datetime
from typing import Dict, Any, List
from decorators import broadcast, rolling
from util import access_case_insensitive
import time
import talib
from talib import MA_Type
from metrics import Metrics

@broadcast
def rsi(data, period=14):
    return talib.RSI(data, timeperiod=period)
    
@broadcast
def sma(data, period=20):
    return talib.SMA(data, timeperiod=period)
    
@broadcast
def bb_upper(data, period=20, std_dev=2):
    upper, middle, lower = talib.BBANDS(data, timeperiod=period, matype=MA_Type.T3)
    return upper

@broadcast
def bb_lower(data, period=20, std_dev=2):
    upper, middle, lower = talib.BBANDS(data, timeperiod=period, matype=MA_Type.T3)
    return lower

@broadcast
def bb_middle(data, period=20, std_dev=2):
    upper, middle, lower = talib.BBANDS(data, timeperiod=period, matype=MA_Type.T3)
    return middle




class CustomStrategy(Strategy):
    """
    A multi-stock technical strategy screen stocks based on SMA, RSI, and trade based on Bollinger Bands.
    -Screen stocks for
        1. price above SMA(50)
        2. RSI(14) between 30 and 70
    -Buy the stock with highest Bollinger Bands width among qualified stocks.
    -Use 5% stop loss and 10% take profit
    -Stocks pool are: ['IBM', 'BA', 'BAC', 'GM', 'C', 'NKE', 'MMM', 'LMT', 'MCD', 'CRM']
    -Invest all available cash each time.
   """
    
    def initialize(self):
        """Initialize the strategy parameters and load data"""
        # Define the list of stock symbols to trade
        if not self.param:
            self.param = {
                'symbols': ['IBM', 'BA', 'BAC', 'GM', 'C', 'NKE', 'MMM', 'LMT', 'MCD', 'CRM'],
                'sma_period': 50,   # Long SMA period
                'rsi_period': 14,        # RSI period
                'rsi_oversold': 30,      # RSI oversold threshold
                'rsi_overbought': 70,    # RSI overbought threshold
                'stop_loss_percent': 5.0,  # Stop loss percentage
                'take_profit_percent': 10.0,  # Take profit percentage


            }

        # Load price data for our symbols
        # Each CSV has a DatetimeIndex and symbol names as columns
        # Load price data for multiple items
        for field in ['close', 'high', 'low', 'open']:
            self.data[field] = pd.read_csv(f"{field}.csv", index_col=0, parse_dates=True)
        print(f"self.data.keys(): {self.data.keys()}")
        


        
        # Calculate SMA
        self.data['sma'] = sma(self.data['close'], period=self.param['sma_period'])
        
        # Calculate RSI
        self.data['rsi'] = rsi(self.data['close'], period=self.param['rsi_period'])
        
        # Calculate Bollinger Bands
        self.data['bb_upper'] = bb_upper(self.data['close'], period=20, std_dev=2)
        self.data['bb_middle'] = bb_middle(self.data['close'], period=20, std_dev=2)
        self.data['bb_lower'] = bb_lower(self.data['close'], period=20, std_dev=2)
        
        # Calculate BB width directly
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
        
        # Create screening criteria
        # Screen 1: Price is above the long-term SMA (bullish trend)
        self.data['screen1'] = self.data['close'] > self.data['sma']
        
        # Screen 2: RSI is below overbought and above oversold (not extreme)
        self.data['screen2'] = (self.data['rsi'] < self.param['rsi_overbought']) & (self.data['rsi'] > self.param['rsi_oversold'])
        
        # Combined screen (both conditions must be true)
        self.data['combined_screen'] = self.data['screen1'] & self.data['screen2']


    
    def step_forward(self, data_handler):
        """
        Make trading decisions based on the latest data
        
        Args:
            data_handler: DataHandler instance containing the latest market data
        """
        # Check if we already have an open position
        open_positions = [trade.symbol for trade in self.portfolio.live_trades()]
        
        # If we have an open position, don't buy a new one
        if open_positions:
            # The framework automatically handles stop loss and take profit
            pass
        else:
            # No open positions, look for a new one
            
            # Get qualified symbols using DataHandler's screen method
            qualified_symbols = data_handler.screen(['combined_screen'])
            
            if qualified_symbols:
                # Get the symbol with the highest BB width using DataHandler's highest method
                best_symbol = data_handler.highest('bb_width', tickers=qualified_symbols)[0]
                
                # Use all available cash for the position
                self.buy(
                    symbol=best_symbol,
                    value=self.portfolio.cash,
                    stop_loss_percent=self.param['stop_loss_percent'],
                    take_profit_percent=self.param['take_profit_percent']
                )

if __name__ == "__main__":
    # Create and run the strategy
    initial_capital = 100000  # Set initial capital
    strategy = CustomStrategy(initial_capital=initial_capital)
    # If you get warning about mismatch index, 
    # it doesn't necessary means there is issue with data, 
    # for example, daily price data and quartely fundamental data.
    strategy.run(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31)
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