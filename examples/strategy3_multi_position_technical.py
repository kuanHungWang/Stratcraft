from stratcraft import Strategy, Direction, screen, screen_sort
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decorators import broadcast, rolling
from indicators import sma, rsi, bb_upper, bb_middle, bb_lower
from util import access_case_insensitive
from data_loader import get_price
from metrics import Metrics

class CustomStrategy(Strategy):
    """
    A multi-stock technical strategy screen stocks based on SMA, RSI, and trade based on Bollinger Bands.
    -Screen stocks for
        1. SMA(10) cross above SMA(50)
        2. close price cross above SMA(10)
    -Buy the stock with highest Bollinger Bands width(20 day, std dev 2) among qualified stocks.
    -Use 5% stop loss and 10% take profit
    -Stocks pool are: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'IBM']
    -Allow multiple stocks to be held at the same time
    -Limit total invested ratio to 60% (keep at least 40% cash)
    -Wait at least 5 days before buying another stock
    -Only hold 3 stocks at the same time
    -Use 5% stop loss and 10% take profit
    """
    
    def initialize(self):
        """Initialize the strategy with data and indicators"""
        # Define parameters
        if not self.param:
            self.param = {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'IBM'],
                'sma_short_period': 10,  # Short SMA period
                'sma_long_period': 50,   # Long SMA period
                'stop_loss_percent': 5.0,  # Stop loss percentage
                'take_profit_percent': 10.0,  # Take profit percentage
                'max_positions': 3,      # Maximum number of positions to hold
                'max_investment_ratio': 0.6,  # Maximum percentage of portfolio to invest
                'min_days_between_buys': 5,  # Minimum days between buy signals
                'bb_period': 20,  # Bollinger Bands period
                'bb_std_dev': 2,  # Bollinger Bands standard deviation
            }
        # Track last buy date
        self.last_buy_date = None
        
        # Load price data for our symbols
        price_data = get_price(self.param['symbols'], data_source='US_stock', items=['close', 'high', 'low', 'open'])
        for item in price_data.keys():
            self.data[item] = price_data[item]
        
        # Calculate technical indicators and store in self.data
        self.data['sma_short'] = sma(self.data['close'], period=self.param['sma_short_period'])
        self.data['sma_long'] = sma(self.data['close'], period=self.param['sma_long_period'])
        
        # Calculate Bollinger Bands for ranking
        self.data['bb_upper'] = bb_upper(self.data['close'], period=self.param['bb_period'], std_dev=self.param['bb_std_dev'])
        self.data['bb_middle'] = bb_middle(self.data['close'], period=self.param['bb_period'], std_dev=self.param['bb_std_dev'])
        self.data['bb_lower'] = bb_lower(self.data['close'], period=self.param['bb_period'], std_dev=self.param['bb_std_dev'])
        
        # Calculate BB width (volatility measure)
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
        self.data['screen1'] = self.data['close'] > self.data['sma_long']
        self.data['screen2'] = self.data['sma_short'] > self.data['sma_long']
        self.data['qualified'] = self.data['screen1'] & self.data['screen2']
    
    def step_forward(self, data_handler):
        """
        Make trading decisions based on the latest data
        
        Args:
            data_handler: DataHandler instance containing the latest market data
        """
        # Get current open positions
        current_positions = self.portfolio.live_trades()
        
        # Check if we've reached maximum number of positions
        if len(current_positions) >= self.param['max_positions']:
            return
            
        # Check if we've reached maximum investment ratio
        if self.portfolio.invest_ratio() > self.param['max_investment_ratio']:
            return
            
        # Check if minimum days between buys has passed
        if self.days_since_last_trade() is not None and self.days_since_last_trade() < self.param['min_days_between_buys']:
            return
        
        # Get qualified symbols using DataHandler's screen method
        qualified_symbols = data_handler.screen(['qualified'])
        
        if not qualified_symbols:
            return
            
        # Find the symbol with the highest BB width among qualified symbols
        best_symbol = data_handler.highest('bb_width', tickers=qualified_symbols)[0]
        
        if not best_symbol:
            return
        
        # Calculate position size (invest 15% of portfolio per position)
        position_value = self.portfolio.cash * 0.15
        
        # Buy the best stock with stop loss and take profit
        self.buy(
            symbol=best_symbol,
            value=position_value,
            stop_loss_percent=self.param['stop_loss_percent'],
            take_profit_percent=self.param['take_profit_percent']
        )
        
        # Update last buy date
        self.last_buy_date = self.current_date

if __name__ == "__main__":
    # Create and run the strategy
    strategy = CustomStrategy()

    # If you get warning about mismatch index, 
    # it doesn't necessary means there is issue with data, 
    # for example, daily price data and quartely fundamental data.

    strategy.run(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31),
        # data_length=2  # Need at least 2 data points to compare current and previous
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

