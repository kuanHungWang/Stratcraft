from stratcraft import Strategy, Direction, DataHandler
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from decorators import broadcast, rolling, available
from indicators import rsi, bb_lower, bb_middle, bb_upper
from util import access_case_insensitive
from metrics import Metrics
from data_loader import get_price, get_fundamental
class CustomStrategy(Strategy):
    """
    Multiple stock with technical and fundamental indicators
    

    - stock pool: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'IBM']
    - Uses fundamental data to screen stocks (quarterly EBITDA growth >0)
    - Amoung qualified stocks, Buys the stock with highest RSI(14)
    - Invest 50% of available cash each time.
    - Wait at least 5 days before buying another stock, 
    - At least keep 50% of initial cash
    - Uses 5% stop loss and 10% take profit for risk management
    """
    
    def initialize(self):
        """Initialize the strategy with data and indicators"""
        # Define parameters
        if not self.param:
            self.param = {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'IBM'],
                'rsi_period': 14,
                'stop_loss_percent': 5.0,
                'take_profit_percent': 10.0,
                'days_since_last_trade': 5,
                'cash_reserve_percentage': 50.0,
                'position_size_percentage': 50.0
                
            }
        
        # Load price data for our symbols
        
        price_data = get_price(self.param['symbols'], data_source='US_stock', items=['close', 'high', 'low', 'open'])
        
        # Store the price data in self.data - handle case sensitivity
        for item in ['close', 'high', 'low', 'open']:
            self.data[item] = price_data[item]
        
        # Calculate RSI indicator and store in self.data
        self.data['rsi'] = rsi(self.data['close'], period=self.param['rsi_period'])
        
        # Load fundamental data - EBITDA and Revenue
        fundamental_data = get_fundamental(self.param['symbols'], ['is_ebitda', 'is_revenue', 'fillingDate'])
        # Important: Do not store the fundamental data in self.data as it has different index, may case warning and error.
        # Just use it as intermediate variable to calculate indicators, or use broadcast decorator to align index to daily price data.
        @broadcast()
        @available(self.data['close'].index, length=2)
        def calculate_revenue_growth(ebitda):
            return (ebitda.iloc[-1] - ebitda.iloc[-2]) / abs(ebitda.iloc[-2])

        # Calculate Revenue growth (quarter-over-quarter)
        self.data['revenue_growth'] = calculate_revenue_growth(fundamental_data['is_revenue'], available_date=fundamental_data['fillingDate'])
        self.data['screen_ebitda'] = self.data['revenue_growth'] > 0



    def step_forward(self, data_handler: DataHandler):
        """
        Make trading decisions based on the latest data
        
        Args:
            data_handler: DataHandler instance containing the latest market data
        """
        # Check if we already have an open position
        qualified = data_handler.screen(['screen_ebitda'])
        buy_list = data_handler.highest('rsi', tickers=qualified)
        
        # Get current open positions
        current_positions = self.portfolio.live_trades()
        
        # Check if we've reached maximum number of positions (only allow one position at a time)
        if len(current_positions) > 0:
            return
            
        # Only buy if at least 5 days have passed since last trade
        if self.days_since_last_trade() is not None and self.days_since_last_trade() < self.param['days_since_last_trade']:
            return
            
        # Check if we have enough cash (keep at least 50% of initial cash)
        if self.portfolio.cash < (self.portfolio.initial_capital * (self.param['cash_reserve_percentage'] / 100.0)):
            return
            
        # Check if we have qualified stocks to buy
        if not buy_list or len(buy_list) == 0:
            return
            
        # Select the highest ranked stock (by RSI)
        best_symbol = buy_list[0]
        
        # Calculate position size (invest 50% of available cash)
        position_value = self.portfolio.cash * (self.param['position_size_percentage'] / 100.0)
        
        # Buy the best stock with stop loss and take profit
        self.buy(
            symbol=best_symbol,
            value=position_value,
            stop_loss_percent=self.param['stop_loss_percent'],
            take_profit_percent=self.param['take_profit_percent']
        )
        




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