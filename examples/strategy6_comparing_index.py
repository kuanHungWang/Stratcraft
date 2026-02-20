from stratcraft import Strategy, Direction, DataHandler
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from decorators import broadcast, rolling, available
from indicators import rsi, bb_lower, bb_middle, bb_upper
from util import access_case_insensitive
from data_loader import get_price, get_stock_list
from metrics import Metrics

class CustomStrategy(Strategy):
    """
    Compare perfomance of individual stocks with index's performance

    1. Asset pool: S&P 500 stocks
    2. Screen with stock price > SMA(20), SMA(20) > SMA(60)
    3. Choose lowest 10 stocks whose 1 week return relative to index is lowest.
    4. Each candidate invest 10% of available cash.
    5. Buy only one stock each day
    6. Wait at least 5 days before buy another stock
    7. Use stop loss and take profit for sell.
    """

    def initialize(self):
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        sp500_list=get_stock_list(component_of='sp500')
        price=get_price(sp500_list, data_source='US_stock', start_date=start_date, end_date=end_date)
        for item in ['close', 'high', 'low', 'open']:
            self.data[item] = price[item]

        index_price=get_price(['sp_500_index'], data_source='US_index', items=['close'], start_date=start_date, end_date=end_date)['close']

        index_price = index_price.reindex(price['close'].index)
        self.data['index_close'] = index_price['sp_500_index']

        @broadcast
        @rolling(window=20)
        def SMA20(close):
            return close.mean()

        @broadcast
        @rolling(window=60)
        def SMA60(close):
            return close.mean()

        sma20 = SMA20(self.data['close'])
        sma60 = SMA60(self.data['close'])
        self.data['screen'] = (self.data['close'] > sma20) & (sma20 > sma60)

        @broadcast
        @rolling(window=5)
        def index_relative_return(stock_close, index_close):
            stock_return=stock_close.iloc[-1]/stock_close.iloc[0]-1
            index_return=index_close.iloc[-1]/index_close.iloc[0]-1
            return stock_return - index_return

        self.data['relative_return'] = index_relative_return(self.data['close'], self.data['index_close'])

        if not self.param:
            self.param = {
                'stop_loss_percent': 5.0,
                'take_profit_percent': 10.0
            }

    def step_forward(self, data_handler: DataHandler):
        """
        1. Screen with stock price > SMA(20), SMA(20) > SMA(60)
        2. Choose lowest 10 stocks whose 1 week return relative to index is lowest.
        3. Each candidate invest 10% of available cash.
        4. Buy only one stock each day
        5. Wait at least 5 days before buy another stock
        6. Use stop loss and take profit for sell.
        """
        # 1. Screen stocks
        qualified = data_handler.screen(['screen'])
        if not qualified or len(qualified) == 0:
            return

        # 2. Sort by 1-week relative return (ascending: lowest first)
        candidates = data_handler.lowest('relative_return', n=10, tickers=qualified)
        
        # 3. Only buy one stock per day, and wait at least 5 days before buying any stock
        if self.days_since_last_trade() is not None and self.days_since_last_trade() < 5:
            return
        current_positions = self.portfolio.live_trades()
        held_symbols = set([t.symbol for t in current_positions])
        # Find candidate that can be bought (not held)
        non_holding_candidates = [sym for sym in candidates if sym not in held_symbols]
        if not non_holding_candidates:
            return
        buy_symbol = non_holding_candidates[0]
        
        # 4. Invest 10% of available cash
        position_value = self.portfolio.cash * 0.1
        if position_value < 1:  # skip if too little cash
            return
        # 5. Buy with stop loss and take profit (use default 5%/10% if not in param)
        stop_loss = self.param.get('stop_loss_percent', 5.0)
        take_profit = self.param.get('take_profit_percent', 10.0)
        self.buy(
            symbol=buy_symbol,
            value=position_value,
            stop_loss_percent=stop_loss,
            take_profit_percent=take_profit
        )

if __name__ == '__main__':
    strategy = CustomStrategy()

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
