from stratcraft import Strategy, Direction, DataHandler
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from decorators import broadcast, rolling, available, grouping
from util import access_case_insensitive
from data_loader import get_price, get_stock_list, load_path, get_fundamental, get_group
from metrics import Metrics
import ta

# Load dynamic paths
load_path('paths.json')


# Example: load single US stock daily price data, with default items ['open', 'high', 'low', 'close', 'volume']
price=get_price(['AAPL'], data_source='US_stock')

# Parameters:
# -----------
# tickers : str or list or None
#     Ticker symbol(s) to retrieve data for. If None, all available tickers will be loaded.
# data_source : Literal['US_stock', 'US_index', 'FX_hourly']
#     Data source to use for price data, US_stock for US stock price data, US_index for US index price data, FX_hourly for FX hourly price data
# items : str or list, optional
#     Price items to retrieve ('open', 'high', 'low', 'close', 'volume', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with price items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns


# Example: load only close data
price=get_price(['AAPL'], data_source='US_stock', items=['close'])
# Parameters:
# -----------
# tickers : str or list or None
#     Ticker symbol(s) to retrieve data for. If None, all available tickers will be loaded.
# data_source : Literal['US_stock', 'US_index', 'FX_hourly']
#     Data source to use for price data, US_stock for US stock price data, US_index for US index price data, FX_hourly for FX hourly price data
# items : str or list, optional
#     Price items to retrieve ('open', 'high', 'low', 'close', 'volume', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with price items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns


# Example: load price data with specific date range
price=get_price(['AAPL'], data_source='US_stock', start_date=datetime(2022, 1, 1), end_date=datetime(2022, 12, 31))
# Parameters:
# -----------
# tickers : str or list or None
#     Ticker symbol(s) to retrieve data for. If None, all available tickers will be loaded.
# data_source : Literal['US_stock', 'US_index', 'FX_hourly']
#     Data source to use for price data, US_stock for US stock price data, US_index for US index price data, FX_hourly for FX hourly price data
# items : str or list, optional
#     Price items to retrieve ('open', 'high', 'low', 'close', 'volume', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with price items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns


# Example: load multiple US stock data
price=get_price(['AAPL', 'MSFT'], data_source='US_stock')
# Parameters:
# -----------
# tickers : str or list or None
#     Ticker symbol(s) to retrieve data for. If None, all available tickers will be loaded.
# data_source : Literal['US_stock', 'US_index', 'FX_hourly']
#     Data source to use for price data, US_stock for US stock price data, US_index for US index price data, FX_hourly for FX hourly price data
# items : str or list, optional
#     Price items to retrieve ('open', 'high', 'low', 'close', 'volume', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with price items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns


# Example: load US stock fundamental data
fund=get_fundamental(['AAPL'], items=['is_eps','is_ebitda', 'fillingDate'])
# Parameters:
# -----------
# tickers : str or list
#     Ticker symbol(s) to retrieve data for
# items : str or list, optional
#     Fundamental items to retrieve (e.g., 'revenue', 'eps', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with fundamental items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns
# Note: There are no ratio data in fundamental data like PE, PB, PS, you need to calculate it yourself.


# Example: load FX hourly price data
price=get_price(['EURUSD'], data_source='FX_hourly')
# Parameters:
# -----------
# tickers : str or list or None
#     Ticker symbol(s) to retrieve data for. If None, all available tickers will be loaded.
# data_source : Literal['US_stock', 'US_index', 'FX_hourly']
#     Data source to use for price data, US_stock for US stock price data, US_index for US index price data, FX_hourly for FX hourly price data
# items : str or list, optional
#     Price items to retrieve ('open', 'high', 'low', 'close', 'volume', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with price items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns

# Example: load S&P 500 index price data
price=get_price(['sp_500_index'], data_source='US_index')
# Parameters:
# -----------
# tickers : str or list or None
#     Ticker symbol(s) to retrieve data for. If None, all available tickers will be loaded.
# data_source : Literal['US_stock', 'US_index', 'FX_hourly']
#     Data source to use for price data, US_stock for US stock price data, US_index for US index price data, FX_hourly for FX hourly price data
# items : str or list, optional
#     Price items to retrieve ('open', 'high', 'low', 'close', 'volume', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with price items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns



# Example: get list of symbols in S&P 500
stock_list=get_stock_list(component_of='sp500')
# Parameters:
# -----------
# tickers : str or list or None
#     Ticker symbol(s) to retrieve data for. If None, all available tickers will be loaded.
# data_source : Literal['US_stock', 'US_index', 'FX_hourly']
#     Data source to use for price data, US_stock for US stock price data, US_index for US index price data, FX_hourly for FX hourly price data
# items : str or list, optional
#     Price items to retrieve ('open', 'high', 'low', 'close', 'volume', etc.)
#     If None, all available items will be loaded
# start_date : str, optional
#     Start date in 'YYYY-MM-DD' format
# end_date : str, optional
#     End date in 'YYYY-MM-DD' format
# Returns:
# --------
# dict
#     Dictionary with price items as keys and DataFrames as values
#     Each DataFrame has dates as index and tickers as columns


# Example: load price data for sp500
stock_list=get_stock_list(component_of='sp500')
price=get_price(stock_list, data_source='US_stock')

# Example: get of symbols of each sector, returns a dict with key as sector name and value as list of symbols
sector=get_group('sector', market='US')  # {'Technology': ['AAPL', 'MSFT',...], 'Consumer Cyclical': ['AMZN', 'TSLA',...], ...}
# Get a dictionary of tickers grouped by industry or sector.
# Parameters:
# group_by : str
#     Grouping criteria ('industry' or 'sector')
# Returns:
# dict
#     Dictionary with industry or sector names as keys and lists of tickers as values

# Example: get of symbols of each industry, returns a dict with key as industry name and value as list of symbols
industry=get_group('industry', market='US')  # {'Consumer Electronics': ['AAPL', 'SONO',...], 'Software - Infrastructure': ['MSFT', 'ADBE',...], ...}
# Get a dictionary of tickers grouped by industry or sector.
# Parameters:
# group_by : str
#     Grouping criteria ('industry' or 'sector')
# Returns:
# dict
#     Dictionary with industry or sector names as keys and lists of tickers as values


# Example: get price data for Technology sector
sector=get_group('sector', market='US')
tech_price=get_price(sector['Technology'], data_source='US_stock')

# get price data for Consumer Electronics industry
industry=get_group('industry', market='US')
consumer_electronics_price=get_price(industry['Consumer Electronics'], data_source='US_stock')

# calculate a indicator for single stock from daily price data using rolling decorator
@rolling(window=14)
def SMA14(price):
    return price.mean()

price=get_price(['AAPL'], data_source='US_stock')
sma14=SMA14(price['close'])


# Example: calculate a indicator for multiple stocks from daily price data using broadcast decorator
@broadcast
def RSI(price):
    return ta.momentum.RSIIndicator(price).rsi()

price=get_price(['AAPL', 'MSFT'], data_source='US_stock')
rsi=RSI(price['close'])

# Example: calculate custom indicator for multiple stocks from daily price data using broadcast and rolling decorator
@broadcast
@rolling(window=14)
def custom_indicator(price):
    return price.max() - price.min()

price=get_price(['AAPL', 'MSFT'], data_source='US_stock')
custom_indicator=custom_indicator(price['close'])

# Example: Align fundamental data with price data using available decorator, applicable scenario: calculate PE ratio
price=get_price('AAPL', data_source='US_stock')['close']['AAPL']
fundamental=get_fundamental('AAPL', items=['is_eps','is_ebitda', 'fillingDate'])
available_date=fundamental['fillingDate']['AAPL']
eps=fundamental['is_eps']['AAPL']
looping_dates=price.index
@available(looping_dates, length=1)
def get_daily_eps(eps):
    return eps.iloc[-1]
daily_eps=get_daily_eps(eps, available_date=available_date)
pe=daily_eps/price
    

# Example: calculate revenue growth using available decorator
price=get_price(['AAPL'], data_source='US_stock')['close']['AAPL']
fundamental=get_fundamental(['AAPL'], items=['is_revenue', 'fillingDate'])
available_date=fundamental['fillingDate']['AAPL']
looping_dates=price.index   
revenue=fundamental['is_revenue']['AAPL']
@available(looping_dates, length=2)
def calculate_revenue_growth(revenue):
    return (revenue.iloc[-1] - revenue.iloc[-2]) / abs(revenue.iloc[-2])
revenue_growth=calculate_revenue_growth(revenue, available_date=available_date)


# Example: calculate revenue growth of multiple stocks using broadcast and available decorator
price=get_price(['AAPL', 'MSFT'], data_source='US_stock')['close']
fundamental=get_fundamental(['AAPL', 'MSFT'], items=['is_revenue', 'fillingDate'])
@broadcast
@available(price.index, length=2)
def calculate_revenue_growth(revenue):
    return (revenue.iloc[-1] - revenue.iloc[-2]) / abs(revenue.iloc[-2])
revenue_growth=calculate_revenue_growth(fundamental['is_revenue'], available_date=fundamental['fillingDate'])


# Example: calculate sector mean return for sp500 using grouping decorator
price=get_price(get_stock_list(component_of='sp500'), data_source='US_stock')
sector=get_group('sector', market='US')
@grouping(groups=sector)
def sector_mean_return(price):
    return price.pct_change().mean(axis=1)
sector_mean_return=sector_mean_return(price['close'])





# Example: Compare individual stock PE ratio relative to its sector PE ratio using broadcast decorator with groups argument
sp500_list=get_stock_list(component_of='sp500')
price=get_price(sp500_list, data_source='US_stock',items=['close'])
fundamental=get_fundamental(sp500_list, items=['is_eps', 'fillingDate'])
groups = get_group('sector', market='US')
eps=fundamental['is_eps']
fillingDate=fundamental['fillingDate']
close=price['close']

@broadcast
@available(looping_dates=close.index, length=1)
def daily_aligned_eps(eps):
    return eps[0]
daily_eps=daily_aligned_eps(eps, available_date=fillingDate)

daily_eps=daily_eps.replace(0, np.nan) # replace 0 with nan to avoid inf
pe=close/daily_eps

@grouping(groups=groups)
def group_pe(pe):
    return pe.median(axis=1)  # use median instead of mean to avoid outlier
sector_pe=group_pe(pe)

@broadcast(groups=groups)
def relative_PE(PE_individual, PE_sectors):
    return PE_individual / PE_sectors

rel_pe = relative_PE(pe, sector_pe)  # pe and sector_pe has different columns, but sector_pe will be broadcasted to pe because we have groups argument in broadcast decorator



# Example: Buy a symbol with fixed share size in step_forward() implementation of Strategy subclass
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        if some_condition:
            self.buy(symbol='AAPL', quantity=100)

# Example: Buy a symbol with fixed dollar value in step_forward() implementation of Strategy subclass
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        if some_condition:
            self.buy(symbol='AAPL', value=1000)

# Example: Buy a symbol with take profit and stop loss in step_forward() implementation of Strategy subclass
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        if some_condition:
            self.buy(symbol='AAPL', quantity=100,
            stop_loss_percent=5,
            take_profit_percent=10)


# Example: Buy a symbol with trailing stop loss in step_forward() implementation of Strategy subclass
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        if some_condition:
            trade = self.buy(symbol='AAPL', quantity=100)
            if trade:
                trailing_stop = TrailingStopLoss(
                    price=trade.entry_price - 1.0,
                    distance=1.0,
                    threshold=trade.entry_price + 1.0,
                    direction=Direction.LONG
                )
                trade.stop_loss = trailing_stop  # Note: use dot notation to set stop loss, not add_stop_loss()


# Example: Access data with indexing of DataHandler
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        current_close = data_handler['close']
        previous_close = data_handler[('close', -2)]  # in run(), set data_length=2, since we want to access the previous two data points


# Example: prepare, calculate market data and indicators in initialize() implementation of Strategy subclass
class MyStrategy(Strategy):
    def initialize(self):
        self.param = {
            'symbol': 'AAPL',
            'sma_period': 20,
        }
        self.data = get_price(self.param['symbol'], data_source='US_stock')
        self.data['sma'] = sma(self.data['close'], period=self.param['sma_period'])

# Example: Run backtest
strategy = MyStrategy()
strategy.run(
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31),
    data_length=2 # If inside step_forward(), we have data_handler[('some_field', -2)]
)

# Example: Get raw data of trade history of backtest
trade_history_df = strategy.portfolio.trade_history()

# Example: Get profit and loss history of backtest
pl_history_df = strategy.portfolio.pl_history()  # columns: cash and equity


# Example: Get result and metrics of backtest
trade_history_df = strategy.portfolio.trade_history()
pl_history_df = strategy.portfolio.pl_history()
if not pl_history_df.empty and not trade_history_df.empty:
    metrics = Metrics(trade_history_df, pl_history_df)
    result = metrics.metrics(concise=True)
    Metrics.pretty_print(result)


# Example: Get cash balance of portfolio
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        cash = self.portfolio.cash

# Example: Get equity of portfolio
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        equity = self.portfolio.equity

# Example: Get live trades (open positions)
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        live_trades = self.portfolio.live_trades()

# Example: Get total cost of open positions
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        total_cost = self.portfolio.total_cost()

# Example: Get the current market value of the portfolio.
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        market_value = self.portfolio.current_market_value()

# Example: Get the ratio of the portfolio's investment value to the initial capital.
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        invest_ratio = self.portfolio.invest_ratio()

# Example: Calculate the total cost of open positions, or for a specific symbol
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        total_cost = self.portfolio.cost()
        aapl_cost = self.portfolio.cost(symbol='AAPL')

# Example: Get the date of the last trade in the portfolio. (Get latest trade date)
class MyStrategy(Strategy):
    # some other part of implementation
    # ...
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        last_trade_date = self.portfolio.date_last_trade()
    
# Example: Screen symbols based on multiple criteria pre-defined in initialize() implementation of Strategy subclass
class MyStrategy(Strategy):
    def initialize(self):
        # some other part of implementation
        # ...
        self.data['criteria_1'] = calculate_criteria_1()
        self.data['criteria_2'] = calculate_criteria_2()
    
    def step_forward(self, data_handler):
        # some logic to generate buy signal
        # ...
        screened_symbols = data_handler.screen(['criteria_1', 'criteria_2'])
        

# Example: Choose the highest n symbols based on a specific field
class MyStrategy(Strategy):
    def initialize(self):
        # some other part of implementation
        # ...
        self.data['compare_field'] = calculate_field()
        self.data['criteria_1'] = calculate_criteria_1()
        self.data['criteria_2'] = calculate_criteria_2()
    
    def step_forward(self, data_handler):
        highest_symbols = data_handler.highest('compare_field', n=5)  # Choose from all symbols
        screened_symbols = data_handler.screen(['criteria_1', 'criteria_2'])
        screened_highest_symbols = data_handler.highest('compare_field', n=5, tickers=screened_symbols)  # Choose from screened symbols by assigning tickers argument


# Example: Choose the lowest n symbols based on a specific field
class MyStrategy(Strategy):
    def initialize(self):
        # some other part of implementation
        # ...
        self.data['compare_field'] = calculate_field()
        self.data['criteria_1'] = calculate_criteria_1()
        self.data['criteria_2'] = calculate_criteria_2()
    
    def step_forward(self, data_handler):
        lowest_symbols = data_handler.lowest('compare_field', n=5)  # Choose from all symbols
        screened_symbols = data_handler.screen(['criteria_1', 'criteria_2'])
        screened_lowest_symbols = data_handler.lowest('compare_field', n=5, tickers=screened_symbols)  # Choose from screened symbols by assigning tickers argument
