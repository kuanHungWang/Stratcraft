from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from optparse import Values
from typing import List, Dict, Callable, Optional, TypedDict, Literal, TypeAlias, Union, Any, Tuple, Iterable
import pandas as pd
from datetime import datetime
import numpy as np
from .util import access_case_insensitive, valid_date_range, valid_symbol

class Direction(Enum):
    LONG = 1
    SHORT = -1

class Action(Enum):
    OPEN=1
    CLOSE=2


@dataclass
class TakeProfit:
    price: float
    # ratio: Optional[float] = 1.0
    # Need validation for constructor input:
    # only one of price, percent, value should be provided


@dataclass
class StopLoss:
    price: float
    # ratio: Optional[float] = 1.0
    # Need validation for constructor input:


@dataclass
class TrailingStopLoss(StopLoss):
    """
    A trailing stop loss that adjusts the stop price based on the distance from the entry price.
    Note: Can only be manually added into the trade. Cannot added by add_stop_loss() method of Trade.
    You must manually reset the stoploss level by calling reset_price() method.
    Args:
        distance (float): The distance from the entry price
        threshold (float): The threshold price
        direction (Direction): The direction of the trade
    """
    
    distance: float
    threshold: float
    direction: Direction


    def reset_price(self, price: float) -> None:
        if self.direction == Direction.LONG:
            if price > self.threshold:
                new_price = price - self.distance
                self.price = max(new_price, self.price)
        else:
            if price < self.threshold:
                new_price = price + self.distance
                self.price = min(new_price, self.price)
        


@dataclass
class Trade:
    direction: Direction
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    market_price: float
    action: Action = Action.OPEN
    stop_loss: Optional[StopLoss] = None
    take_profit: Optional[TakeProfit] = None
    closing_trade: Optional[Trade] = None
    _closed: bool = False
    _win: bool = False

    def is_opposite(self, trade: Trade) -> bool:
        """
        Check if a trade is opposite to the given trade.
        
        Args:
            trade (Trade): The trade to compare with
            
        Returns:
            bool: True if trades are for the same symbol but opposite directions
        """
        return (self.symbol == trade.symbol and 
                self.direction != trade.direction)

    def close(self, trade: Trade):
        """
        Close this trade with another trade, updating closed status, win/loss, and reference.
        
        Args:
            trade (Trade): The closing trade that closes this position
            
        Raises:
            ValueError: If closing trade is not opposite to this trade
        """
        if not self.is_opposite(trade):
            raise ValueError("Closing trade must be opposite to the original trade")
        self._closed = True
        self.closing_trade = trade
        self._win = self.win()

    def add_stop_loss(self, price: float|None=None, percent: float|None=None, value: float|None=None):
        """
        Add stop loss to the trade. Only one of price, percent, or value should be provided.
        
        Args:
            price (float, optional): Fixed stop loss price
            percent (float, optional): Stop loss percentage from entry price
            value (float, optional): Stop loss value difference from entry price
            ratio (float, optional): Ratio for partial stop loss. Defaults to 1.0
            
        Raises:
            ValueError: If more than one of price, percent, value is provided
        """
        if sum(x is not None for x in [price, percent, value]) != 1:
            raise ValueError("Exactly one of price, percent, or value must be provided")
        
        if percent is not None:
            price = (self.entry_price * (1 - percent) if self.direction == Direction.LONG 
                    else self.entry_price * (1 + percent))
        elif value is not None:
            price = (self.entry_price - value if self.direction == Direction.LONG 
                    else self.entry_price + value)
            
        self.stop_loss = StopLoss(price=price)

    def add_take_profit(self, price: float|None=None, percent: float|None=None, value: float|None=None):
        """
        Add take profit to the trade. Only one of price, percent, or value should be provided.
        
        Args:
            price (float, optional): Fixed take profit price
            percent (float, optional): Take profit percentage from entry price
            value (float, optional): Take profit value difference from entry price
            ratio (float, optional): Ratio for partial take profit. Defaults to 1.0
            
        Raises:
            ValueError: If more than one of price, percent, value is provided
        """
        if sum(x is not None for x in [price, percent, value]) != 1:
            raise ValueError("Exactly one of price, percent, or value must be provided")
        
        if percent is not None:
            price = (self.entry_price * (1 + percent) if self.direction == Direction.LONG 
                    else self.entry_price * (1 - percent))
        elif value is not None:
            price = (self.entry_price + value if self.direction == Direction.LONG 
                    else self.entry_price - value)
            
        self.take_profit = TakeProfit(price=price)

    def stop_loss_triggered(self, high: float, low: float) -> bool:
        """
        Check if stop loss is triggered based on price range.
        
        Args:
            high (float): Highest price in the period
            low (float): Lowest price in the period
            
        Returns:
            bool: True if stop loss is triggered
        """
        if not self.stop_loss:
            return False
            
        if self.direction == Direction.LONG:
            return low <= self.stop_loss.price
        else:
            return high >= self.stop_loss.price

    def take_profit_triggered(self, high: float, low: float) -> bool:
        """
        Check if take profit is triggered based on price range.
        
        Args:
            high (float): Highest price in the period
            low (float): Lowest price in the period
            
        Returns:
            bool: True if take profit is triggered
        """
        if not self.take_profit:
            return False
            
        if self.direction == Direction.LONG:
            return high >= self.take_profit.price
        else:
            return low <= self.take_profit.price

    def is_closed(self) -> bool:
        """
        Returns True if the trade is closed (either by taking a profit or losing the position).
        """
        return self._closed

    def win(self) -> bool:
        """
        Returns True if the trade wins.
        """
        if not self._closed:
            return False
        if not self.closing_trade:
            print("Warning: trade is set to closed but no closing trade is set.")
            return False
        if self.direction == Direction.LONG:
            return self.closing_trade.entry_price > self.entry_price
        else:
            return self.closing_trade.entry_price < self.entry_price

    def pl(self) -> float:
        """Return the profit and loss of the trade."""
        # case of closed trade, use closing trade price to calculate pl
        if self._closed:
            if self.direction==Direction.LONG:
                return (self.closing_trade.entry_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - self.closing_trade.entry_price) * self.quantity
        # case of open trade, use market price to calculate pl
        else:
            if self.direction == Direction.LONG:
                return (self.market_price - self.entry_price) * self.quantity
            else:
                return (self.entry_price - self.market_price) * self.quantity

scalar: TypeAlias = Union[int, float, bool]
basic_structured_data: TypeAlias = pd.Series|pd.DataFrame



class MarketHandler:
    def __init__(self, market_data: Any):
        self._data: Dict[str, float|Dict[str, float]] = market_data
        self.current_date: Optional[datetime] = None


    def open_price(self, symbol:str)->float|Dict[str, float]:
        open_price = access_case_insensitive('open', self._data)
        # open_price = self._data['Open']
        if symbol == '':
            return open_price
        else:
            return open_price[symbol]

    def high_price(self, symbol:str)->float|Dict[str, float]:
        high_price = access_case_insensitive('high', self._data)
        # high_price = self._data['High']
        if symbol == '':
            return high_price
        else:
            return high_price[symbol]

    def low_price(self, symbol:str)->float|Dict[str, float]:
        low_price = access_case_insensitive('low', self._data)
        # low_price = self._data['Low']

        if symbol == '':
            return low_price
        else:
            return low_price[symbol]

    def close_price(self, symbol:str)->float|Dict[str, float]:
        close_price = access_case_insensitive('close', self._data)
        # close_price = self._data['Close']
        if symbol == '':
            return close_price
        else:
            return close_price[symbol]


    def update_market_data(self, market_data: Any) -> None:
        """
        Update market data with new data.
        
        Args:
            market_data (Any): New market data in the same format as initialization
        """
        self._data = market_data 
        # dict with for key: open, high, low, close
        # value is float if single symbol, dict if multiple symbols


    def buy_at_open(self, symbol: str, quantity: int|None=None, value: float|None=None) -> Trade:
        """
        Create a buy trade at market open price.
        
        Args:
            symbol (str): Symbol to trade
            quantity (int, optional): Quantity to buy
            value (float, optional): Dollar value to buy
            
        Returns:
            Trade: New buy trade
            
        Raises:
            ValueError: If neither or both quantity and value are provided
        """
        if sum(x is not None for x in [quantity, value]) != 1:
            raise ValueError("Exactly one of quantity or value must be provided")
        
        open_price = self.open_price(symbol)

        if value is not None:
            quantity = int(value / open_price)
            
        return Trade(
            direction=Direction.LONG,
            symbol=symbol,
            quantity=quantity,
            entry_price=open_price,
            entry_date=self.current_date,
            market_price=open_price
        )

    def sell_at_open(self, symbol: str, quantity: int|None=None, value: float|None=None) -> Trade:
        """
        Create a sell trade at market open price.
        
        Args:
            symbol (str): Symbol to trade
            quantity (int, optional): Quantity to sell
            value (float, optional): Dollar value to sell
            
        Returns:
            Trade: New sell trade
            
        Raises:
            ValueError: If neither or both quantity and value are provided
        """
        if sum(x is not None for x in [quantity, value]) != 1:
            raise ValueError("Exactly one of quantity or value must be provided")
        
        open_price = self.open_price(symbol)


        if value is not None:
            quantity = int(value / open_price)
            
        return Trade(
            direction=Direction.SHORT,
            symbol=symbol,
            quantity=quantity,
            entry_price=open_price,
            entry_date=self.current_date,
            market_price=open_price
        )

    def check_stop_loss(self, trade: Trade) -> Trade|None:
        """
        Check if trade's stop loss is triggered.
        
        Args:
            trade (Trade): Trade to check
            
        Returns:
            Trade|None: Trade if stop loss triggered, None otherwise
        """
        if trade.stop_loss_triggered(
            high=self.high_price(trade.symbol),
            low=self.low_price(trade.symbol)
        ):
            # Create a closing trade with opposite direction
            closing_trade = Trade(
                direction=Direction.SHORT if trade.direction == Direction.LONG else Direction.LONG,
                symbol=trade.symbol,
                quantity=trade.quantity,
                entry_price=self.close_price(trade.symbol),
                entry_date=self.current_date,
                market_price=self.close_price(trade.symbol),
                action=Action.CLOSE
            )
            return closing_trade
        return None

    def check_take_profit(self, trade: Trade) -> Trade|None:
        """
        Check if trade's take profit is triggered.
        
        Args:
            trade (Trade): Trade to check
            
        Returns:
            Trade|None: Trade if take profit triggered, None otherwise
        """
        if trade.take_profit_triggered(
            high=self.high_price(trade.symbol),
            low=self.low_price(trade.symbol)
        ):
            # Create a closing trade with opposite direction
            closing_trade = Trade(
                direction=Direction.SHORT if trade.direction == Direction.LONG else Direction.LONG,
                symbol=trade.symbol,
                quantity=trade.quantity,
                entry_price=self.close_price(trade.symbol),
                entry_date=self.current_date,
                market_price=self.close_price(trade.symbol),
                action=Action.CLOSE
            )
            return closing_trade
        return None

    def update_price(self, trades: List[Trade]) -> None:
        """
        Update market prices of trades using close prices.
        
        Args:
            trades (List[Trade]): List of trades to update
        """
        for trade in trades:
            trade.market_price = self.close_price(trade.symbol)

class Portfolio():
    def __init__(self, initial_capital: float):
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._cash_series: Optional[pd.Series] = None
        self._positions: List[Trade] = []
        self._equity_series: pd.Series|None = None  

    @property
    def cash(self) -> float:
        """
        Get the current cash balance of the portfolio.
        
        Returns:
            float: The current cash amount.
        """
        return self._cash

    @property
    def cash_series(self) -> pd.Series:
        """
        Get the historical cash series.
        
        Returns:
            pd.Series: The historical cash series.
        """
        return self._cash_series

    @property
    def equity(self) -> float:
        """
        Get the current equity value of the portfolio.
        
        Returns:
            float: Current equity value. Returns 0.0 if equity series is not initialized.
        """
        if self._equity_series is None or self._equity_series.empty:
            return self._cash
        return self._equity_series.iloc[-1]

    @property
    def equity_series(self) -> pd.Series:
        """
        Get the historical equity series.
        
        Returns:
            pd.Series: The historical equity series.
        """
        return self._equity_series
        
    @equity_series.setter
    def equity_series(self, value: pd.Series) -> None:
        """
        Set the historical equity series.
        
        Args:
            value (pd.Series): The historical equity series to set.
        """
        self._equity_series = value

    def add_position(self, position: Trade):
        """
        Add a new trade position to the portfolio.
        
        Args:
            position (Trade): The trade position to be added to the portfolio.
        """
        self._positions.append(position)

    def live_trades(self) -> List[Trade]:
        """
        Get a list of all live (open) trades in the portfolio.
        
        Returns:
            List[Trade]: A list of all live trades in the portfolio.
        """
        return [t for t in self._positions if not t._closed]

    def check_opposite_position(self, trades: List[Trade])->List[Trade]:
        """
        Check if there are opposite positions for the input trades in the portfolio.
        For each input trade, if an opposite position exists, close that position
        and mark the input trade as used by setting it to None.
        
        Args:
            trades (List[Trade]): List of trades to check for opposite positions.
            
        Returns:
            List[Trade]: Same length as input list, with used trades replaced by None.
        """
        result = trades.copy()
        for i, trade in enumerate(trades):
            for pos in self._positions:
                if (pos.symbol == trade.symbol and 
                    pos.direction != trade.direction and 
                    not pos._closed):
                    # Close the existing position with this trade
                    pos.close(trade)
                    # Mark this trade as used
                    result[i] = None
                    break
        return result

    @property
    def initial_capital(self) -> float:
        """
        Get the initial capital of the portfolio.
        
        Returns:
            float: The initial capital amount.
        """
        return self._initial_capital

    def update_cash(self, date: datetime) -> None:
        """
        Update the portfolio's cash balance based on trades entered or closed on the given date.
        Creates or updates the cash series with the new balance.
        
        Args:
            date (datetime): The date to check for trade entries and exits and update cash.
        """
        if self._cash_series is None:
            self._cash_series = pd.Series([self._cash], index=[date])
            return

        # Check for trades entered on this date
        for trade in self._positions:
            if trade.entry_date == date and not trade._closed:
                # Subtract entry cost from cash
                self._cash -= trade.entry_price * trade.quantity

        # Check for trades closed on this date
        for trade in self._positions:
            if trade._closed and trade.closing_trade and trade.closing_trade.entry_date == date:
                # Add closing price to cash
                self._cash += trade.closing_trade.entry_price * trade.quantity

        # Append new cash value to series
        self._cash_series[date] = self._cash

    def current_market_value(self) -> float:
        """
        Get the current market value of the portfolio.
        
        Returns:
            float: The current market value of the portfolio.
        """
        total_market_value = 0
        for trade in self._positions:
            if not trade._closed:
                market_value = trade.market_price * trade.quantity * trade.direction.value
                total_market_value += market_value
        return total_market_value

    def invest_ratio(self) -> float:
        """
        Get the ratio of the portfolio's investment value to the initial capital.
        
        Returns:
            float: The investment ratio.
        """
        return 1 - self.cash / self.equity

    def update_equity(self,date: datetime) -> None:
        """
        Update the portfolio's equity series based on the latest cash series and position market values.
        Should be called after update_cash() to ensure cash values are current.
        Creates a new equity series if none exists.
        
        Equity is calculated as: cash + sum(position market values)
        """
        if self._equity_series is None:
            self._equity_series = pd.Series([self._cash], index=[date])
            return
        
        # Add market value of open positions
        # total_market_value = 0
        self._equity_series[date] = self._cash + self.current_market_value()

    def current_equity(self) -> float:
        """
        Get the current equity value of the portfolio.
        
        Returns:
            float: Current equity value. Returns 0.0 if equity series is not initialized.
        """
        return self.equity

    def cost(self, symbol: str|None=None) -> float:
        """
        Calculate the total cost of open positions, either for a specific symbol
        or for the entire portfolio.
        
        Args:
            symbol (str|None, optional): Symbol to calculate cost for. 
                If None, calculates cost for all positions. Defaults to None.
        
        Returns:
            float: Total cost of the specified positions.
        """
        total_cost = 0.0
        for trade in self.live_trades():
            if not trade._closed:  # Only count open positions
                if symbol is None or trade.symbol == symbol:
                    total_cost += abs(trade.entry_price * trade.quantity)
        return total_cost

    def pl_history(self) -> pd.DataFrame:
        """
        Get a DataFrame of profit/loss history.
        
        Returns:
            pd.DataFrame: DataFrame containing profit/loss history with columns for
                         date, symbol, quantity, entry_price, closing_price, and p/l.
        """
        return pd.DataFrame({'cash': self.cash_series, 'equity': self.equity_series})

    def trade_history(self) -> pd.DataFrame:
        """
        Convert trades in self._positions into a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing trade information with columns for
                          trade date, close date, p/l (if applicable), and all properties
                          of Trade class except closing_trade, stop_loss, and take_profit.
        """
        if not self._positions:
            return pd.DataFrame()
            
        trades_data = []
        for trade in self._positions:
            trade_dict = {
                'direction': trade.direction.value,
                'symbol': trade.symbol,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'entry_date': trade.entry_date,
                'action': trade.action.value,
                'closed': trade._closed,
                'win': trade._win,
            }
            
            # Add close date if trade is closed
            if trade._closed and trade.closing_trade:
                trade_dict['close_date'] = trade.closing_trade.entry_date
                trade_dict['p/l'] = trade.pl()
            else:
                trade_dict['close_date'] = None
                trade_dict['p/l'] = None
                
            trades_data.append(trade_dict)
            
        return pd.DataFrame(trades_data)
    
    
    def date_last_trade(self) -> datetime|None:
        """
        Get the date of the last trade in the portfolio.
        
        Returns:
            datetime|None: The date of the last trade, or None if no trades have been made.
        """
        return max((trade.entry_date for trade in self._positions), default=None) if self._positions else None

class DataHandler:
    def __init__(self, data: Dict[str, pd.DataFrame|pd.Series]):
        self.data = data
        close_df = access_case_insensitive('close', self.data)
        if isinstance(close_df, pd.DataFrame):
            self.tickers = close_df.columns
        else:
            self.tickers = [close_df.name]


    def __getitem__(self, key):
        """
        Provides a convenient way to access data.
        
        Usage:
        - handler['item'] or handler[('item',)] returns the latest data point of 'item'
        - handler[('item', n)] returns the nth data point of 'item' (negative indexing supported)
        
        If the result is a Series with a single value or a DataFrame with shape (1,1),
        it will be converted to a scalar (float, int, or bool).
        
        Args:
            key: str or tuple of (str, int)
                If str: the key of the item in self.data
                If tuple: (item, n) where item is the key and n is the index
                
        Returns:
            The data point(s) requested, converted to scalar if possible
        """
        if isinstance(key, str):
            # If key is just a string, return the latest data point
            item = key
            idx = -1
        elif isinstance(key, tuple) and len(key) == 2:
            # If key is a tuple of (item, idx), return the specified data point
            item, idx = key
        else:
            raise KeyError(f"Invalid key format: {key}. Use 'item' or ('item', idx)")
        try:
            data_item = access_case_insensitive(item, self.data)
        except KeyError:
            raise KeyError(f"Item '{item}' not found in data, check implementation of initialize()")

        
        # Get the data at the specified index
        if isinstance(data_item, pd.DataFrame):
            try:
                result = data_item.iloc[idx]
            except IndexError as e:
                raise IndexError(f"IndexError in DataHandler: {e}. This may be caused by insufficient data history. If you are using negative indices (e.g., -2, -3), ensure that the data_length argument in strategy.run() is set to at least the highest |n| needed in step_forward().")
            # If it's a Series with a single value, convert to scalar
            if len(result) == 1:
                return result.iloc[0]
            return result
        elif isinstance(data_item, pd.Series):
            try:
                return data_item.iloc[idx]
            except IndexError as e:
                raise IndexError(f"IndexError in DataHandler: {e}. This may be caused by insufficient data history. If you are using negative indices (e.g., -2, -3), ensure that the data_length argument in strategy.run() is set to at least the highest |n| needed in step_forward().")
        else:
            raise TypeError(f"Unexpected data type for '{item}': {type(data_item)}")

    def screen(self, masks: List[Union[Iterable, str]]) -> List[str]:
        """
        Screens symbols based on multiple criteria.
        
        Args:
            masks: List of strings or boolean iterables
                - For string type: must be one of keys of self.data, uses the latest data point
                - For Iterable type: must be boolean or int values
        
        Returns:
            List of symbols that pass all criteria in the mask
        """
        if not masks:
            return []
        
        # Get the latest data point for each mask
        processed_masks = []
        
        for mask in masks:
            if isinstance(mask, str):
                # If mask is a string, get the latest data point
                latest_data = self[mask]
                processed_masks.append(latest_data)
            elif isinstance(mask, list) or isinstance(mask, np.ndarray):
                # If mask is already an iterable, use it directly
                mask_series = pd.Series(mask, index=self.tickers)
                processed_masks.append(mask_series)
            elif isinstance(mask, pd.Series):
                processed_masks.append(mask)
            else:
                raise TypeError(f"Unexpected mask type: {type(mask)}, must be iterable or string")
        
        combined_mask = pd.Series(True, index=self.tickers)
        for mask in processed_masks:
            combined_mask = combined_mask & mask
        
        return self.tickers[combined_mask].tolist()
        


    def highest(self, item: str, n: int=1, tickers: List[str]=None) -> List[str]:
        """
        Returns tickers with the highest values of the specified item.
        
        Args:
            item: The key of the item in self.data
            n: The number of highest values to return (default: 1)
            tickers: List of tickers to consider (default: None for all tickers)
        
        Returns:
            List of tickers with the highest values
        """
        if item not in self.data:
            raise KeyError(f"Item '{item}' not found in data")
        
        # Get the latest data point
        latest_data = self[item]
        
        # Filter by tickers if specified
        if tickers:
            latest_data = latest_data[tickers]
        
        # Sort in descending order and get top n
        sorted_data = latest_data.sort_values(ascending=False)
        return sorted_data.index[:n].tolist()

    def lowest(self, item: str, n: int=1, tickers: List[str]=None) -> List[str]:
        """
        Returns tickers with the lowest values of the specified item.
        
        Args:
            item: The key of the item in self.data
            n: The number of lowest values to return (default: 1)
            tickers: List of tickers to consider (default: None for all tickers)
        
        Returns:
            List of tickers with the lowest values
        """
        
        # Get the latest data point
        latest_data = self[item]
        
        # Filter by tickers if specified
        if tickers:
            latest_data = latest_data[tickers]
        
        # Sort in ascending order and get top n
        sorted_data = latest_data.sort_values(ascending=True)
        return sorted_data.index[:n].tolist()

class Strategy(ABC):
    def __init__(self, 
                # profit_loss: ProfitLoss|None=None, 
                initial_capital: float=1_000_000,
                commission: float=0.0,
                bid_ask_spread: float=0.0,
                param: Dict[str, Any]=None,
                portfolio: Portfolio|None=None):

        self.portfolio = portfolio if portfolio is not None else Portfolio(initial_capital)
        self.market_handler = MarketHandler({})  # Initialize with empty market data
        # self.profit_loss = profit_loss if profit_loss is not None else ProfitLoss()
        # self.profit_loss.portfolio = self.portfolio
        # self.profit_loss.market_handler = self.market_handler
        self.current_date = None
        self.previous_date = None
        self.data = {}
        self.param = param
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the strategy before running.
        main user interface
        set data and indicators here
        """
        pass
    
    def _before_step(self) -> None:
        """Called before each step."""
        # like set  update data in market_handler and update current date of self.profit_loss.
        self._update_market_handler()
        # self.profit_loss.current_date = self.current_date


    def _after_step(self) -> None:
        """Called after each step."""
        # like check for stop loss and take profit, update market price, cash, equity of portfolio.
        # update all price of live trades in portfolio by self.market_handler.update_price(self.portfolio.live_trades())
        self._check_stop_loss()
        self._check_take_profit()
        self.portfolio.update_cash(self.current_date)
        self.portfolio.update_equity(self.current_date)
        

    @abstractmethod
    def step_forward(self, data: Dict[str, Any]) -> None:
        """
        Called when new data is available for the strategy to make decisions.

        This method is the main user interface for implementing trading logic. It is called
        by the backtester on each step with the latest available market data.

        Args:
            data: A dictionary containing the latest market data with the following structure:
                - Keys: Field names (e.g., 'open', 'close', 'volume', custom indicators)
                - Values: Either DataFrame or Series objects

                Data types:
                - DataFrame: Ticker-specific data where columns are ticker symbols
                - Series: Non-ticker specific data (e.g., market indices, economic indicators)

                All data objects share the same index (timestamps). Access patterns:
                - Latest data: data['field_name'].iloc[-1]
                - Previous data: data['field_name'].iloc[-2]

        Note:
            Use the provided 'data' parameter instead of accessing self.data or calling
            self._get_data() directly.
        """
        pass
    
    def _get_data(self, date, length=1)->dict[str, pd.DataFrame|pd.Series]|None:
        """
        Get data for a specific date and length.
        
        Args:
            date: The target date to get data for
            length: The number of dates to include (including the target date)
            
        Returns:
            A dictionary with field names as keys and DataFrames/Series as values
        """
        result = {}
        
        # Handle case where data is not initialized
        if not hasattr(self, 'data') or self.data is None:
            return None
        
        # Check data length - if any data field has enough rows, return it
        has_enough_data = False
        for field_name, field_data in self.data.items():
            if isinstance(field_data, (pd.DataFrame, pd.Series)) and len(field_data) >= length:
                has_enough_data = True
                break
                
        if not has_enough_data:
            return None
                   
        def get_n_rows_to_date(data: pd.DataFrame|pd.Series, date: Any, n: int)->pd.Series|pd.DataFrame|None:
            # Get all rows up to and including the date
            subset = data.loc[:date]
            
            # If we have fewer rows than requested, return None
            if len(subset) < n:
                return None
            
            # Otherwise, return the last n rows
            return subset.iloc[-n:]

        # Process each field in the data
        for field_name, field_data in self.data.items():
            if field_name == 'index':  # Skip the index item
                continue
            if isinstance(field_data, pd.DataFrame):
                # For DataFrames (multiple symbols), always return a DataFrame
                # Return a DataFrame with the specified date range
                data_subset = get_n_rows_to_date(field_data, date, length)
                if data_subset is None:
                    return None
                result[field_name] = data_subset
            elif isinstance(field_data, pd.Series):
                # For Series (single symbol), always return a Series
                # Return a Series with the specified date range
                data_subset = get_n_rows_to_date(field_data, date, length)
                if data_subset is None:
                    return None
                result[field_name] = data_subset
        return result

    def _update_market_handler(self)->None:
        """Update market handler with new market data according to self.current_date.
        use access_case_insensitive() to access self.data
        must consider type of self.data (multiple symbols or single symbol)
        In the case of single symbol, self.market_handler._data will be {str :float}
        In the case of multiple symbols, self.market_handler._data will be {str : {str : float}}
        raise error if self.data has no key of 'Close', 'High', 'Low', 'Open'
        
        """
        if not hasattr(self, 'data') or self.data is None:
            raise AttributeError("Strategy has no data attribute or data is None")
        
        # Check if all required fields are present
        required_fields = ['open', 'high', 'low', 'close']
        for field in required_fields:
            if not any(k.lower() == field.lower() for k in self.data.keys()):
                raise ValueError(f"Required field '{field}' not found in data")
        
        # Get data for the current date
        data = self._get_data(self.current_date, length=1)
        
        # If no data is available for the current date, return
        if not data:
            return
        
        # Create a new dictionary for market data
        market_data = {}
        
        # Process each field
        for field in required_fields:
            # Use access_case_insensitive to get the field data regardless of case
            
            field_data = access_case_insensitive(field, data)
            
            # Check if the field data is a Series (multiple symbols) or scalar (single symbol)
            if isinstance(field_data, pd.DataFrame):
                # Multiple symbols case
                # Create a dictionary for each symbol
                market_data[field] = {}
                for symbol in field_data.columns:
                    market_data[field][symbol] = field_data[symbol].iloc[-1]
            else:
                # Single symbol case
                market_data[field] = field_data.iloc[-1]
        
        # Update the market handler with the new data
        self.market_handler.update_market_data(market_data)
        self.market_handler.current_date = self.current_date

    def _check_stop_loss(self)->None:
        """Check and handle stop loss triggers for all open trades.

        This method:
        1. Gets current live trades
        2. Calls check_stop_loss() of MarketHandler for each trade
        3. If triggered, closes the trade with the returned closing trade
        
        Note:
            The trade will not be removed from the portfolio, but its properties will be updated.
            According to rule of conservative, if both stop loss and take profit are triggered,
            we assume that the stop loss is hit first.
        """
        if not self.market_handler:
            return
            
        live_trades = self.portfolio.live_trades()
        
        for trade in live_trades:
            closing_trade = self.market_handler.check_stop_loss(trade)
            if closing_trade:
                trade.close(closing_trade)

    def _check_take_profit(self)->None:
        """Check and handle take profit triggers for all open trades.

        This method:
        1. Gets current live trades
        2. Calls check_take_profit() of MarketHandler for each trade
        3. If triggered, closes the trade with the returned closing trade
        
        Note:
            The trade will not be removed from the portfolio, but its properties will be updated.
            According to rule of conservative, if both stop loss and take profit are triggered,
            we assume that the stop loss is hit first.
        """
        if not self.market_handler:
            return
            
        live_trades = self.portfolio.live_trades()
        
        for trade in live_trades:
            closing_trade = self.market_handler.check_take_profit(trade)
            if closing_trade:
                trade.close(closing_trade)

    def _process_trade(self, 
                  direction: Direction,
                  symbol: List[str]|str|None=None, 
                  quantity: List[int]|int|None=None, 
                  value: List[float]|float|None=None,
                  stop_loss_percent: List[float]|float|None=None,
                  stop_loss_value: List[float]|float|None=None,
                  take_profit_percent: List[float]|float|None=None,
                  take_profit_value: List[float]|float|None=None,
                  )->List[Trade]|Trade|None:
        """
        Helper method to process trade creation for both buy and sell operations.
        
        Args:
            direction: Direction of the trade (LONG for buy, SHORT for sell)
            symbol: Symbol(s) to trade. If None, uses all available symbols.
            quantity: Quantity to trade. Cannot be used with value.
            value: Dollar value to trade. Cannot be used with quantity.
            stop_loss_percent: Stop loss as a percentage of entry price.
            stop_loss_value: Stop loss as a dollar value from entry price.
            take_profit_percent: Take profit as a percentage of entry price.
            take_profit_value: Take profit as a dollar value from entry price.
            
        Returns:
            List[Trade]|Trade|None: The trade(s) added to the portfolio, or None if no trades were added.
            
        Raises:
            ValueError: If input validation fails.
        """
        # 1. Validate the input arguments
        if quantity is not None and value is not None:
            raise ValueError("Only one of quantity or value should be provided")
        
        if quantity is None and value is None:
            raise ValueError("Either quantity or value must be provided")
            
        # Handle the case where symbol is None
        symbol = '' if symbol is None else symbol
        
        # Convert single values to lists for uniform processing
        is_list_input = isinstance(symbol, list)
        
        if not is_list_input:
            symbols = [symbol]
            quantities = [quantity] if quantity is not None else None
            values = [value] if value is not None else None
            sl_percents = [stop_loss_percent] if stop_loss_percent is not None else None
            sl_values = [stop_loss_value] if stop_loss_value is not None else None
            tp_percents = [take_profit_percent] if take_profit_percent is not None else None
            tp_values = [take_profit_value] if take_profit_value is not None else None
        else:
            symbols = symbol
            quantities = quantity if isinstance(quantity, list) else [quantity] * len(symbols) if quantity is not None else None
            values = value if isinstance(value, list) else [value] * len(symbols) if value is not None else None
            sl_percents = stop_loss_percent if isinstance(stop_loss_percent, list) else [stop_loss_percent] * len(symbols) if stop_loss_percent is not None else None
            sl_values = stop_loss_value if isinstance(stop_loss_value, list) else [stop_loss_value] * len(symbols) if stop_loss_value is not None else None
            tp_percents = take_profit_percent if isinstance(take_profit_percent, list) else [take_profit_percent] * len(symbols) if take_profit_percent is not None else None
            tp_values = take_profit_value if isinstance(take_profit_value, list) else [take_profit_value] * len(symbols) if take_profit_value is not None else None
        
        # Validate list lengths
        if quantities is not None and len(quantities) != len(symbols):
            raise ValueError("Length of quantity list must match length of symbol list")
        if values is not None and len(values) != len(symbols):
            raise ValueError("Length of value list must match length of symbol list")
        if sl_percents is not None and len(sl_percents) != len(symbols):
            raise ValueError("Length of stop_loss_percent list must match length of symbol list")
        if sl_values is not None and len(sl_values) != len(symbols):
            raise ValueError("Length of stop_loss_value list must match length of symbol list")
        if tp_percents is not None and len(tp_percents) != len(symbols):
            raise ValueError("Length of take_profit_percent list must match length of symbol list")
        if tp_values is not None and len(tp_values) != len(symbols):
            raise ValueError("Length of take_profit_value list must match length of symbol list")
        
        # 2. Get the Trades by calling appropriate method of MarketHandler
        trades = []
        for i, sym in enumerate(symbols):
            # Create trade based on direction
            if direction == Direction.LONG:
                if quantities is not None:
                    trade = self.market_handler.buy_at_open(sym, quantity=quantities[i])
                else:
                    trade = self.market_handler.buy_at_open(sym, value=values[i])
            else:  # Direction.SHORT
                if quantities is not None:
                    trade = self.market_handler.sell_at_open(sym, quantity=quantities[i])
                else:
                    trade = self.market_handler.sell_at_open(sym, value=values[i])
            
            # Add stop loss if provided
            if sl_percents is not None:
                sl_factor = (1 - sl_percents[i]/100) if direction == Direction.LONG else (1 + sl_percents[i]/100)
                sl_price = trade.entry_price * sl_factor
                trade.add_stop_loss(price=sl_price)
            elif sl_values is not None:
                sl_price = trade.entry_price - sl_values[i] if direction == Direction.LONG else trade.entry_price + sl_values[i]
                trade.add_stop_loss(price=sl_price)
            
            # Add take profit if provided
            if tp_percents is not None:
                tp_factor = (1 + tp_percents[i]/100) if direction == Direction.LONG else (1 - tp_percents[i]/100)
                tp_price = trade.entry_price * tp_factor
                trade.add_take_profit(price=tp_price)
            elif tp_values is not None:
                tp_price = trade.entry_price + tp_values[i] if direction == Direction.LONG else trade.entry_price - tp_values[i]
                trade.add_take_profit(price=tp_price)
            
            trades.append(trade)
        
        # 3. Check if the trades are opposite to existing trades
        remaining_trades = self.portfolio.check_opposite_position(trades)
        
        # 4. Add the trades to the portfolio if they weren't used to close other trades
        added_trades = []
        for i, trade in enumerate(remaining_trades):
            if trade is not None:  # Trade wasn't used to close an opposite position
                self.portfolio.add_position(trade)
                added_trades.append(trade)
        
        # 5. Return the trades which were added to the portfolio
        if len(added_trades) == 0:
            return None
        elif not is_list_input:
            return added_trades[0] if added_trades else None
        else:
            return added_trades

    def buy(self, symbol: List[str]|str|None=None, 
            quantity: List[int]|int|None=None, 
            value: List[float]|float|None=None,
            stop_loss_percent: List[float]|float|None=None,
            stop_loss_value: List[float]|float|None=None,
            take_profit_percent: List[float]|float|None=None,
            take_profit_value: List[float]|float|None=None,
            )->List[Trade]|Trade|None:
        """
        Add a buy position to the portfolio.
        
        Args:
            symbol: Symbol(s) to trade. If None, uses all available symbols.
            quantity: Quantity to buy. Cannot be used with value.
            value: Dollar value to buy. Cannot be used with quantity.
            stop_loss_percent: Stop loss as a percentage of entry price.
            stop_loss_value: Stop loss as a dollar value below entry price.
            take_profit_percent: Take profit as a percentage of entry price.
            take_profit_value: Take profit as a dollar value above entry price.
            
        Returns:
            List[Trade]|Trade|None: The trade(s) added to the portfolio, or None if no trades were added.
            
        Raises:
            ValueError: If input validation fails.
        """
        return self._process_trade(
            direction=Direction.LONG,
            symbol=symbol,
            quantity=quantity,
            value=value,
            stop_loss_percent=stop_loss_percent,
            stop_loss_value=stop_loss_value,
            take_profit_percent=take_profit_percent,
            take_profit_value=take_profit_value
        )

    def sell(self, symbol: List[str]|str|None=None, 
            quantity: List[int]|int|None=None, 
            value: List[float]|float|None=None,
            stop_loss_percent: List[float]|float|None=None,
            stop_loss_value: List[float]|float|None=None,
            take_profit_percent: List[float]|float|None=None,
            take_profit_value: List[float]|float|None=None,
            )->List[Trade]|Trade|None:
        """
        Add a sell position to the portfolio.
        
        Args:
            symbol: Symbol(s) to trade. If None, uses all available symbols.
            quantity: Quantity to sell. Cannot be used with value.
            value: Dollar value to sell. Cannot be used with quantity.
            stop_loss_percent: Stop loss as a percentage of entry price.
            stop_loss_value: Stop loss as a dollar value above entry price.
            take_profit_percent: Take profit as a percentage of entry price.
            take_profit_value: Take profit as a dollar value below entry price.
            
        Returns:
            List[Trade]|Trade|None: The trade(s) added to the portfolio, or None if no trades were added.
            
        Raises:
            ValueError: If input validation fails.
        """
        return self._process_trade(
            direction=Direction.SHORT,
            symbol=symbol,
            quantity=quantity,
            value=value,
            stop_loss_percent=stop_loss_percent,
            stop_loss_value=stop_loss_value,
            take_profit_percent=take_profit_percent,
            take_profit_value=take_profit_value
        )
    def _analyze_data(self):
        """Analyze the data in self.data before running the back-testing simulation
           called by run() after self.initialize()"""
        print('Analyzing data...')
        if not self.data:
            print("No data found in self.data")
            return
        
        # Print all items in self.data
        print(f"items in self.data: {', '.join([repr(k) for k in self.data.keys()])}")
        
        # Get reference index from 'close' data
        close_key = next((k for k in self.data.keys() if k.lower() == 'close'), None)
        if not close_key:
            print("Warning: 'close' data not found for reference index")
            return
            
        reference_index = self.data[close_key].index
        
        # Track items that are OK
        ok_items = []
        
        # Analyze each item in self.data
        for key, item in self.data.items():
            if key == 'index':  # Skip the index item
                continue
                
            # Check if it's a Series or DataFrame
            if isinstance(item, pd.Series):
                # 1. Check if it has the same date index
                if not item.index.equals(reference_index):
                    first_indices = ', '.join([str(idx) for idx in item.index[:3]])
                    last_indices = ', '.join([str(idx) for idx in item.index[-3:]])
                    print(f"Warning: mismatch index of {key}: [{first_indices}, ...{last_indices}](length={len(item.index)}, dtype={item.index.dtype})")
                    continue
                
                # 2. Check if it's all NaN
                if item.isna().all():
                    print(f"Warning: '{key}' is all NaN.")
                    continue
                
                # 3. Check for NaN in the middle
                non_na_indices = np.where(~(item.isna()))[0]
                if len(non_na_indices) > 0:
                    first_valid = non_na_indices[0]
                    last_valid = non_na_indices[-1]
                    middle_slice = item.iloc[first_valid:last_valid+1]
                    middle_na_count = middle_slice.isna().sum()
                    
                    if middle_na_count > 0:
                        print(f"Warning: '{key}' has {middle_na_count} missing values (for case of pd.Series)")
                        continue
                
                # 4. Check if signal/criteria/screen/filter items have triggered
                if any(key.startswith(prefix) for prefix in ['signal_', 'criteria_', 'screen_', 'filter_']):
                    # Check if it's boolean or numeric and if any value is True or 1
                    if (item.dtype == bool or np.issubdtype(item.dtype, np.number)) and not item.any():
                        print(f"Warning: '{key}' has never triggered. (for case of pd.Series)")
                        continue
                
                # If all checks pass, add to OK items
                ok_items.append(key)
                
            elif isinstance(item, pd.DataFrame):
                # 1. Check if it has the same date index
                if not item.index.equals(reference_index):
                    first_indices = ', '.join([str(idx) for idx in item.index[:3]])
                    last_indices = ', '.join([str(idx) for idx in item.index[-3:]])
                    print(f"Warning: mismatch index of {key}: [{first_indices}, ...{last_indices}](length={len(item.index)}, dtype={item.index.dtype})")
                    continue
                
                # 2. Check if it's all NaN
                if item.isna().all().all():
                    print(f"Warning: '{key}' is all NaN.")
                    continue
                
                # 3. Check for NaN in the middle for each column
                total_missing = 0
                problem_tickers = []
                
                for col in item.columns:
                    col_data = item[col]
                    non_na_indices = np.where(~(col_data.isna()))[0]
                    
                    if len(non_na_indices) > 0:
                        first_valid = non_na_indices[0]
                        last_valid = non_na_indices[-1]
                        middle_slice = col_data.iloc[first_valid:last_valid+1]
                        middle_na_count = middle_slice.isna().sum()
                        
                        if middle_na_count > 0:
                            total_missing += middle_na_count
                            problem_tickers.append(col)
                
                if total_missing > 0:
                    # Display up to 3 tickers if there are too many
                    ticker_display = problem_tickers[:3]
                    if len(problem_tickers) > 3:
                        ticker_display_str = f"{ticker_display[0]}, {ticker_display[1]}, {ticker_display[2]}, and other {len(problem_tickers) - 3} tickers"
                    else:
                        ticker_display_str = ", ".join([f"{t}" for t in ticker_display])
                    
                    print(f"Warning: '{key}' totally has {total_missing} missing values in tickers {ticker_display_str}")
                    continue
                
                # 4. Check if signal/criteria/screen/filter items have triggered
                if any(key.startswith(prefix) for prefix in ['signal_', 'criteria_', 'screen_', 'filter_']):
                    # Check if it's boolean or numeric and if any value is True or 1
                    if (item.dtypes.iloc[0] == bool or np.issubdtype(item.dtypes.iloc[0], np.number)):
                        # Check which columns never triggered
                        never_triggered = []
                        for col in item.columns:
                            if not item[col].any():
                                never_triggered.append(col)
                        
                        if never_triggered:
                            # Display up to 3 tickers if there are too many
                            ticker_display = never_triggered[:3]
                            if len(never_triggered) > 3:
                                ticker_display_str = f"{ticker_display[0]}, {ticker_display[1]}, {ticker_display[2]}, and other {len(never_triggered) - 3} tickers"
                            else:
                                ticker_display_str = ", ".join([f"{t}" for t in ticker_display])
                            
                            print(f"Warning: in '{key}', {ticker_display_str} have never triggered")
                            continue
                
                # If all checks pass, add to OK items
                ok_items.append(key)
        
        # Print OK items
        if ok_items:
            if len(ok_items) == len(self.data) - (1 if 'index' in self.data else 0):
                print("All items OK.")
            else:
                print(f"data {', '.join([repr(item) for item in ok_items])} OK")
    
    def run(self, 
            start_date: Optional[datetime], 
            end_date: Optional[datetime],
            data_length: int=1) -> None:
        """Run the back-testing simulation."""
        # 1. Before looping
        # a. Call initialize()
        self.initialize()
        self._analyze_data()
        
        # b. Use valid_date_range() to get the valid date range
        if not hasattr(self, 'data') or not self.data:
            raise ValueError("Strategy has no data or data is empty")
        
        
        
        # Get valid date range from data
        valid_start, valid_end = valid_date_range(self.data)
        
        # Override with user-specified dates if provided
        if start_date is not None:
            valid_start = max(valid_start, start_date)
        if end_date is not None:
            valid_end = min(valid_end, end_date)
            
        # Ensure start date is before end date
        if valid_start > valid_end:
            raise ValueError(f"Start date {valid_start} is after end date {valid_end}")
        
        # c. Add 'index' item to self.data with the valid date range
        # Get all dates in the valid range
        # If 'index' is not in self.data, extract it from another item
        if 'index' not in self.data:
            # Find the first item that is a Series or DataFrame
            for key, value in self.data.items():
                if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
                    # Extract the index and add it to self.data
                    self.data['index'] = pd.Series(value.index, index=value.index)
                    break
            else:
                # If no Series or DataFrame is found, raise an error
                raise ValueError("No Series or DataFrame found in self.data to extract index from")
        all_dates = sorted([d for d in self.data['index'] if valid_start <= d <= valid_end])
        

        # 2. Loop through each date in the valid date range
        for date in all_dates:

            # a. Update self.current_date to the looping date
            self.current_date = date
            
            # b. Call _before_step()
            self._before_step()
            
            # c. Use self.get_data() to get the data for the current date and call step_forward()
            if self.previous_date is not None:
                current_data = self._get_data(self.previous_date, length=data_length)
                if current_data is not None:
                    data_handler = DataHandler(current_data)
                    self.step_forward(data_handler)
            
            # d. Call _after_step()
            self._after_step()
            self.previous_date = date

    # *************utils*************
    def days_since_last_trade(self) -> int:
        if self.current_date is None:
            return None
        last_trade_date = self.portfolio.date_last_trade()
        if last_trade_date is None:
            return None
        return (self.current_date - last_trade_date).days


    def check_signals(self):
        """
        Checks for signal items that are all False or 0 and prints warnings.
        
        Examines all items with prefixes 'signal_', 'criteria_', 'screen_', 'filter_'
        """
        signal_prefixes = ['signal_', 'criteria_', 'screen_', 'filter_']
        warnings = []
        
        for key in self.data:
            if any(key.startswith(prefix) for prefix in signal_prefixes):
                data_item = self.data[key]
                
                # Check if the item is a boolean or numeric type
                if isinstance(data_item, pd.DataFrame):
                    latest_data = data_item.iloc[-1]
                    if (latest_data.dtype == bool or np.issubdtype(latest_data.dtype, np.number)) and \
                       latest_data.sum() == 0:
                        warnings.append(f"Warning: Signal '{key}' has all False/0 values")
                elif isinstance(data_item, pd.Series):
                    if (data_item.dtype == bool or np.issubdtype(data_item.dtype, np.number)) and \
                       data_item.iloc[-1] == 0:
                        warnings.append(f"Warning: Signal '{key}' has False/0 value")
        
        if warnings:
            print("\n".join(warnings))

    def check_indicators(self):
        """
        Checks for indicator items that are all NaN and prints warnings.
        
        Excludes basic market data: open, high, low, close, volume, adj_close
        """
        basic_data = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        warnings = []
        
        for key in self.data:
            if key.lower() not in basic_data:
                data_item = self.data[key]
                
                # Check if it's a Series or DataFrame
                if isinstance(data_item, pd.Series):
                    # Check if it's all NaN
                    if data_item.isna().all():
                        warnings.append(f"Warning: Indicator '{key}' is all NaN")
                elif isinstance(data_item, pd.DataFrame):
                    # Check if it's all NaN
                    if data_item.isna().all().all():
                        warnings.append(f"Warning: Indicator '{key}' is all NaN")
        
        if warnings:
            print("\n".join(warnings))



# short hand for common strategy functions
def screen(data, fields:List[str])->list:
    """
    Screen the data for the specified fields.

    Args:
        data (Dict[str, pd.Series]): The data to screen.
        Its just the data argument you will get in Strategy.step_forward().
        fields (List[str]): The fields to screen for.

    Returns:
        List[str]: The symbols that match the criteria.
    """
    first_data = data[fields[0]]
    columns = first_data.columns
    
    global_mask = pd.Series(True, index=columns)
    for field in fields:
        if field not in data:
            raise ValueError(f"Field {field} not found in data")
        field_data = data[field].iloc[-1].transpose()
        global_mask &= field_data
    return global_mask.index[global_mask].tolist()


def screen_sort(data, fields:List[str], highest:str)->str:
    """
    Screen the data for the specified fields.
    A useful shorthand for multi-symbol strategy which screen according to multiple criteria and sort by one field.

    Args:
        data (Dict[str, pd.Series]): The data to screen.
        Its just the data argument you will get in Strategy.step_forward().
        fields (List[str]): The fields to screen for.
        highest (str): The field to sort by.

    Returns:
        str: The symbol that is highest in the specified field and matches the criteria.
    """
    if highest not in data.keys():
        raise ValueError(f"Field {highest} not found in data")
    if not isinstance(data[highest], pd.DataFrame):
        raise ValueError(f"Field {highest} is not a DataFrame")
    
    for field in fields:
        if field not in data.keys():
            raise ValueError(f"Field {field} not found in data")
        if not isinstance(data[field], pd.DataFrame):
            raise ValueError(f"Field {field} is not a DataFrame")
        if len(data[field].columns) != len(data[highest].columns):
            raise ValueError(f"All fields must have the same number of columns")
    
    first_data = data[fields[0]]
    columns = first_data.columns
    global_mask = pd.Series(True, index=columns)
    for field in fields:
        field_data = data[field].iloc[-1].transpose()
        global_mask &= field_data
    data_for_highest = data[highest].iloc[-1]
    
    filtered_highest = data_for_highest[global_mask.to_list()]
    if filtered_highest.empty:
        return None
    return filtered_highest.idxmax()