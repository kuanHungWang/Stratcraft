# StratCraft

A Python backtesting framework for building and evaluating quantitative trading strategies. StratCraft provides a clean, decorator-based API for defining indicators, handling multi-symbol data, and running event-driven backtests with portfolio tracking and performance analytics.

## Features

- **Event-driven backtesting** — step-by-step simulation with clean separation of `initialize()` and `step_forward()` hooks
- **Multi-symbol support** — buy/sell across multiple tickers in a single call
- **Decorator toolkit** — `@broadcast`, `@rolling`, `@grouping`, `@available` for composable indicator computation
- **Portfolio management** — track cash, equity, open positions, stop-loss and take-profit triggers
- **Performance metrics** — Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, win rate, profit factor, and more
- **Interactive charts** — Plotly-based equity curve with trade entry markers and optional additional panels
- **Data screening** — filter and rank symbols by criteria within `step_forward()`

## Installation

Install from PyPI:

```bash
pip install stratcraft
```

[![PyPI version](https://badge.fury.io/py/stratcraft.svg)](https://badge.fury.io/py/stratcraft)

Or install from source:

```bash
git clone https://github.com/kuanhungwang/stratcraft.git
cd stratcraft
pip install -e .
```

## Quick Start

```python
from stratcraft import Strategy, DataHandler
from datetime import datetime

class SMAStrategy(Strategy):
    def initialize(self):
        # Load price data and compute indicators here
        self.data = {
            'open':  open_df,
            'high':  high_df,
            'low':   low_df,
            'close': close_df,
        }
        self.data['sma20'] = close_df.rolling(20).mean()

    def step_forward(self, data: DataHandler):
        close = data['close']      # latest close (scalar or Series)
        sma20 = data['sma20']

        if close > sma20:
            self.buy(symbol='AAPL', value=10_000)
        else:
            self.sell(symbol='AAPL', value=10_000)

strategy = SMAStrategy(initial_capital=100_000)
strategy.run(
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31),
    data_length=2,   # rows of history passed to step_forward
)
```

## Core Concepts

### Strategy lifecycle

| Method | When called | Purpose |
|---|---|---|
| `initialize()` | Once before the loop | Load data, compute indicators, set parameters |
| `step_forward(data)` | Every trading day | Implement trading logic, call `buy()`/`sell()` |

### Buying and selling

```python
# By quantity
self.buy(symbol='AAPL', quantity=100)
self.sell(symbol='AAPL', quantity=100)

# By dollar value
self.buy(symbol='AAPL', value=10_000)

# With stop-loss and take-profit (percentage of entry price)
self.buy(symbol='AAPL', quantity=100, stop_loss_percent=5, take_profit_percent=10)

# Multiple symbols at once
self.buy(symbol=['AAPL', 'MSFT'], value=[5_000, 5_000])
```

### Accessing historical data inside `step_forward`

```python
def step_forward(self, data: DataHandler):
    latest_close  = data['close']           # latest bar
    prev_close    = data[('close', -2)]     # bar before latest
    latest_sma    = data['sma20']
```

> Set `data_length` in `strategy.run()` to the maximum look-back depth you need.

### Trailing stop-loss

```python
from stratcraft import TrailingStopLoss, Direction

trade = self.buy(symbol='AAPL', quantity=100)
if trade:
    trade.stop_loss = TrailingStopLoss(
        price=trade.entry_price - 1.0,   # initial stop price
        distance=1.0,                     # trail distance
        threshold=trade.entry_price + 1.0,
        direction=Direction.LONG,
    )
```

Call `trade.stop_loss.reset_price(current_price)` each bar to move the stop up.

### Screening and ranking symbols

```python
def step_forward(self, data: DataHandler):
    # Filter symbols passing all boolean criteria
    candidates = data.screen(['criteria_momentum', 'criteria_volume'])

    # Rank by a field, optionally from a filtered subset
    top5 = data.highest('momentum_score', n=5, tickers=candidates)
    bot5 = data.lowest('volatility', n=5)
```

Prefix keys with `signal_`, `criteria_`, `screen_`, or `filter_` in `self.data` — StratCraft will warn you during `run()` if any never triggered.

### Portfolio queries inside `step_forward`

```python
cash         = self.portfolio.cash
equity       = self.portfolio.equity
live         = self.portfolio.live_trades()
cost_aapl    = self.portfolio.cost(symbol='AAPL')
mkt_val      = self.portfolio.current_market_value()
invest_ratio = self.portfolio.invest_ratio()
days         = self.days_since_last_trade()
```

## Decorators

Decorators live in `stratcraft.decorators` and are designed to be composed.

### `@broadcast`

Applies a single-symbol (Series) function across all columns of a DataFrame:

```python
from stratcraft.decorators import broadcast
import ta

@broadcast
def RSI(price):
    return ta.momentum.RSIIndicator(price).rsi()

rsi = RSI(close_df)   # returns DataFrame with same columns as close_df
```

### `@rolling`

Turns a scalar reduction into a rolling-window Series/DataFrame:

```python
from stratcraft.decorators import rolling

@rolling(window=14)
def avg_range(high, low):
    return (high - low).mean()

atr = avg_range(high_df, low_df)
```

Compose with `@broadcast` for multi-symbol rolling indicators:

```python
@broadcast
@rolling(window=20)
def momentum(price):
    return (price.iloc[-1] / price.iloc[0]) - 1
```

### `@grouping`

Applies a function to user-defined groups of symbols and returns a group-level DataFrame:

```python
from stratcraft.decorators import grouping

sector = {'Technology': ['AAPL', 'MSFT'], 'Finance': ['JPM', 'BAC']}

@grouping(groups=sector)
def sector_return(price):
    return price.pct_change().mean(axis=1)

sector_ret = sector_return(close_df)  # columns: Technology, Finance
```

### `@available`

Aligns low-frequency data (e.g. quarterly earnings) to a daily time series, exposing only data that would have been available on each date:

```python
from stratcraft.decorators import available, broadcast

@broadcast
@available(looping_dates=close.index, length=1)
def daily_eps(eps):
    return eps.iloc[-1]

eps_daily = daily_eps(fundamental['is_eps'], available_date=fundamental['fillingDate'])
pe = close / eps_daily
```

## Performance Analysis

```python
from stratcraft.metrics import Metrics

trade_df  = strategy.portfolio.trade_history()
pl_df     = strategy.portfolio.pl_history()

m = Metrics(trade_df, pl_df)

# Full metrics dict
results = m.metrics()

# Concise subset: cumulative return, annual return, win rate, volatility, Sharpe, max drawdown, # trades
results = m.metrics(concise=True)

Metrics.pretty_print(results)
```

**Metrics computed:**

| Metric | Description |
|---|---|
| Cumulative returns | Total return over the period |
| Annual return | Annualised CAGR |
| Annual volatility | Annualised std dev of daily returns |
| Sharpe ratio | Annual return / annual volatility |
| Sortino ratio | Annual return / downside deviation |
| Calmar ratio | Annual return / max drawdown |
| Max drawdown | Largest peak-to-trough decline |
| Omega ratio | Sum of gains / sum of losses above threshold |
| Stability | R² of linear fit on cumulative returns |
| Win rate | Fraction of closed trades that are profitable |
| Profit factor | Total wins / total losses |
| Avg win / loss trade p/l | Mean P&L of winning/losing trades |
| Skew / Kurtosis | Distribution shape of daily returns |
| Tail ratio | 95th percentile return / 5th percentile return |

### Interactive chart

```python
fig = m.chart_history()
fig.show()

# With additional panels
fig = m.chart_history(additional_chart_data={
    'price_data':  {'AAPL': aapl_df},
    'indicators':  {'RSI': rsi_series, 'SMA20': sma_series},
    'cash_ratio':  pl_df['cash'] / pl_df['equity'],
})
fig.write_html('backtest.html')
```

## Project Structure

```
stratcraft/
├── stratcraft.py      # Core: Strategy, Portfolio, Trade, MarketHandler, DataHandler
├── decorators.py      # @broadcast, @rolling, @grouping, @available
├── metrics.py         # Metrics class with performance analytics and Plotly charts
├── util.py            # Helper utilities (case-insensitive access, date range, symbol alignment)
├── examples.py        # API usage reference
└── examples/          # Complete runnable strategy examples
    ├── strategy1_single_stock_technical.py
    ├── strategy2_multi_stock_technical.py
    ├── strategy3_multi_position_technical.py
    ├── strategy4_single_stock_technical_fundamental.py
    ├── strategy5_multi_stock_technical_fundamental.py
    └── strategy6_comparing_index.py
```

## Requirements

- Python >= 3.10
- pandas >= 1.5
- numpy >= 1.23
- plotly >= 5.0

## License

MIT
