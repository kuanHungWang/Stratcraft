import pandas as pd
import numpy as np
from stratcraft.decorators import broadcast


@broadcast
def sma(data: pd.Series, period: int = 20) -> pd.Series:
    return data.rolling(window=period).mean()


@broadcast
def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@broadcast
def bb_upper(data: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
    mid = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    return mid + std_dev * std


@broadcast
def bb_middle(data: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
    return data.rolling(window=period).mean()


@broadcast
def bb_lower(data: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
    mid = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    return mid - std_dev * std
