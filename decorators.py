from functools import wraps
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Union, TypeVar, Optional

# Type hints
DataType = TypeVar('DataType', pd.Series, pd.DataFrame)
SeriesOrFloat = Union[pd.Series, float]

def broadcast(func: Callable = None, *, groups: Optional[Dict[str, List[str]]] = None) -> Callable:
    """
    Decorator to broadcast a function that operates on a single symbol (Series)
    to work with multiple symbols (DataFrame).
    
    The decorated function should take a pd.Series as its first argument.
    When called with a DataFrame, the function will be applied to each column.
    
    When groups is provided, the behavior is inverse of grouping:
    It broadcasts group-level data to individual members within that group.
    
    Args:
        func (Callable, optional): Function that operates on a single symbol (Series)
        groups (Dict[str, List[str]], optional): Dictionary mapping group names to lists of symbols.
            When provided, enables inverse grouping functionality.
        
    Returns:
        Callable: Function that can operate on both Series and DataFrame
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> DataType:
            # special case of kwargs for available_date
            available_date = kwargs.pop('available_date', None) 
            if groups is None:
                # Original broadcast functionality
                # Check if any args are DataFrames
                df_indices = [i for i, arg in enumerate(args) if isinstance(arg, pd.DataFrame)]
                
                if not df_indices:
                    # No DataFrames found, just call the function normally
                    return func(*args, **kwargs)
                
                
                # Get the shape from the first DataFrame found
                first_df = args[df_indices[0]]
                result_index = first_df.index
                result_columns = first_df.columns
                
                # Verify all DataFrames have compatible shapes
                for idx in df_indices[1:]:
                    df = args[idx]
                    if not df.columns.equals(result_columns) or not df.index.equals(result_index):
                        raise ValueError(f"All DataFrame inputs must have the same shape and alignment, get {df.shape} VS {first_df.shape}")
                
                # Create list of arguments for each column
                results = {}
                for col in result_columns:
                    col_args = list(args)  # Create a new list for each column
                    # Replace DataFrame arguments with their column Series
                    for df_idx in df_indices:
                        col_args[df_idx] = args[df_idx][col]
                    # special process for available_date kwargs if provided.
                    if available_date is not None:
                        results[col] = func(*col_args, available_date=available_date[col], **kwargs)
                    else:
                        results[col] = func(*col_args, **kwargs)
                    
                # If the result is a Series/scalar for each column, combine into DataFrame
                if all(isinstance(val, (pd.Series, float, int)) for val in results.values()):
                    return pd.DataFrame(results)

                return results
            else:
                # Inverse grouping functionality
                # The last argument should be the grouped DataFrame
                if len(args) < 1:
                    raise ValueError("At least one argument is required for grouped broadcast")
                
                # Find DataFrame arguments
                df_indices = [i for i, arg in enumerate(args) if isinstance(arg, pd.DataFrame)]
                if not df_indices:
                    raise ValueError("No DataFrame arguments found")
                
                # The last DataFrame should be the grouped data
                grouped_df_idx = df_indices[-1]
                grouped_df = args[grouped_df_idx]
                
                # Verify the grouped DataFrame has the group names as columns
                missing_groups = [group for group in groups.keys() if group not in grouped_df.columns]
                if missing_groups:
                    raise ValueError(f"Grouped DataFrame is missing the following group columns: {missing_groups}")
                
                # Get the individual DataFrame(s) for validation and result index
                individual_df_indices = df_indices[:-1]
                if individual_df_indices:
                    first_individual_df = args[individual_df_indices[0]]
                    result_index = first_individual_df.index
                    
                    # Verify all individual DataFrames have compatible indices
                    for idx in individual_df_indices[1:]:
                        if not args[idx].index.equals(result_index):
                            raise ValueError("All individual DataFrame inputs must have the same index alignment")
                    
                    # Verify grouped DataFrame has compatible index
                    if not grouped_df.index.equals(result_index):
                        raise ValueError("Grouped DataFrame must have the same index as individual DataFrames")
                else:
                    # If no individual DataFrames, use the grouped DataFrame's index
                    result_index = grouped_df.index
                
                # Create result DataFrame
                result = pd.DataFrame(index=result_index)
                
                # Process each group and its members
                for group_name, members in groups.items():
                    if group_name not in grouped_df.columns:
                        continue  # Skip if group not in grouped DataFrame
                        
                    for member in members:
                        # Check if member exists in all individual DataFrames
                        valid_member = True
                        for idx in individual_df_indices:
                            if member not in args[idx].columns:
                                valid_member = False
                                break
                                
                        if valid_member:
                            # Prepare arguments for this member
                            member_args = list(args)
                            
                            # Replace individual DataFrames with their member column
                            for idx in individual_df_indices:
                                member_args[idx] = args[idx][member]
                                
                            # Replace grouped DataFrame with its group column
                            member_args[grouped_df_idx] = grouped_df[group_name]
                            
                            # Call function with prepared arguments
                            
                            if available_date is not None:
                                calculated_value = func(*member_args, available_date=available_date[member], **kwargs)
                            else:
                                calculated_value = func(*member_args, **kwargs)
                            try:
                                result[member] = calculated_value
                            except Exception as e:
                                print(f"Error setting result in broadcast with grouping: {e}")
                                print(f"column: {member}, group: {group_name}, calculated_value: {calculated_value}")
                                raise
                
                return result
        return wrapper
    
    # Handle both @broadcast and @broadcast(groups=...)
    if func is None:
        return decorator
    return decorator(func)

def rolling(window: int) -> Callable:
    """
    Decorator to create a rolling window calculation.
    
    The decorated function should take a pd.Series as input and return a scalar value.
    The decorator will apply this function to a rolling window of the data.
    
    Args:
        window (int): Size of the rolling window
        
    Returns:
        Callable: Function that performs rolling window calculation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*data: DataType, **kwargs) -> DataType:
            first_data = data[0]

            if isinstance(first_data, pd.DataFrame):
                result = pd.DataFrame(index=first_data.index, columns=first_data.columns)
            elif isinstance(first_data, pd.Series):
                result = pd.Series(index=first_data.index)
            else:
                raise ValueError("Input must be a pd.DataFrame or pd.Series")

            # Apply rolling window function
            for d in zip(*[r.rolling(window=window) for r in data]):
                first = d[0]
                if len(first) < window:
                    result.loc[first.index[-1]] = np.nan
                else:
                    try:
                        calculated_value = func(*d, **kwargs)
                        
                    except Exception as e:
                        print(f"Error applying function in rolling window: {e}")
                        print(f"rolling slices:")
                        for s in d:
                            print(f"type: {type(s)}, shape: {s.shape}, columns: {s.columns}")
                        raise
                    try:
                        index = first.index[-1]
                        result.loc[index] = calculated_value
                    except Exception as e:
                        print(f"Error setting result in rolling window: {e}")
                        print(f"index: {index}, calculated_value: {calculated_value}")
                        raise
            return result

        return wrapper
    return decorator

def grouping(groups: Dict[str, List[str]]) -> Callable:
    """
    Decorator to perform calculations on groups of symbols.
    
    The decorated function should take a DataFrame as input and return a Series or scalar.
    The decorator will apply this function to each group of symbols.
    
    Args:
        groups (Dict[str, List[str]]): Dictionary mapping group names to lists of symbols
        
    Returns:
        Callable: Function that performs group-wise calculations
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            # Find DataFrame arguments
            df_indices = [i for i, arg in enumerate(args) if isinstance(arg, pd.DataFrame)]
            if not df_indices:
                raise ValueError("No DataFrame arguments found")
                
            # Get reference DataFrame for index
            ref_df = args[df_indices[0]]
            
            # Verify all DataFrames have compatible indices
            for idx in df_indices[1:]:
                if not args[idx].index.equals(ref_df.index):
                    raise ValueError("All DataFrame inputs must have the same index alignment")
            
            results = {}
            for group_name, symbols in groups.items():
                # Prepare group arguments
                group_args = list(args)  # Create new args list for this group
                
                # Filter and replace each DataFrame with its group data
                valid_symbols = None
                for df_idx in df_indices:
                    df = args[df_idx]
                    # Get valid symbols for this DataFrame
                    current_valid = [s for s in symbols if s in df.columns]
                    # Initialize or update valid_symbols to intersection
                    if valid_symbols is None:
                        valid_symbols = current_valid
                    else:
                        valid_symbols = [s for s in valid_symbols if s in current_valid]
                        
                if valid_symbols:  # Only process group if it has valid symbols
                    # Replace each DataFrame with its filtered version
                    for df_idx in df_indices:
                        group_args[df_idx] = args[df_idx][valid_symbols]
                    
                    # Call function with group data
                    result = func(*group_args, **kwargs)
                    
                    # Handle different return types
                    if isinstance(result, (pd.Series, pd.DataFrame)):
                        results[group_name] = result
                    else:
                        results[group_name] = pd.Series(result, index=ref_df.index)
            
            return pd.DataFrame(results, index=ref_df.index)
        return wrapper
    return decorator

def available(looping_dates: pd.DatetimeIndex, length: int = 1) -> Callable:
    """
    Decorator for financial report data that provides only the data available on a given date.
    
    This decorator is useful when dealing with financial report data (quarterly, yearly, etc.)
    and applying it to daily trading decisions. It ensures that only data that would have been
    available on a given date is passed to the decorated function.
    
    Args:
        looping_dates (pd.DatetimeIndex): DatetimeIndex of dates to loop through (usually daily)
        length (int): Number of most recent data points to provide to the function
        
    Returns:
        Callable: Function that handles financial report data availability
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*data: pd.Series, available_date: pd.Series = None, **kwargs) -> pd.Series:
            if available_date is None:
                raise ValueError("available_date parameter is required for the available decorator")
                
            # Ensure data is a Series
            if not all(isinstance(d, pd.Series) for d in data):
                raise ValueError("data must be a pandas Series, use the @broadcast decorator for DataFrames")
                
            # Ensure available_date is a Series
            if not isinstance(available_date, pd.Series):
                raise ValueError(f"available_date must be a pandas Series with the same index as data, not {type(available_date)}")
                
            # Convert dates to datetime if they are strings
            if pd.api.types.is_string_dtype(available_date.values):
                available_date = pd.to_datetime(available_date)
                
            # Ensure indices match
            if not all(d.index.equals(available_date.index) for d in data):
                raise ValueError("data and available_date must have the same index")
                
            # Create result Series with the same index as looping_dates
            # Initialize with object dtype to avoid warnings when setting different types
            result = pd.Series(index=looping_dates, dtype='object')
            
            # Process each date in looping_dates
            for date in looping_dates:
                # Get indices where data is available on or before the current date
                mask = available_date <= date
                if mask.sum() >= length:
                    # Get the last 'length' available data points
                    avail_indices = mask[mask].index[-length:]
                    avail_data = [d.loc[avail_indices] for d in data]
                    try:
                        result_value = func(*avail_data, **kwargs)
                    except Exception as e:
                        print(f"Error applying function in available decorator: {e}")
                        print(f"input args:")
                        for s in avail_data:
                            print(f"type: {type(s)}, shape: {s.shape}")
                        raise
                    # Convert to float if boolean to avoid dtype warning
                    if isinstance(result_value, bool):
                        result_value = float(result_value)
                    result.loc[date] = result_value
                else:
                    # Not enough data points available
                    result.loc[date] = np.nan
            return result
                
        return wrapper
    return decorator

# Example usage functions
@broadcast
def simple_return(prices: pd.Series) -> pd.Series:
    """Calculate simple returns"""
    return prices.pct_change()

@rolling(window=20)
def mean_return(returns: pd.Series) -> float:
    """Calculate mean return over a window"""
    return returns.mean()

@grouping(groups={'tech': ['AAPL', 'MSFT'], 'finance': ['JPM', 'BAC']})
def group_mean(data: pd.DataFrame) -> pd.Series:
    """Calculate mean across group members"""
    return data.mean(axis=1)

# Example of nested decorators
@broadcast
@rolling(window=20)
def momentum(prices: pd.Series) -> float:
    """Calculate momentum as price change over window"""
    return (prices.iloc[-1] / prices.iloc[0]) - 1

@grouping(groups={'tech': ['AAPL', 'MSFT'], 'finance': ['JPM', 'BAC']})
@rolling(window=20)
def group_volatility(prices: pd.DataFrame) -> float:
    """Calculate volatility of group returns"""
    returns = prices.pct_change()
    return returns.std()