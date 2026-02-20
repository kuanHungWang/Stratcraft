from datetime import datetime
from typing import Dict, Tuple, List, Any
import pandas as pd


def access_case_insensitive(key: str, dictionary: dict[str, Any])->Any:
    """
    access value in dictionary through key in a case insensitive way.
    if cannot find even after case insensitve, raise error.
    for example, {'Close': ...}, key = 'close' or 'cLoSe' will return ...
    """
    for k in dictionary:
        if k.lower() == key.lower():
            return dictionary[k]
    raise KeyError(f"Key '{key}' not found even after case-insensitive search")

def valid_date_range(data: Dict[str, pd.Series|pd.DataFrame]) -> Tuple[datetime, datetime]:
    """
    data being a dictionary of Serie or DataFrame, which has deffirent number of
    NA value in the front and back, (the NA in the middle is not concerned here)
    Get the valid date range from the data.
    For example, 
    the value of S1 is [na, na, na, 10, 11, 12, 13, na]
    the value of S2 is [na, na, 21, 22, 23, 24, na, na]
    the index are both [d0, d1, d2, d3, d4, d5, d6, d7]
    Thus the valid date range is [d3, d5]
    """
    # Find the latest start date and earliest end date across all series/dataframes
    start_dates = []
    end_dates = []
    
    for _, df_or_series in data.items():
        # Skip empty dataframes/series
        if len(df_or_series) == 0:
            continue
            
        # Get first and last non-NA index
        if isinstance(df_or_series, pd.DataFrame):
            # For DataFrame, consider a row as NA if all values are NA
            first_valid = df_or_series.dropna(how='all').index.min()
            last_valid = df_or_series.dropna(how='all').index.max()
        else:  # Series
            first_valid = df_or_series.dropna().index.min()
            last_valid = df_or_series.dropna().index.max()
            
        start_dates.append(first_valid)
        end_dates.append(last_valid)
    
    if not start_dates or not end_dates:
        raise ValueError("No valid data found in any of the provided series/dataframes")
        
    # Return the latest start date and earliest end date
    return max(start_dates), min(end_dates)


def valid_symbol(data: Dict[str, pd.DataFrame]) -> List[str]:
    """
    data being a dictionary of DataFrames, which ideally should have the same set of columns
    However, in practice they might have different missing columns
    So this function check all columns of dataFrames as set and return the intersection.
    for example, 
    columns of df1 = {'a', 'b', 'c'}, columns of df2 = {'b', 'c', 'd'}
    thus the intersection is {'b', 'c'}
    """
    if not data:
        return []
        
    # Get sets of columns for each DataFrame
    column_sets = [set(df.columns) for df in data.values() if isinstance(df, pd.DataFrame)]
    
    if not column_sets:
        return []
        
    # Find the intersection of all column sets
    common_columns = column_sets[0]
    for column_set in column_sets[1:]:
        common_columns = common_columns.intersection(column_set)
    
    # Convert to sorted list for consistent output
    return sorted(list(common_columns))

def align_columns(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Use valid_symbol to get the intersection of columns and return a new dictionary 
    that take only common columns in all dataframes.
    """
    common_columns = valid_symbol(data)
    
    # Create a new dictionary with only the common columns
    aligned_data = {}
    for key, df in data.items():
        if isinstance(df, pd.DataFrame):
            # Only keep common columns
            aligned_data[key] = df[common_columns].copy()
        else:
            # If not a DataFrame, keep as is
            aligned_data[key] = df
    
    return aligned_data