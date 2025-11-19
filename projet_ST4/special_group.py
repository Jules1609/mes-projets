import warnings
import pandas as pd
import numpy as np

def special_group(table: pd.DataFrame, bytime: bool=False, group_amount: int=0, time_interval: str='minute') -> pd.DataFrame:
    """
    Groups the DataFrame into special groups based on the 'group_amount' parameter.
    
    Parameters:
    - table: pd.DataFrame - The input DataFrame to be grouped.
    - group_amount: int - The number of groups to create.
    
    Returns:
    - pd.DataFrame - A DataFrame with 2 additional columns : returns and deltaT, all volumes are summed when grouped but the mid_prices are averaged.
    """
    # Validate group_amount
    if bytime is True:
        assert group_amount == 0, "group_amount must be 0 when bytime is True"
    elif group_amount <= 0:
        raise ValueError("group_amount must be a positive integer")
    
    supported_intervals = ['minute']
    if time_interval not in supported_intervals:
        raise ValueError(f"time_interval must be one of {supported_intervals}")
    
    # Check if the DataFrame is empty
    if table.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in table.columns:
        table = table.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Ensure the DataFrame has the required columns
    required_columns = ['mid_price', 'V_lo_b', 'V_c_b', 'V_ex_b', 'V_lo_a', 'V_c_a', 'V_ex_a']
    if not all(col in table.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain the following columns: {required_columns}")
    if any(col not in required_columns for col in table.columns if col not in ['date', 'time', 'datetime']):
        for col in table.columns:
            if col not in required_columns and col not in ['date', 'time', 'datetime']:
                warnings.warn(f"Unexpected column found: {col}")
    
    # Ensure 'source_file' column is not present
    if 'source_file' in table.columns:
        table = table.drop(columns=['source_file']).reset_index(drop=True)
    
    # Ensure 'datetime' column is present or create it from 'date' and 'time'
    if 'datetime' in table.columns:
        table['datetime'] = pd.to_datetime(table['datetime'])
    elif 'date' in table.columns or 'time' in table.columns:
        assert 'time' in table.columns, "Both 'date' and 'time' columns must be present to create 'datetime'"
        assert 'date' in table.columns, "Both 'date' and 'time' columns must be present to create 'datetime'"
        table['datetime'] = pd.to_datetime(table['date'] + ' ' + table['time'])
        table = table.drop(columns=['date', 'time'])
    else:
        raise ValueError("DataFrame must have either 'datetime' column or both 'date' and 'time' columns")
    
    # Create a new DataFrame with grouped data
    if not bytime:
        grouped = table.groupby(np.arange(len(table)) // group_amount).agg({
        'datetime': 'first',
        'V_lo_b': 'sum',
        'V_c_b': 'sum',
        'V_ex_b': 'sum',
        'V_lo_a': 'sum',
        'V_c_a': 'sum',
        'V_ex_a': 'sum',
        'mid_price': 'mean'
        }).reset_index(drop=True)
    elif time_interval == 'minute':
        table['datetime'] = table['datetime'].dt.floor('min')
        grouped = table.groupby(table['datetime']).agg({
            'V_lo_b': 'sum',
            'V_c_b': 'sum',
            'V_ex_b': 'sum',
            'V_lo_a': 'sum',
            'V_c_a': 'sum',
            'V_ex_a': 'sum',
            'mid_price': 'mean'
        }).reset_index()
    else:
        raise ValueError("Unsupported time_interval. Currently only 'minute' is supported when bytime is True.")       

    # Calculate log returns
    grouped['returns'] = np.log(grouped['mid_price'] / grouped['mid_price'].shift(1)).fillna(0)
    if not bytime:
        grouped['deltaT'] = grouped['datetime'].diff().dt.total_seconds().astype(np.float64).fillna(0.0)
    
    grouped_cut = grouped.iloc[2:].reset_index(drop=True)  # Remove the 2 first rows
    
    return grouped_cut
