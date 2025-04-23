# data_processor.py
# (Or rename data_storage.py and use this content)

import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import hashlib

# --- Configuration (Optional Caching for Processed Data) ---
CACHE_DIR = Path("./data_cache") # Base directory for storing cached files
ADJUSTED_OHLC_CACHE_DIR = CACHE_DIR / "adjusted_ohlc"
DEFAULT_CACHE_EXPIRY_SECONDS = 24 * 60 * 60 # Cache expires after 1 day (adjust as needed)

# Ensure cache directory exists
ADJUSTED_OHLC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Cache Helper Functions (Similar to before, but for processed data) ---

def _get_cache_filepath(identifier: str, cache_subdir: Path, format: str = "parquet") -> Path:
    """Constructs the filepath for a cache file."""
    filename = f"{identifier}.{format}"
    return cache_subdir / filename

def _save_to_cache(df: pd.DataFrame, filepath: Path):
    """Saves a DataFrame to a cache file (using Parquet)."""
    try:
        # Ensure the directory exists before saving
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=True)
        print(f"Processed data saved to cache: {filepath}")
    except Exception as e:
        print(f"Error saving processed data to cache ({filepath}): {e}")

def _load_from_cache(filepath: Path, expiry_seconds: int = DEFAULT_CACHE_EXPIRY_SECONDS) -> pd.DataFrame | None:
    """Loads a DataFrame from cache if it exists and hasn't expired."""
    if not filepath.exists():
        return None
    try:
        file_mod_time = filepath.stat().st_mtime
        if (time.time() - file_mod_time) > expiry_seconds:
            print(f"Processed data cache file expired: {filepath}")
            return None
        df = pd.read_parquet(filepath)
        print(f"Processed data loaded from cache: {filepath}")
        return df
    except Exception as e:
        print(f"Error loading processed data from cache ({filepath}): {e}")
        return None

def _generate_cache_identifier(ohlc_df: pd.DataFrame, actions_df: pd.DataFrame) -> str:
    """Generates a unique identifier based on the input dataframes for caching."""
    # Use hashing on a representation of the dataframes to create a consistent key
    # Using head/tail and shape might be sufficient and faster than hashing the whole df
    ohlc_repr = f"{ohlc_df.shape}-{ohlc_df.head(1).to_string()}-{ohlc_df.tail(1).to_string()}"
    actions_repr = f"{actions_df.shape}-{actions_df.head(1).to_string()}-{actions_df.tail(1).to_string()}"
    combined_repr = f"{ohlc_repr}-{actions_repr}"
    return hashlib.md5(combined_repr.encode()).hexdigest()


# --- Core Processing Function ---

def calculate_adjusted_ohlc(ohlc_df: pd.DataFrame,
                            actions_df: pd.DataFrame,
                            use_cache: bool = True) -> pd.DataFrame | None:
    """
    Adjusts the OHLC data for stock splits and dividends.

    Args:
        ohlc_df (pd.DataFrame): DataFrame with raw 'Open', 'High', 'Low', 'Close', 'Volume'.
                                Must have a DatetimeIndex.
        actions_df (pd.DataFrame): DataFrame with 'Dividends' and 'Stock Splits'.
                                   Must have a DatetimeIndex.
                                   'Stock Splits' should be the factor (e.g., 2.0 for 2-for-1).
                                   Zeros indicate no action on that date.
        use_cache (bool): Whether to attempt loading/saving from/to cache.

    Returns:
        pd.DataFrame | None: DataFrame with adjusted OHLCV columns ('Adj Open',
                              'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume')
                              or None if input is invalid.
    """
    # --- Input Validation ---
    if not isinstance(ohlc_df, pd.DataFrame) or ohlc_df.empty:
        print("Error: Input ohlc_df is invalid or empty.")
        return None
    if not isinstance(actions_df, pd.DataFrame): # actions_df can be empty if no actions
        print("Error: Input actions_df is invalid.")
        return None
    if not isinstance(ohlc_df.index, pd.DatetimeIndex):
        print("Error: ohlc_df must have a DatetimeIndex.")
        return None
     # Allow actions_df to be empty or have non-datetime index initially, will align later
    # if not isinstance(actions_df.index, pd.DatetimeIndex) and not actions_df.empty :
    #     print("Error: actions_df must have a DatetimeIndex if not empty.")
    #     return None

    required_ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in ohlc_df.columns for col in required_ohlc_cols):
        print(f"Error: ohlc_df must contain columns: {required_ohlc_cols}")
        return None

    # --- Caching Logic ---
    cache_identifier = "default_id" # Placeholder
    cache_filepath = None
    if use_cache:
        try:
            # Generate identifier based on input data
            cache_identifier = _generate_cache_identifier(ohlc_df, actions_df)
            cache_filepath = _get_cache_filepath(cache_identifier, ADJUSTED_OHLC_CACHE_DIR)
            cached_result = _load_from_cache(cache_filepath)
            if cached_result is not None:
                return cached_result
        except Exception as e:
            print(f"Warning: Cache identifier generation or loading failed: {e}")
            # Continue without cache if identifier generation fails

    print("Calculating adjusted OHLC data...")

    # --- Processing ---
    adj_df = ohlc_df.copy()

    # Ensure actions_df index is datetime if not empty
    if not actions_df.empty and not isinstance(actions_df.index, pd.DatetimeIndex):
         try:
             actions_df.index = pd.to_datetime(actions_df.index)
         except Exception as e:
             print(f"Error converting actions_df index to DatetimeIndex: {e}")
             return None


    # Align actions_df to ohlc_df index, filling missing dates with 0
    actions_aligned = actions_df.reindex(adj_df.index, fill_value=0)

    # --- FIX: Initialize adjusted columns and explicitly cast to float ---
    adj_df['Adj Open'] = adj_df['Open'].astype(float)
    adj_df['Adj High'] = adj_df['High'].astype(float)
    adj_df['Adj Low'] = adj_df['Low'].astype(float)
    adj_df['Adj Close'] = adj_df['Close'].astype(float)
    adj_df['Adj Volume'] = adj_df['Volume'].astype(float) # Keep volume as float too

    # Cumulative adjustment factor, start with 1.0
    cum_adj_factor = 1.0

    # Iterate backwards through time (from most recent to oldest)
    for date in reversed(adj_df.index):
        # Apply the current cumulative factor first
        # Now multiplying float columns by a float, no warning expected
        adj_df.loc[date, ['Adj Open', 'Adj High', 'Adj Low', 'Adj Close']] *= cum_adj_factor

        # Check for actions on this date (or technically, effective before market open *this* date)
        dividend = actions_aligned.loc[date, 'Dividends'] if 'Dividends' in actions_aligned.columns else 0
        split_factor = actions_aligned.loc[date, 'Stock Splits'] if 'Stock Splits' in actions_aligned.columns else 0

        # Update cumulative factor for dates *before* this one

        # 1. Dividend Adjustment
        if dividend > 0:
            # Price factor = (1 - dividend / close_price_before_dividend)
            # Use the close price of the *current* day before adjustment factor was applied
            # Need the price *before* the `*= cum_adj_factor` line for this date
            # Let's calculate the 'unadjusted' close for this specific calculation
            unadj_close_for_div_calc = adj_df.loc[date, 'Adj Close'] / cum_adj_factor
            if unadj_close_for_div_calc > 0: # Avoid division by zero
                 dividend_adj_factor = 1.0 - (dividend / unadj_close_for_div_calc)
                 cum_adj_factor *= dividend_adj_factor
            else:
                 print(f"Warning: Skipping dividend adjustment on {date} due to zero closing price before adjustment.")


        # 2. Split Adjustment
        if split_factor > 0:
            # Price factor = 1 / split_factor
            # Volume factor = split_factor
            split_price_adj_factor = 1.0 / split_factor
            cum_adj_factor *= split_price_adj_factor
            # Volume adjustment is handled after the loop


    # --- Post-Loop Volume Adjustment for Splits ---
    # Create a cumulative split factor series going forward in time
    adj_df['Split Factor'] = actions_aligned['Stock Splits'].replace(0, 1.0) # Replace 0 splits with 1
    # Calculate cumulative product of split factors *backward* in time, then reverse it
    cum_split_factor_backward = adj_df['Split Factor'].iloc[::-1].cumprod().iloc[::-1]
    # The factor to multiply volume by is the total split ratio from that day forward
    # Shift ensures the factor applies to dates *before* the split date
    volume_adj_factor = cum_split_factor_backward.shift(-1, fill_value=1.0)

    # Apply the volume adjustment factor
    adj_df['Adj Volume'] = adj_df['Volume'].astype(float) * volume_adj_factor


    # Clean up temporary column
    adj_df = adj_df.drop(columns=['Split Factor'])


    # --- Save to Cache ---
    if use_cache and cache_filepath:
        _save_to_cache(adj_df, cache_filepath)

    print("Finished calculating adjusted OHLC data.")
    return adj_df

# --- Example Usage ---
if __name__ == '__main__':
    # Create dummy data for demonstration
    dates = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09'])
    # Make input data integer type to replicate the warning scenario
    ohlc_data = {
        'Open': [100, 102, 105, 106, 108], # Raw prices (using integers post-split for demo)
        'High': [103, 106, 107, 110, 112],
        'Low': [99, 101, 104, 104, 106],
        'Close': [102, 105, 106, 108, 110],
        'Volume': [1000, 1100, 1200, 2300, 2400] # Raw volume
    }
    raw_ohlc_df = pd.DataFrame(ohlc_data, index=dates)

    action_data = {
        'Dividends': [0, 0, 0.5, 0, 0], # Dividend on Jan 5th (ex-date)
        'Stock Splits': [0, 0, 0, 2.0, 0]  # 2-for-1 split on Jan 6th (ex-date) -> affects prices before Jan 6
    }
    actions_df = pd.DataFrame(action_data, index=dates)

    print("--- Raw OHLC Data (Integers) ---")
    print(raw_ohlc_df)
    print(raw_ohlc_df.dtypes) # Show dtypes
    print("\n--- Actions Data ---")
    print(actions_df)

    # Calculate adjusted data
    adjusted_df = calculate_adjusted_ohlc(raw_ohlc_df, actions_df, use_cache=False) # Disable cache for demo run

    if adjusted_df is not None:
        print("\n--- Adjusted OHLC Data ---")
        # Display relevant columns side-by-side for comparison
        print(adjusted_df[['Close', 'Adj Close', 'Volume', 'Adj Volume']])
        print(adjusted_df.dtypes) # Show dtypes (Adj cols should be float)


    # Example demonstrating cache
    # print("\n--- Running again (should use cache if enabled) ---")
    # Note: For cache to work, run the script twice.
    # In a real app, cache would persist between runs.
    # adjusted_df_cached = calculate_adjusted_ohlc(raw_ohlc_df, actions_df, use_cache=True)
    # if adjusted_df_cached is not None:
    #     print("Successfully loaded/calculated:")
    #     print(adjusted_df_cached[['Close', 'Adj Close', 'Volume', 'Adj Volume']].head(2))

