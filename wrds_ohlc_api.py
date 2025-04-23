import os
import wrds
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime
import yfinance as yf

def init_db_connection(usr_name, password):
    """
    Initialize a connection to the WRDS database.
    """
    os.environ['PGHOST'] = 'wrds-pgdata.wharton.upenn.edu'
    os.environ['PGPORT'] = '9737'
    os.environ['PGDATABASE'] = 'wrds'
    os.environ['PGUSER'] = usr_name                    
    os.environ['PGPASSWORD'] = password           
    db = wrds.Connection()
    return db

def parse_granularity(gran_str):
    """Convert a granularity string to seconds.
       Valid inputs: 1s, 5s, 15s, 30s, 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d.
    """
    unit = gran_str[-1]
    value = int(gran_str[:-1])
    if unit == 's':
        return value
    elif unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    else:
        raise ValueError("Invalid granularity")

def iterate_months(start_date, end_date):
    months = []
    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        months.append((current.year, current.month))
        current = next_month
    return months

def get_ohlc_data(db, ticker, year, granularity, start_date=None, end_date=None):
    if start_date is None:
        start_date = date(year, 1, 1)
    if end_date is None:
        end_date = date(year, 12, 31)

    gran_seconds = parse_granularity(granularity)
    
    query = f"""
        WITH base AS (
            SELECT
                (date + time_m) AS ts,
                ROUND((best_bid + best_ask) / 2, 2) AS price
            FROM
                taqm_{year}.complete_nbbo_{year}
            WHERE
                sym_root = '{ticker}'
                AND date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                AND time_m >= '09:30:00'
                AND time_m <= '16:00:00'
        ),
        intervals AS (
            SELECT
                ts,
                price,
                timestamp 'epoch' + floor(extract(epoch from ts) / {gran_seconds}) * interval '{gran_seconds} second' AS interval_start
            FROM base
        )
        SELECT
            interval_start AS timestamp,
            (ARRAY_AGG(price ORDER BY ts) FILTER (WHERE price IS NOT NULL))[1] AS open,
            MAX(price) AS high,
            MIN(price) AS low,
            (ARRAY_AGG(price ORDER BY ts DESC) FILTER (WHERE price IS NOT NULL))[1] AS close
        FROM intervals
        GROUP BY interval_start
        ORDER BY interval_start
    """
    df = db.raw_sql(query)
    return df

def get_multi_month_data(db, ticker, start_date, end_date, granularity):
    months = iterate_months(start_date, end_date)
    dataframes = []

    def task_for_month(year, month):
        # Get start and end date of this month
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        # Clip to global start and end
        start = max(start, start_date)
        end = min(end, end_date)
        return get_ohlc_data(db, ticker, year, granularity, start_date=start, end_date=end)

    with ThreadPoolExecutor(max_workers=min(len(months), 12)) as executor:
        futures = {executor.submit(task_for_month, y, m): (y, m) for (y, m) in months}

        for future in futures:
            y, m = futures[future]
            try:
                df_month = future.result()
                dataframes.append(df_month)
                print(f"Data for {y}-{m:02d} retrieved successfully.")
            except Exception as e:
                print(f"Error retrieving data for {y}-{m:02d}: {e}")

    final_df = pd.concat(dataframes).sort_values(by='timestamp').reset_index(drop=True)
    final_df["ticker"] = ticker
    return final_df

def get_events(ticker):
    df = yf.download(ticker,period="max",actions=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    events = df[(df['Dividends'] > 0) | (df['Stock Splits'] > 0)]
    events = events[['Dividends', 'Stock Splits']]
    events["ticker"] = ticker
    return events

def adjust_ohlc(raw_df, event_df):
    event_df = event_df.copy()

    event_df['split_factor'] = (1 / event_df['Stock Splits'].replace(0, 1))[::-1].cumprod()[::-1]
    event_df['div_cumsum'] = event_df['Dividends'][::-1].cumsum()[::-1]

    # Convert to numpy for fast indexing
    event_dates = event_df.index.to_numpy()
    split_factor_arr = event_df['split_factor'].to_numpy()
    div_adjust_arr = event_df['div_cumsum'].to_numpy()

    # Align each raw timestamp to its corresponding future event
    raw_times = raw_df['timestamp'].dt.floor('D').to_numpy()
    event_idx = np.searchsorted(event_dates, raw_times, side='right')
    event_idx = np.clip(event_idx, 0, len(event_df) - 1)

    # Get corresponding adjustments
    sf = split_factor_arr[event_idx]
    da = div_adjust_arr[event_idx]

    # Adjust each OHLC field
    raw_df['open']  = raw_df['open']  * sf - da
    raw_df['high']  = raw_df['high']  * sf - da
    raw_df['low']   = raw_df['low']   * sf - da
    raw_df['close'] = raw_df['close'] * sf - da

    ohlc_cols = ['open', 'high', 'low', 'close']
    raw_df[ohlc_cols] = raw_df[ohlc_cols].round(2)

    return raw_df

def get_adjusted_ohlc(db, ticker, start_date, end_date, granularity):
    raw_df = get_multi_month_data(db, ticker, start_date, end_date, granularity)
    
    # Get event data
    event_df = get_events(ticker)
    
    # Adjust OHLC data
    adjusted_df = adjust_ohlc(raw_df, event_df)
    
    return adjusted_df