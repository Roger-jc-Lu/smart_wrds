import os
import wrds
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime
from pathlib import Path
import yfinance as yf

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

VECTOR_DB_PATH = Path("./vector_db_cache_chroma")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
COLLECTION_NAME = "stock_events_lc_hf_chroma"

_embedding_function = None
_vector_store = None

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

def initialize_embedding_model():
    global _embedding_function
    if _embedding_function is None:
        try:
            encode_kwargs = {'normalize_embeddings': True}
            _embedding_function = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                encode_kwargs=encode_kwargs
            )
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading HuggingFace embedding model '{EMBEDDING_MODEL_NAME}': {e}")
            print("Ensure 'sentence-transformers' and potentially 'torch'/'tensorflow' are installed.")
            _embedding_function = False 
    return _embedding_function if _embedding_function else None 

def initialize_vector_store():
    global _vector_store
    if not isinstance(_vector_store, Chroma): 
        embedding_func = initialize_embedding_model()
        if embedding_func:
            try:
                VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
                _vector_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=embedding_func,
                    persist_directory=str(VECTOR_DB_PATH)
                )
                # print(f"LangChain Chroma vector store initialized. Collection: '{COLLECTION_NAME}'")
            except Exception as e:
                print(f"Error initializing LangChain Chroma vector store: {e}")
                _vector_store = False 
        else:
            print("Error: Embedding function not initialized. Cannot create vector store.")
            _vector_store = False 
    return _vector_store 

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

def format_event_to_text(event_date: pd.Timestamp, event_type: str, event_value: float, ticker: str) -> str:
    date_str = event_date.strftime('%Y-%m-%d')
    if event_type == 'Dividend':
        return f"On {date_str}, {ticker} issued a dividend of ${event_value:.4f} per share."
    elif event_type == 'Split':
        return f"On {date_str}, {ticker} had a stock split with a factor of {event_value:.4f}."
    else:
        return f"On {date_str}, {ticker} had an event: Type={event_type}, Value={event_value:.4f}."
    
def save_events(ticker: str, event_df: pd.DataFrame):
    vector_store = initialize_vector_store()
    if not isinstance(vector_store, Chroma):
        print("Error: LangChain vector store not available. Cannot save events.")
        return
    if event_df is None: 
         print(f"Event data for {ticker} is None. Skipping save.")
         return
    if not event_df.empty and not isinstance(event_df.index, pd.DatetimeIndex):
         print(f"Error: Non-empty event_df for {ticker} must have a DatetimeIndex.")
         return

    # print(f"Processing and saving events for {ticker} to ChromaDB vector store...")
    documents_to_add = []
    ids_to_add = [] 

    if not event_df.empty:
        for event_date, row in event_df.iterrows():
            event_timestamp = event_date.timestamp()

            # Process Dividend
            if 'Dividends' in row and row['Dividends'] > 0:
                event_type = 'Dividend'
                event_value = row['Dividends']
                text = format_event_to_text(event_date, event_type, event_value, ticker)
                event_id = f"{ticker}_{event_date.strftime('%Y%m%d')}_{event_type}" # Unique ID
                metadata = {
                    "ticker": ticker,
                    "date_str": event_date.strftime('%Y-%m-%d'),
                    "timestamp": event_timestamp,
                    "event_type": event_type,
                    "value": float(event_value),
                    "source": "yfinance"
                }
                doc = Document(page_content=text, metadata=metadata)
                documents_to_add.append(doc)
                ids_to_add.append(event_id)

            # Process Split
            if 'Stock Splits' in row and row['Stock Splits'] > 0:
                event_type = 'Split'
                event_value = row['Stock Splits']
                text = format_event_to_text(event_date, event_type, event_value, ticker)
                event_id = f"{ticker}_{event_date.strftime('%Y%m%d')}_{event_type}" # Unique ID
                metadata = {
                    "ticker": ticker,
                    "date_str": event_date.strftime('%Y-%m-%d'),
                    "timestamp": event_timestamp,
                    "event_type": event_type,
                    "value": float(event_value),
                    "source": "yfinance"
                }
                doc = Document(page_content=text, metadata=metadata)
                documents_to_add.append(doc)
                ids_to_add.append(event_id)

    if not documents_to_add:
        # This case handles when the input event_df was empty or had no valid events
        print(f"No valid events found or processed for {ticker}. No changes made to vector store for this ticker.")
        # Note: This doesn't delete existing entries for the ticker if the new fetch is empty.
        # Add deletion logic here if required (e.g., get all IDs for the ticker and delete).
        return

    try:
        vector_store.add_documents(documents=documents_to_add, ids=ids_to_add)
        # Persisting is generally handled automatically by ChromaDB in persistent mode,
        # but calling it explicitly ensures writes are flushed if needed.
        vector_store.persist()
        print(f"Successfully saved/updated {len(documents_to_add)} events for {ticker} in ChromaDB vector store.")

    except Exception as e:
        print(f"Error adding event data for {ticker} to ChromaDB vector store: {e}")

def load_events(ticker: str) -> pd.DataFrame | None:
    vector_store = initialize_vector_store()
    if not isinstance(vector_store, Chroma):
        print("Error: LangChain vector store not available. Cannot load events.")
        return None

    # print(f"Querying ChromaDB vector store for events for {ticker} between {start_date} and {end_date}...")

    try:
        results = vector_store.get(
            where={
                "$and": [
                    {"ticker": ticker}
                ]
            },
            include=["metadatas"] # Only need metadata
        )

        if not results or not results.get('ids'):
            print(f"No events found for {ticker} in the specified date range in ChromaDB vector store.")
            return pd.DataFrame(columns=['Dividends', 'Stock Splits'], index=pd.DatetimeIndex([]))

        event_data = []
        metadatas = results['metadatas']

        for meta in metadatas:
             if not meta: continue
             event_date = pd.to_datetime(meta.get('date_str'))
             event_type = meta.get('event_type')
             value = meta.get('value')

             if event_date is None or event_type is None or value is None:
                 print(f"Warning: Skipping record with incomplete metadata: {meta}")
                 continue

             data_row = {'Date': event_date}
             if event_type == 'Dividend':
                 data_row['Dividends'] = value
                 data_row['Stock Splits'] = 0
             elif event_type == 'Split':
                 data_row['Dividends'] = 0
                 data_row['Stock Splits'] = value
             else:
                 print(f"Warning: Skipping record with unexpected event type: {event_type}")
                 continue
             event_data.append(data_row)

        if not event_data:
             print(f"No valid event data reconstructed for {ticker}.")
             return pd.DataFrame(columns=['Dividends', 'Stock Splits'], index=pd.DatetimeIndex([]))

        temp_df = pd.DataFrame(event_data)
        reconstructed_df = temp_df.groupby('Date').agg(
             Dividends=('Dividends', 'sum'),
             Stock_Splits=('Stock Splits', 'sum')
        ).rename(columns={'Stock_Splits': 'Stock Splits'})

        reconstructed_df.index = pd.to_datetime(reconstructed_df.index)
        reconstructed_df = reconstructed_df.sort_index()

        print(f"Successfully loaded and reconstructed {len(reconstructed_df)} event records for {ticker} from ChromaDB vector store.")
        return reconstructed_df

    except Exception as e:
        print(f"Error querying or processing event data for {ticker} from ChromaDB vector store: {e}")
        return None

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
    initialize_vector_store()
    raw_df = get_multi_month_data(db, ticker, start_date, end_date, granularity)

    if not isinstance(_vector_store, Chroma): 
         print("Error: Critical components (ChromaDB or Embedding Model) failed to initialize. Returning raw_df.")
         return raw_df 

    if raw_df is None or raw_df.empty:
         print(f"Error: Failed to retrieve raw OHLC data for {ticker}. Cannot adjust.")
         return raw_df 

    event_df = load_events(ticker)

    fetch_fresh = False
    if event_df is None:
         print("Failed to load suitable events from ChromaDB vector store.")
         fetch_fresh = True
    # Add more sophisticated logic here if needed (e.g., check latest timestamp in DB for the ticker)

    if fetch_fresh:
        print(f"Fetching fresh event data for {ticker} from yfinance...")
        try:
            fresh_event_df = get_events(ticker) 
            if fresh_event_df is not None:
                 save_events(ticker, fresh_event_df)
                 event_df = fresh_event_df 
            else: 
                 print(f"Warning: Failed to fetch event data for {ticker} from yfinance.")
                 if event_df is None: 
                      event_df = pd.DataFrame(columns=['Dividends', 'Stock Splits'], index=pd.DatetimeIndex([]))

        except Exception as e:
            print(f"Error fetching or saving event data for {ticker}: {e}.")
            if event_df is None: 
                 event_df = pd.DataFrame(columns=['Dividends', 'Stock Splits'], index=pd.DatetimeIndex([]))


    if event_df.empty:
        print("Error: Event data is None after attempting load and fetch. Cannot perform adjustments.")
        adjusted_df = raw_df.copy()
    else:
        adjusted_df = adjust_ohlc(raw_df.copy(), event_df)


    return adjusted_df