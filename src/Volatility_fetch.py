import yfinance as yf
import pandas as pd
import time
import random
import numpy as np
import re
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import pickle

# ==========================================
# 1. Configuration
# ==========================================
TICKER_LIST_CSV = "data/processed/call_level_features.csv"   # Input file, must contain ticker, call_date
OUTPUT_DIR = Path("./realized_vol_results")    # Output directory for results
CHUNK_SIZE = 50                                # Number of tickers per batch
DELAY_BETWEEN_CHUNKS = 2                       # Delay between batches (seconds)
USE_CACHE = True                               # Whether to use local cache
CACHE_DIR = Path("./price_cache")              # Cache directory

# Event study parameters
PRE_WINDOW = 30            # Pre-event window length (trading days), used to determine data range (not directly used for vol calculation)
POST_WINDOWS = [1, 5, 21]  # Post-event window lengths (trading days)
ANNUALIZE = False          # Whether to annualize volatility (False: output daily vol; True: multiply by sqrt(252))

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if USE_CACHE:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Helper functions (download, cache, cleaning)
# ==========================================
def safe_download(ticker, start_date, end_date, max_retries=3):
    """Download data with retries and exponential backoff"""
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        reraise=True
    )
    def _download():
        time.sleep(random.uniform(0.1, 0.5))
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise Exception(f"No data returned for {ticker}")
        return data

    try:
        return _download()
    except Exception as e:
        print(f"❌ Failed to download {ticker} after {max_retries} attempts: {e}")
        return None

def load_from_cache(ticker, start_date, end_date):
    if not USE_CACHE:
        return None
    cache_key = f"{ticker}_{start_date}_{end_date}.pkl"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_cache(ticker, start_date, end_date, data):
    if not USE_CACHE or data is None:
        return
    cache_key = f"{ticker}_{start_date}_{end_date}.pkl"
    cache_path = CACHE_DIR / cache_key
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def extract_number(cell):
    """Extract a float number from strings like 'Ticker\nA 0.015767\ndtype: float64'"""
    if pd.isna(cell) or not isinstance(cell, str):
        return cell
    match = re.search(r'(\d+\.\d+)', cell)
    return float(match.group(1)) if match else np.nan

def clean_results(df):
    """Clean the results DataFrame: extract numeric values from string columns and drop the 1-day vol column."""
    for col in ['realized_vol_5d', 'realized_vol_21d']:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)
    # Drop the 1-day volatility column because it is always NaN (needs at least 2 returns)
    if 'realized_vol_1d' in df.columns:
        df = df.drop(columns=['realized_vol_1d'])
    return df

# ==========================================
# 3. Compute realized volatility for a single event window (based on daily returns)
# ==========================================
def compute_window_realized_vol(price_series, call_date, post_windows, annualize=False):
    """
    Compute realized volatility (standard deviation of daily returns) for post-event windows.
    
    Parameters:
        price_series: pandas Series, index = dates, values = adjusted close prices
        call_date: earnings call date
        post_windows: list of post-event window lengths (trading days)
        annualize: whether to annualize (multiply by sqrt(252))
    
    Returns:
        dict: {window: realized_volatility}
    """
    call_dt = pd.to_datetime(call_date)
    
    # Get price data after the event date (starting from the event day or first trading day after)
    future_prices = price_series[price_series.index >= call_dt]
    if len(future_prices) < 2:
        return {w: np.nan for w in post_windows}
    
    # Calculate daily log returns
    log_returns = np.log(future_prices / future_prices.shift(1)).dropna()
    
    results = {}
    for w in post_windows:
        if w <= 1:
            # 1-day window cannot compute standard deviation (need at least 2 return points)
            results[w] = np.nan
            continue
        
        # Take the first w trading days (need at least w return points)
        if len(log_returns) < w:
            results[w] = np.nan
            continue
        
        # Take the first w returns (corresponding to window [1, w] days)
        window_returns = log_returns.iloc[:w]
        # Compute sample standard deviation (ddof=1)
        vol = window_returns.std(ddof=1)
        if annualize:
            vol = vol * np.sqrt(252)
        results[w] = vol
    
    return results

# ==========================================
# 4. Main program
# ==========================================
def main():
    print("📂 Reading event CSV...")
    events_df = pd.read_csv(TICKER_LIST_CSV)
    events_df['call_date'] = pd.to_datetime(events_df['call_date'], format='%Y%m%d')
    
    # Get all call dates for each ticker
    ticker_groups = events_df.groupby('ticker')['call_date'].apply(list).to_dict()
    unique_tickers = list(ticker_groups.keys())
    print(f"📊 Total unique tickers: {len(unique_tickers)}, total events: {len(events_df)}")
    
    # Process in chunks
    ticker_chunks = [unique_tickers[i:i+CHUNK_SIZE] for i in range(0, len(unique_tickers), CHUNK_SIZE)]
    
    # Store results for each event
    all_results = []
    
    max_post_window = max(POST_WINDOWS)   # 21
    
    for chunk_idx, ticker_chunk in enumerate(ticker_chunks):
        print(f"\n🔄 Processing chunk {chunk_idx+1}/{len(ticker_chunks)} with {len(ticker_chunk)} tickers...")
        
        for ticker in ticker_chunk:
            earnings_dates = ticker_groups[ticker]
            # Data download range: earliest call_date - PRE_WINDOW*2 to latest call_date + max_post_window*2
            min_date = min(earnings_dates) - pd.Timedelta(days=PRE_WINDOW*2)
            max_date = max(earnings_dates) + pd.Timedelta(days=max_post_window*2)
            start_str = min_date.strftime("%Y-%m-%d")
            end_str = max_date.strftime("%Y-%m-%d")
            
            # Try to load from cache
            cached_data = load_from_cache(ticker, start_str, end_str)
            if cached_data is not None:
                price_df = cached_data
                print(f"  ✅ {ticker} loaded from cache")
            else:
                print(f"  📥 Downloading {ticker} ({start_str} to {end_str})...")
                price_df = safe_download(ticker, start_str, end_str)
                if price_df is None or price_df.empty:
                    print(f"  ❌ {ticker} download failed, skipping")
                    continue
                save_to_cache(ticker, start_str, end_str, price_df)
                print(f"  ✅ {ticker} downloaded successfully, {len(price_df)} records")
            
            # Extract adjusted close price series
            adj_col = 'Adj Close' if 'Adj Close' in price_df.columns else 'Close'
            price_series = price_df[adj_col].copy()
            price_series = price_series.sort_index()
            
            # Compute window realized volatility for each event of this ticker
            for call_date in earnings_dates:
                if call_date < price_series.index.min() or call_date > price_series.index.max():
                    print(f"  ⚠️ {ticker} event {call_date.date()} out of price range, skipping")
                    continue
                
                vols = compute_window_realized_vol(price_series, call_date, POST_WINDOWS, annualize=ANNUALIZE)
                record = {
                    'ticker': ticker,
                    'call_date': call_date,
                }
                for w, vol in vols.items():
                    record[f'realized_vol_{w}d'] = vol
                all_results.append(record)
        
        # Delay between chunks
        if chunk_idx < len(ticker_chunks) - 1:
            print(f"  ⏳ Waiting {DELAY_BETWEEN_CHUNKS} seconds before next chunk...")
            time.sleep(DELAY_BETWEEN_CHUNKS)
    
    # ==========================================
    # 5. Save results (with cleaning)
    # ==========================================
    results_df = pd.DataFrame(all_results)
    
    # Apply cleaning: extract numbers from problematic string columns and drop the 1-day vol column
    results_df = clean_results(results_df)
    
    output_csv = OUTPUT_DIR / "event_realized_volatility.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Computation completed! Processed {len(results_df)} events.")
    print(f"📄 Cleaned results saved to {output_csv}")
    

if __name__ == "__main__":
    results = main()