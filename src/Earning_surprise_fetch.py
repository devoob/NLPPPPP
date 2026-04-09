import yfinance as yf
import pandas as pd
import time

# 1. Read your CSV file
# Assume the file name is volatility_cleaned.csv
df = pd.read_csv('volatility_cleaned.csv')
df['call_date'] = pd.to_datetime(df['call_date'])

# Initialize columns
df['reported_eps'] = None
df['estimated_eps'] = None
df['surprise_pct'] = None

unique_tickers = df['ticker'].unique()
total = len(unique_tickers)

print(f"Starting processing, total {total} stocks...\n" + "-"*50)

for i, ticker_symbol in enumerate(unique_tickers):
    print(f"[{i+1}/{total}] Querying: {ticker_symbol}...")
    try:
        tk = yf.Ticker(ticker_symbol)
        
        # Get historical earnings dates and surprise values
        # yfinance earnings_dates usually contains a long history
        earnings = tk.get_earnings_dates(limit=100)
        
        if earnings is not None and not earnings.empty:
            # Convert date format, remove timezone for matching
            earnings = earnings.reset_index()
            earnings['Earnings Date'] = pd.to_datetime(earnings['Earnings Date']).dt.tz_localize(None)
            
            # Find all rows of the current stock in the original dataframe
            mask = df['ticker'] == ticker_symbol
            indices = df[mask].index
            
            found_count = 0
            for idx in indices:
                target_date = df.loc[idx, 'call_date']
                
                # Calculate absolute date difference
                diff = (earnings['Earnings Date'] - target_date).dt.days.abs()
                closest_idx = diff.idxmin()
                
                # If the difference is within 4 days (considering weekends and earnings release time difference)
                if diff[closest_idx] <= 4:
                    res = earnings.iloc[closest_idx]
                    df.at[idx, 'reported_eps'] = res['Reported EPS']
                    df.at[idx, 'estimated_eps'] = res['EPS Estimate']
                    df.at[idx, 'surprise_pct'] = res['Surprise(%)']
                    
                    print(f"   ✅ Match successful | Date: {target_date.date()} | Actual: {res['Reported EPS']} | Surprise: {res['Surprise(%)']}%")
                    found_count += 1
                else:
                    print(f"   ❌ Match failed | Date: {target_date.date()} (closest earnings date: {earnings.iloc[closest_idx]['Earnings Date'].date()})")
            
            if found_count == 0:
                print(f"   ⚠️ No matches found for any dates of this stock")
        else:
            print(f"   🚫 Unable to fetch earnings table for {ticker_symbol} from yfinance")
            
    except Exception as e:
        print(f"   💥 Error: {ticker_symbol} encountered an exception - {e}")
    
    # Print separator, slight sleep to avoid high frequency
    print("-" * 30)
    time.sleep(0.8)

# 2. Save final results
df.to_csv('volatility_with_surprise.csv', index=False)
print("\n" + "="*50)
print("All tasks completed!")
print("Results saved to: volatility_with_surprise.csv")