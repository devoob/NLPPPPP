import wrds
import pandas as pd
import numpy as np

# 1. Connect to WRDS
conn = wrds.Connection()

# 2. Download Compustat annual data (only needed fields)
comp = conn.raw_sql("""
    SELECT tic, datadate, prcc_f, csho, ceq, lt, at, oiadp
    FROM comp.funda
    WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C' AND fic='USA'
""", date_cols=['datadate'])

# 3. Compute four variables
comp['mktcap'] = comp['prcc_f'] * comp['csho']
comp['bm'] = comp['ceq'] / comp['mktcap']
comp['leverage'] = comp['lt'] / comp['at']
comp['profitability'] = comp['oiadp'] / comp['at']

# Keep required columns
comp = comp[['tic', 'datadate', 'mktcap', 'bm', 'leverage', 'profitability']]

# 4. Process your event table
df_events = pd.read_csv("volatility_with_surprise.csv")  # Adjust path as needed
df_events['call_date'] = pd.to_datetime(df_events['call_date'])
df_events['tic'] = df_events['ticker'].str.upper().astype(str)

# 5. Ensure tic in comp is also string
comp['tic'] = comp['tic'].astype(str)

# 6. Sort (required for merge_asof)
df_events = df_events.sort_values('call_date')
comp = comp.sort_values('datadate')

# 7. Merge
merged = pd.merge_asof(
    df_events,
    comp,
    left_on='call_date',
    right_on='datadate',
    by='tic',
    direction='backward'
)

# ==========================================
# 8. Clean the merged DataFrame
# ==========================================
# Drop the 'realized_vol_1d' column (the third column in your original table)
merged = merged.drop(columns=['realized_vol_1d'])

# Drop any rows that contain at least one null value
merged_cleaned = merged.dropna()

# ==========================================
# 9. Save and preview results
# ==========================================
merged_cleaned.to_csv("merged_cleaned.csv", index=False)
print("Cleaned merged data saved to 'merged_cleaned.csv'")
