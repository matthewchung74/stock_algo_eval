import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from datetime import datetime

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY_ID")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# Set up Alpaca client
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Parameters
symbol = "NVDA"
timeframe = TimeFrame.Minute  # We'll filter for 5-min bars below
start_date = "2023-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
output_csv = "nvda_5min_bars.csv"

# Download 1-min bars, then resample to 5-min
request_params = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Minute,
    start=start_date,
    end=end_date
)
bars = client.get_stock_bars(request_params).df

# Just reset the index
bars = bars.reset_index()

# Resample to 5-min bars
bars['timestamp'] = pd.to_datetime(bars['timestamp'])
bars = bars.set_index('timestamp')
bars_5min = bars.resample('5T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'trade_count': 'sum',
    'vwap': 'mean'
}).dropna().reset_index()

# Filter for regular trading hours (9:30am–4:00pm US/Eastern)
# Only use tz_convert since timestamps are already tz-aware
bars_5min['timestamp'] = bars_5min['timestamp'].dt.tz_convert('America/New_York')
bars_5min = bars_5min[
    (bars_5min['timestamp'].dt.time >= pd.to_datetime('09:30:00').time()) &
    (bars_5min['timestamp'].dt.time <= pd.to_datetime('16:00:00').time())
]

# Save to CSV
bars_5min.to_csv(output_csv, index=False)
print(f"✅ Saved {len(bars_5min)} rows to {output_csv}") 