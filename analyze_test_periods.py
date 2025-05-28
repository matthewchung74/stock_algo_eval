import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

# Configuration
TEST_SAMPLES = 36
NUM_PERIODS_TO_ANALYZE = 100  # Analyze last 100 possible test periods

print("🔍 Analyzing volatility across different test periods...")
print("Dataset: matthewchung74/nvda_5_min_bars")

# Load the NVDA dataset
dataset = load_dataset("matthewchung74/nvda_5_min_bars")
df_raw = dataset['train'].to_pandas()

# Filter by ET trading hours
temp_et_timestamps = pd.to_datetime(df_raw['timestamp']).dt.tz_convert('America/New_York')
et_start_time = pd.to_datetime('09:30:00').time()
et_end_time = pd.to_datetime('16:00:00').time()
trading_hours_mask = (temp_et_timestamps.dt.time >= et_start_time) & \
                     (temp_et_timestamps.dt.time <= et_end_time)
df_raw = df_raw[trading_hours_mask].copy()

# Prepare data
df = pd.DataFrame()
df['ds'] = pd.to_datetime(df_raw['timestamp']).dt.tz_localize(None)
df['price'] = df_raw['close']

print(f"Total samples available: {len(df)}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

# Analyze different test periods
results = []

for i in range(NUM_PERIODS_TO_ANALYZE):
    # Calculate start and end indices for this test period
    end_idx = len(df) - i
    start_idx = end_idx - TEST_SAMPLES
    
    if start_idx < 0:
        break
    
    # Extract test period data
    test_period = df.iloc[start_idx:end_idx].copy()
    
    # Calculate volatility metrics
    prices = test_period['price'].values
    price_range = prices.max() - prices.min()
    price_std = np.std(prices)
    price_mean = np.mean(prices)
    cv = price_std / price_mean * 100  # Coefficient of variation
    
    # Calculate returns volatility
    returns = np.diff(prices) / prices[:-1]
    returns_std = np.std(returns) * 100  # Percentage
    
    # Store results
    results.append({
        'period_id': i,
        'start_date': test_period['ds'].iloc[0],
        'end_date': test_period['ds'].iloc[-1],
        'start_idx': start_idx,
        'end_idx': end_idx,
        'price_range': price_range,
        'price_std': price_std,
        'price_mean': price_mean,
        'cv': cv,
        'returns_std': returns_std,
        'min_price': prices.min(),
        'max_price': prices.max()
    })

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

print(f"\n📊 Analyzed {len(results_df)} different test periods")
print("\n🏆 TOP 10 MOST VOLATILE TEST PERIODS (by price range):")
print("="*80)

top_volatile = results_df.nlargest(10, 'price_range')
for idx, row in top_volatile.iterrows():
    print(f"Rank {idx+1}: Period {row['period_id']} | Range: ${row['price_range']:.2f} | Std: ${row['price_std']:.3f}")
    print(f"   Date: {row['start_date']} to {row['end_date']}")
    print(f"   Price: ${row['min_price']:.2f} - ${row['max_price']:.2f} | Returns Std: {row['returns_std']:.3f}%")
    print()

print("\n📉 CURRENT TEST PERIOD (Period 0 - most recent):")
current = results_df.iloc[0]
print(f"   Range: ${current['price_range']:.2f} | Std: ${current['price_std']:.3f}")
print(f"   Date: {current['start_date']} to {current['end_date']}")
print(f"   Price: ${current['min_price']:.2f} - ${current['max_price']:.2f}")
print(f"   Returns Std: {current['returns_std']:.3f}%")

# Find current period's rank
current_rank = results_df['price_range'].rank(ascending=False).iloc[0]
print(f"   Volatility Rank: {current_rank:.0f} out of {len(results_df)} periods")

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Price range over time
plt.subplot(2, 2, 1)
plt.plot(results_df['period_id'], results_df['price_range'], 'b-', alpha=0.7)
plt.scatter(0, current['price_range'], color='red', s=100, zorder=5, label='Current Period')
plt.xlabel('Periods Ago')
plt.ylabel('Price Range ($)')
plt.title('Price Range Across Different Test Periods')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Returns volatility
plt.subplot(2, 2, 2)
plt.plot(results_df['period_id'], results_df['returns_std'], 'g-', alpha=0.7)
plt.scatter(0, current['returns_std'], color='red', s=100, zorder=5, label='Current Period')
plt.xlabel('Periods Ago')
plt.ylabel('Returns Std (%)')
plt.title('Returns Volatility Across Different Test Periods')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Price level over time
plt.subplot(2, 2, 3)
plt.plot(results_df['period_id'], results_df['price_mean'], 'purple', alpha=0.7)
plt.scatter(0, current['price_mean'], color='red', s=100, zorder=5, label='Current Period')
plt.xlabel('Periods Ago')
plt.ylabel('Mean Price ($)')
plt.title('Average Price Level Across Different Test Periods')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Coefficient of variation
plt.subplot(2, 2, 4)
plt.plot(results_df['period_id'], results_df['cv'], 'orange', alpha=0.7)
plt.scatter(0, current['cv'], color='red', s=100, zorder=5, label='Current Period')
plt.xlabel('Periods Ago')
plt.ylabel('Coefficient of Variation (%)')
plt.title('Relative Volatility (CV) Across Different Test Periods')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_period_volatility_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📈 Volatility analysis plot saved: test_period_volatility_analysis.png")

# Recommend better test periods
print(f"\n💡 RECOMMENDATIONS:")
print("="*50)

# Find periods with high volatility but not extreme outliers
median_range = results_df['price_range'].median()
q75_range = results_df['price_range'].quantile(0.75)

good_periods = results_df[
    (results_df['price_range'] >= q75_range) & 
    (results_df['price_range'] <= results_df['price_range'].quantile(0.95))
].head(5)

print(f"Current period volatility: ${current['price_range']:.2f} (Rank {current_rank:.0f})")
print(f"Median volatility: ${median_range:.2f}")
print(f"75th percentile: ${q75_range:.2f}")
print()
print("🎯 RECOMMENDED HIGH-VOLATILITY TEST PERIODS:")

for idx, row in good_periods.iterrows():
    periods_ago = row['period_id']
    start_idx = row['start_idx']
    end_idx = row['end_idx']
    
    print(f"\nOption {idx+1}: Period {periods_ago} periods ago")
    print(f"   Volatility: ${row['price_range']:.2f} range, ${row['price_std']:.3f} std")
    print(f"   Date: {row['start_date']} to {row['end_date']}")
    print(f"   Array indices: {start_idx} to {end_idx}")
    print(f"   To use: test_df = df.iloc[{start_idx}:{end_idx}].copy()")

print(f"\n🔧 To implement a more volatile test period:")
print("1. Choose one of the recommended periods above")
print("2. Modify the data split logic in your training scripts")
print("3. Use: test_df = df.iloc[start_idx:end_idx].copy()")
print("4. Adjust train_df accordingly to avoid data leakage") 