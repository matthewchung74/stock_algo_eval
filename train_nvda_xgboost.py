import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import shutil
from datasets import load_dataset
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
TRAINING_SIZE = 8000  # Number of samples for training
TEST_SAMPLES = 36     # Number of samples for testing

# Use a more volatile test period (Period 43 from analysis) - can be toggled
USE_VOLATILE_PERIOD = True
VOLATILE_TEST_START_IDX = 99022
VOLATILE_TEST_END_IDX = 99122  # Expanded to 100 rows

# XGBoost specific parameters
LOOKBACK_WINDOW = 20  # Number of previous periods to use as features
N_ESTIMATORS = 200    # Number of boosting rounds
MAX_DEPTH = 8         # Maximum tree depth
LEARNING_RATE = 0.07   # Learning rate

# Clean up and create organized directory structure for plots
if os.path.exists('plots/xgboost'):
    shutil.rmtree('plots/xgboost')
    print("🗑️  Deleted existing XGBoost plots folder")

plot_dir = 'plots/xgboost/nvda'
os.makedirs(plot_dir, exist_ok=True)
print(f"📁 Created directory structure: {plot_dir}")

print("🚀 Downloading NVDA dataset from Hugging Face...")
print("Dataset: matthewchung74/nvda_5_min_bars")

# Try to load the NVDA dataset from Hugging Face, else fall back to local CSV
try:
    from datasets import load_dataset
    dataset = load_dataset("matthewchung74/nvda_5_min_bars")
    df_raw = dataset['train'].to_pandas()
    print("✅ Successfully downloaded NVDA dataset from Hugging Face")
    print(f"📊 Dataset shape: {df_raw.shape}")
    print(f"📋 Columns: {list(df_raw.columns)}")
except Exception as e:
    print(f"⚠️  Could not load from Hugging Face: {e}")
    print("📂 Loading NVDA data from local CSV: nvda_5min_bars.csv")
    df_raw = pd.read_csv("nvda_5min_bars.csv", parse_dates=['timestamp'])
    # Rename columns to match expected
    df_raw = df_raw.rename(columns={
        'timestamp': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'trade_count': 'trade_count',
        'vwap': 'vwap'
    })
    print(f"✅ Loaded local CSV. Shape: {df_raw.shape}")
    print(f"📋 Columns: {list(df_raw.columns)}")

# --- Filtering by ET trading hours ---
temp_et_timestamps = pd.to_datetime(df_raw['timestamp']).dt.tz_convert('America/New_York')

# Define ET trading hours
et_start_time = pd.to_datetime('09:30:00').time()
et_end_time = pd.to_datetime('16:00:00').time()

# Create a boolean mask based on ET trading hours
trading_hours_mask = (temp_et_timestamps.dt.time >= et_start_time) & \
                     (temp_et_timestamps.dt.time <= et_end_time)

# Filter the DataFrame to include only trading hours
df_raw = df_raw[trading_hours_mask].copy()

# Prepare the data
df = pd.DataFrame()
df['ds'] = pd.to_datetime(df_raw['timestamp']).dt.tz_localize(None)
df['price'] = df_raw['close']
df['volume'] = df_raw['volume']
df['high'] = df_raw['high']
df['low'] = df_raw['low']
df['open'] = df_raw['open']
df['vwap'] = df_raw['vwap']
df['trade_count'] = df_raw['trade_count']
# Shift target to T+6 bar's close to match production-ready script
df['target'] = df['price'].shift(-6)

print(f"\nDataset info:")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
print(f"Total samples: {len(df)}")
print(f"Sample data:")
print(df.head())

# Split data using volatile test period or original logic
if USE_VOLATILE_PERIOD:
    print(f"\n🎯 Using HIGH-VOLATILITY test period (Period 43):")
    test_df = df.iloc[VOLATILE_TEST_START_IDX:VOLATILE_TEST_END_IDX].copy()
    
    # Use training data that ends before the test period to avoid data leakage
    train_end_idx = VOLATILE_TEST_START_IDX
    train_start_idx = max(0, train_end_idx - TRAINING_SIZE)
    train_df = df.iloc[train_start_idx:train_end_idx].copy()
    
    actual_training_size = len(train_df)
    
    print(f"   Test period: {test_df['ds'].min()} to {test_df['ds'].max()}")
    print(f"   Test indices: {VOLATILE_TEST_START_IDX} to {VOLATILE_TEST_END_IDX}")
    print(f"   Training period: {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"   Training indices: {train_start_idx} to {train_end_idx}")
    
    # Calculate test period volatility
    test_prices = test_df['price'].values
    test_range = test_prices.max() - test_prices.min()
    test_std = np.std(test_prices)
    test_returns_std = np.std(np.diff(test_prices) / test_prices[:-1]) * 100
    
    print(f"   Test period volatility:")
    print(f"     Price range: ${test_range:.2f} (${test_prices.min():.2f} - ${test_prices.max():.2f})")
    print(f"     Price std: ${test_std:.3f}")
    print(f"     Returns std: {test_returns_std:.3f}%")
    
else:
    # Original logic (most recent period)
    total_samples = len(df)
    if TRAINING_SIZE + TEST_SAMPLES > total_samples:
        print(f"⚠️  Warning: Requested training size ({TRAINING_SIZE}) + test size ({TEST_SAMPLES}) = {TRAINING_SIZE + TEST_SAMPLES}")
        print(f"   exceeds total samples ({total_samples}). Using maximum available training data.")
        train_df = df[:-TEST_SAMPLES].copy()
        test_df = df[-TEST_SAMPLES:].copy()
        actual_training_size = len(train_df)
    else:
        # Use the last TRAINING_SIZE + TEST_SAMPLES samples, then split
        recent_data = df[-(TRAINING_SIZE + TEST_SAMPLES):].copy()
        train_df = recent_data[:-TEST_SAMPLES].copy()
        test_df = recent_data[-TEST_SAMPLES:].copy()
        actual_training_size = len(train_df)

print(f"\n📊 Data Split Configuration:")
print(f"   Requested training size: {TRAINING_SIZE:,}")
print(f"   Actual training size: {actual_training_size:,}")
print(f"   Test size: {len(test_df):,}")
print(f"   Training period: {train_df['ds'].min()} to {train_df['ds'].max()}")
print(f"   Test period: {test_df['ds'].min()} to {test_df['ds'].max()}")

# Calculate training data span
training_days = actual_training_size / 78  # Approximate trading days
training_weeks = training_days / 5
training_months = training_weeks / 4.3
print(f"   Training span: ~{training_days:.0f} trading days (~{training_weeks:.1f} weeks, ~{training_months:.1f} months)")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df, lookback_window=20):
    """Production-ready feature engineering for XGBoost"""
    features_df = df.copy()
    # Basic features
    features_df['volume'] = df['volume']
    features_df['returns'] = df['price'].pct_change()
    features_df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
    features_df['price_change'] = df['price'].diff()
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    features_df['rsi_14'] = calculate_rsi(df['price'], 14)
    features_df['rsi_9'] = calculate_rsi(df['price'], 9)
    features_df['rsi_21'] = calculate_rsi(df['price'], 21)
    # MACD
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    macd, macd_signal, macd_hist = calculate_macd(df['price'])
    features_df['macd'] = macd
    features_df['macd_signal'] = macd_signal
    features_df['macd_histogram'] = macd_hist
    # Bollinger Bands
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(df['price'])
    features_df['bb_upper'] = bb_upper
    features_df['bb_lower'] = bb_lower
    features_df['bb_middle'] = bb_middle
    features_df['bb_position'] = (df['price'] - bb_lower) / (bb_upper - bb_lower)
    features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    # Stochastic Oscillator
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['price'])
    features_df['stoch_k'] = stoch_k
    features_df['stoch_d'] = stoch_d
    # Technical indicators - moving averages
    for period in [5, 10, 20, 50]:
        features_df[f'sma_{period}'] = df['price'].rolling(period).mean()
        features_df[f'ema_{period}'] = df['price'].ewm(span=period).mean()
        features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std()
        features_df[f'price_vs_sma_{period}'] = df['price'] / features_df[f'sma_{period}'] - 1
        features_df[f'price_vs_ema_{period}'] = df['price'] / features_df[f'ema_{period}'] - 1
    # Volume Rate of Change
    features_df['volume_roc_5'] = features_df['volume'].pct_change(5)
    features_df['volume_roc_10'] = features_df['volume'].pct_change(10)
    # Volume-Weighted Price indicators
    features_df['vwap_deviation'] = (df['price'] - df['vwap']) / df['vwap']
    # On-Balance Volume (OBV)
    def calculate_obv(close, volume):
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]
        return pd.Series(obv, index=close.index)
    features_df['obv'] = calculate_obv(df['price'], df['volume'])
    features_df['obv_sma_10'] = features_df['obv'].rolling(10).mean()
    features_df['obv_ratio'] = features_df['obv'] / features_df['obv_sma_10']
    # Bid-Ask Spread proxy using high-low
    features_df['spread_proxy'] = (df['high'] - df['low']) / df['price']
    # Price efficiency measures
    features_df['price_efficiency_5'] = abs(df['price'].pct_change(5)) / features_df['volatility_5']
    features_df['price_efficiency_10'] = abs(df['price'].pct_change(10)) / features_df['volatility_10']
    # Advanced Volume Features
    for period in [5, 10, 20]:
        features_df[f'volume_sma_{period}'] = features_df['volume'].rolling(period).mean()
        features_df[f'volume_ema_{period}'] = features_df['volume'].ewm(span=period).mean()
    features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_10']
    features_df['volume_price_trend'] = features_df['volume'] * features_df['returns']
    features_df['volume_momentum_5'] = features_df['volume'].rolling(5).mean() / features_df['volume'].rolling(10).mean()
    features_df['volume_momentum_10'] = features_df['volume'].rolling(10).mean() / features_df['volume'].rolling(20).mean()
    # Enhanced OHLC features
    features_df['high_low_ratio'] = df['high'] / df['low']
    features_df['close_open_ratio'] = df['price'] / df['open']
    features_df['price_to_vwap'] = df['price'] / df['vwap']
    features_df['body_size'] = abs(df['price'] - df['open']) / df['price']
    features_df['upper_shadow'] = (df['high'] - np.maximum(df['price'], df['open'])) / df['price']
    features_df['lower_shadow'] = (np.minimum(df['price'], df['open']) - df['low']) / df['price']
    features_df['total_range'] = (df['high'] - df['low']) / df['price']
    features_df['gap'] = (df['open'] - df['price'].shift(1)) / df['price'].shift(1)
    features_df['gap_filled'] = ((df['low'] <= df['price'].shift(1)) & (features_df['gap'] > 0)) | \
                               ((df['high'] >= df['price'].shift(1)) & (features_df['gap'] < 0))
    # Enhanced Time Features
    features_df['hour'] = features_df['ds'].dt.hour
    features_df['minute'] = features_df['ds'].dt.minute  
    features_df['day_of_week'] = features_df['ds'].dt.dayofweek
    features_df['time_of_day'] = features_df['hour'] * 60 + features_df['minute']
    features_df['is_market_open'] = features_df['hour'].between(9, 16)
    features_df['is_power_hour'] = features_df['hour'] == 15
    features_df['is_first_hour'] = features_df['hour'] == 9
    features_df['session_progress'] = (features_df['hour'] - 9.5) / 6.5
    # Multi-timeframe momentum
    for window in [3, 5, 10, 15, 20]:
        features_df[f'momentum_{window}'] = df['price'].pct_change(window)
        features_df[f'momentum_strength_{window}'] = abs(features_df[f'momentum_{window}']) / features_df['volatility_20']
    # Enhanced Lag features
    for lag in range(1, 21):
        features_df[f'price_lag_{lag}'] = df['price'].shift(lag)
        features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
        features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
        if lag <= 10:
            features_df[f'volatility_lag_{lag}'] = features_df['volatility_20'].shift(lag)
            features_df[f'rsi_lag_{lag}'] = features_df['rsi_14'].shift(lag)
    # Enhanced Rolling window features
    for window in [5, 10, 20]:
        features_df[f'price_min_{window}'] = df['price'].rolling(window).min()
        features_df[f'price_max_{window}'] = df['price'].rolling(window).max()
        features_df[f'price_std_{window}'] = df['price'].rolling(window).std()
        features_df[f'price_position_{window}'] = (df['price'] - features_df[f'price_min_{window}']) / \
                                                 (features_df[f'price_max_{window}'] - features_df[f'price_min_{window}'])
        if window >= 10:
            features_df[f'price_volume_corr_{window}'] = df['price'].rolling(window).corr(df['volume'])
            features_df[f'returns_volume_corr_{window}'] = features_df['returns'].rolling(window).corr(df['volume'])
    # Volatility regimes
    features_df['volatility_regime'] = pd.qcut(features_df['volatility_20'].rank(method='first'), 
                                              q=3, labels=[0, 1, 2]).astype(float)
    features_df['volume_regime'] = pd.qcut(features_df['volume'].rank(method='first'), 
                                          q=3, labels=[0, 1, 2]).astype(float)
    # Trend strength indicators
    for period in [10, 20]:
        def rolling_slope(series, window):
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
            return pd.Series(slopes, index=series.index)
        features_df[f'trend_strength_{period}'] = rolling_slope(df['price'], period)
        features_df[f'trend_strength_norm_{period}'] = features_df[f'trend_strength_{period}'] / df['price']
    return features_df

print(f"\n🔧 Creating features for XGBoost model...")
print(f"   Lookback window: {LOOKBACK_WINDOW} periods")
print(f"   Feature engineering in progress...")

# Create features for training and test data
train_features_df = create_features(train_df)
test_features_df = create_features(test_df)

# Select feature columns (exclude target, metadata, and current-bar price/volume/etc.)
feature_columns = [col for col in train_features_df.columns 
                  if col not in ['ds', 'price', 'high', 'low', 'open', 'vwap', 'trade_count', 'target']]

print(f"   Created {len(feature_columns)} features:")
print(f"   Feature categories:")
print(f"     - Price/Returns: {len([c for c in feature_columns if any(x in c for x in ['price', 'returns', 'sma', 'ema'])])}")
print(f"     - Volatility: {len([c for c in feature_columns if 'volatility' in c])}")
print(f"     - Volume: {len([c for c in feature_columns if 'volume' in c])}")
print(f"     - Time: {len([c for c in feature_columns if any(x in c for x in ['hour', 'minute', 'day', 'time'])])}")
print(f"     - Lagged: {len([c for c in feature_columns if 'lag' in c])}")
print(f"     - Rolling: {len([c for c in feature_columns if any(x in c for x in ['min', 'max', 'std'])])}")

# =============================================================================
# PREPARE TRAINING DATA
# =============================================================================

print(f"\n🤖 Preparing XGBoost training data...")

# Remove rows with NaN values (due to rolling windows, lags, and target shift)
train_features_clean = train_features_df.dropna(subset=feature_columns + ['target'])
test_features_clean = test_features_df.dropna(subset=feature_columns + ['target'])

# Add a check for empty test set
if len(test_features_clean) == 0:
    raise ValueError("Test set is empty after dropping NaNs. Increase test set size or adjust feature engineering.")

print(f"   Training samples after cleaning: {len(train_features_clean)} (removed {len(train_features_df) - len(train_features_clean)} NaN rows)")
print(f"   Test samples after cleaning: {len(test_features_clean)} (removed {len(test_features_df) - len(test_features_clean)} NaN rows)")

# Prepare feature matrices
X_train = train_features_clean[feature_columns].values
y_train = train_features_clean['target'].values

X_test = test_features_clean[feature_columns].values
y_test = test_features_clean['target'].values

print(f"   Training features shape: {X_train.shape}")
print(f"   Training target shape: {y_train.shape}")
print(f"   Test features shape: {X_test.shape}")
print(f"   Test target shape: {y_test.shape}")

# =============================================================================
# XGBOOST MODEL TRAINING
# =============================================================================

print(f"\n🚀 Training XGBoost model...")
print(f"   Parameters:")
print(f"     n_estimators: {N_ESTIMATORS}")
print(f"     max_depth: {MAX_DEPTH}")
print(f"     learning_rate: {LEARNING_RATE}")

# Create XGBoost model
model = xgb.XGBRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train the model
model.fit(X_train, y_train)

print(f"✅ XGBoost model training completed!")

# =============================================================================
# FORECASTING AND EVALUATION
# =============================================================================

print(f"\n📈 Generating XGBoost forecasts for {len(y_test)} periods...")

# Generate predictions
test_predictions = model.predict(X_test)

print(f"   Predictions shape: {test_predictions.shape}")
print(f"   Predictions range: ${test_predictions.min():.2f} - ${test_predictions.max():.2f}")

# Get actual values for comparison
actual_values = y_test

# Calculate error metrics
mae = mean_absolute_error(actual_values, test_predictions)
mse = mean_squared_error(actual_values, test_predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_predictions - actual_values) / actual_values)) * 100

period_type = "VOLATILE PERIOD" if USE_VOLATILE_PERIOD else "RECENT PERIOD"
print(f"\nNVDA XGBoost Performance Metrics ({len(y_test)} sample horizon - {period_type}):")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Mean Square Error (MSE): ${mse:.2f}")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print(f"\n📊 Analyzing feature importance...")

# Get feature importance
feature_importance = model.feature_importances_
feature_names = feature_columns

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"   Top 10 most important features:")
for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
    print(f"     {i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")

# =============================================================================
# VISUALIZATION
# =============================================================================

# Plot BEFORE fitting - Training data only
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(pd.to_datetime(train_df['ds']), train_df['price'], 'b-', label='Training Data', alpha=0.7)
plt.plot(pd.to_datetime(test_features_clean['ds']), actual_values, 'g-', label='Test Data (Actual)', alpha=0.7, linewidth=3)
title_suffix = " (VOLATILE PERIOD)" if USE_VOLATILE_PERIOD else ""
plt.title(f'BEFORE Forecast - NVDA Training vs Test Data{title_suffix}\n(Training: {len(train_features_clean):,} samples, Test: {len(y_test)} samples)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.xticks(rotation=45)
plt.legend()

# Plot AFTER fitting - Training data + Predictions
plt.subplot(2, 2, 2)
# Show last part of training data for context
context_start = max(0, len(train_features_clean) - 200)
train_context = train_features_clean.iloc[context_start:]
plt.plot(pd.to_datetime(train_context['ds']), train_context['price'], 'b-', label='Training Data (Last 200)', alpha=0.7)
plt.plot(pd.to_datetime(test_features_clean['ds']), test_predictions, 'r-', label='XGBoost Forecast', alpha=0.8, linewidth=3)
title_suffix = " (VOLATILE)" if USE_VOLATILE_PERIOD else ""
plt.title(f'AFTER Forecast - NVDA Training Data + XGBoost{title_suffix}\n({len(y_test)} sample horizon)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.xticks(rotation=45)
plt.legend()

# Compare predictions to actual test data
plt.subplot(2, 2, 3)
plt.plot(pd.to_datetime(test_features_clean['ds']), actual_values, 'g-', label='Actual Test Data', linewidth=3, marker='o', markersize=6)
plt.plot(pd.to_datetime(test_features_clean['ds']), test_predictions, 'r-', label='XGBoost Forecast', alpha=0.8, linewidth=3, marker='s', markersize=6)
title_suffix = " (VOLATILE PERIOD)" if USE_VOLATILE_PERIOD else ""
plt.title(f'NVDA Prediction vs Actual{title_suffix}\n({len(y_test)} sample test period)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.xticks(rotation=45)
plt.legend()

# Scatter plot
plt.subplot(2, 2, 4)
plt.scatter(actual_values, test_predictions, alpha=0.8, s=100, c=range(len(actual_values)), cmap='viridis')
plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
title_suffix = " (VOLATILE)" if USE_VOLATILE_PERIOD else ""
plt.title(f'NVDA Actual vs Predicted{title_suffix}\nMAE: ${mae:.2f}, RMSE: ${rmse:.2f}\nMAPE: {mape:.2f}%')
plt.colorbar(label='Sample Order')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'NVDA_XGBoost_main_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/xgboost/nvda/NVDA_XGBoost_main_analysis.png")

# =============================================================================
# FEATURE IMPORTANCE VISUALIZATION
# =============================================================================

plt.figure(figsize=(15, 10))

# Plot 1: Top 20 feature importance
plt.subplot(2, 2, 1)
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()

# Plot 2: Feature importance by category
plt.subplot(2, 2, 2)
categories = {
    'Price/Returns': [c for c in feature_columns if any(x in c for x in ['price', 'returns', 'sma', 'ema', 'close', 'ratio'])],
    'Volatility': [c for c in feature_columns if 'volatility' in c],
    'Volume': [c for c in feature_columns if 'volume' in c],
    'Time': [c for c in feature_columns if any(x in c for x in ['hour', 'minute', 'day', 'time'])],
    'Lagged': [c for c in feature_columns if 'lag' in c],
    'Rolling': [c for c in feature_columns if any(x in c for x in ['min', 'max', 'std'])]
}

category_importance = {}
for cat, features in categories.items():
    cat_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
    category_importance[cat] = cat_importance

plt.pie(category_importance.values(), labels=category_importance.keys(), autopct='%1.1f%%')
plt.title('Feature Importance by Category')

# Plot 3: Cumulative feature importance
plt.subplot(2, 2, 3)
cumulative_importance = np.cumsum(importance_df['importance'])
plt.plot(range(len(cumulative_importance)), cumulative_importance, 'b-', linewidth=2)
plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Learning curve (if available)
plt.subplot(2, 2, 4)
if hasattr(model, 'evals_result_'):
    eval_results = model.evals_result_
    plt.plot(eval_results['validation_0']['rmse'], label='Training RMSE')
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    # Show feature count by category as alternative
    cat_counts = {cat: len(features) for cat, features in categories.items()}
    plt.bar(cat_counts.keys(), cat_counts.values())
    plt.title('Feature Count by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Features')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'NVDA_XGBoost_feature_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/xgboost/nvda/NVDA_XGBoost_feature_analysis.png")

# =============================================================================
# DIRECTIONAL ACCURACY ANALYSIS
# =============================================================================

print(f"\n" + "="*60)
print(f"NVDA XGBOOST DIRECTIONAL PREDICTION ANALYSIS - {period_type} ({len(y_test)} samples)")
print("="*60)

# Calculate actual direction changes
actual_directions = []
predicted_directions = []

# Compare with the last training point for the first prediction
last_train_price = train_features_clean['price'].iloc[-1]
actual_directions.append(1 if actual_values[0] > last_train_price else -1)
predicted_directions.append(1 if test_predictions[0] > last_train_price else -1)

# For remaining predictions, compare with previous period
for i in range(1, len(actual_values)):
    actual_dir = 1 if actual_values[i] > actual_values[i-1] else -1
    actual_directions.append(actual_dir)
    
    pred_dir = 1 if test_predictions[i] > test_predictions[i-1] else -1
    predicted_directions.append(pred_dir)

# Convert to numpy arrays
actual_directions = np.array(actual_directions)
predicted_directions = np.array(predicted_directions)

# Calculate directional accuracy
correct_directions = (actual_directions == predicted_directions)
directional_accuracy = np.mean(correct_directions) * 100

print(f"Directional Accuracy: {directional_accuracy:.1f}%")
print(f"Correct predictions: {np.sum(correct_directions)}/{len(correct_directions)}")

# Directional analysis plot
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
sample_indices = range(1, len(actual_directions) + 1)
plt.plot(sample_indices, actual_directions, 'go-', label='Actual Direction', linewidth=3, markersize=8)
plt.plot(sample_indices, predicted_directions, 'ro-', label='Predicted Direction', linewidth=3, markersize=8)
title_suffix = " (VOLATILE)" if USE_VOLATILE_PERIOD else ""
plt.title(f'NVDA Direction Predictions{title_suffix}\n(1 = Up, -1 = Down)')
plt.xlabel('Sample Number')
plt.ylabel('Direction')
plt.legend()
plt.ylim(-1.5, 1.5)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
accuracy_by_sample = np.cumsum(correct_directions) / np.arange(1, len(correct_directions) + 1) * 100
plt.plot(sample_indices, accuracy_by_sample, 'b-', linewidth=3, marker='o', markersize=8)
title_suffix = " (VOLATILE)" if USE_VOLATILE_PERIOD else ""
plt.title(f'Cumulative Directional Accuracy{title_suffix}\nFinal: {directional_accuracy:.1f}%')
plt.xlabel('Sample Number')
plt.ylabel('Accuracy (%)')
plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(actual_directions, predicted_directions, labels=[-1, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up'])
title_suffix = " (VOLATILE)" if USE_VOLATILE_PERIOD else ""
plt.title(f'NVDA Direction Confusion Matrix{title_suffix}')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'NVDA_XGBoost_directional_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/xgboost/nvda/NVDA_XGBoost_directional_analysis.png")

# =============================================================================
# PREDICTED VS ACTUAL VALUES PLOT
# =============================================================================

plt.figure(figsize=(14, 8))

# Plot 1: Time series comparison
plt.subplot(2, 1, 1)
dates = pd.to_datetime(test_features_clean['ds'])
plt.plot(dates, actual_values, 'g-', label='Actual Values', linewidth=3, marker='o', markersize=6, alpha=0.8)
plt.plot(dates, test_predictions, 'r-', label='XGBoost Forecast', linewidth=3, marker='s', markersize=6, alpha=0.8)
title_suffix = " (VOLATILE PERIOD)" if USE_VOLATILE_PERIOD else ""
plt.title(f'NVDA XGBoost: Predicted vs Actual Values{title_suffix}\n({len(y_test)} sample test period)')
plt.xlabel('Date/Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Error analysis
plt.subplot(2, 1, 2)
errors = test_predictions - actual_values
plt.plot(dates, errors, 'purple', linewidth=2, marker='d', markersize=5, alpha=0.8)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Perfect Prediction')
plt.axhline(y=np.mean(errors), color='red', linestyle='-', alpha=0.7, label=f'Mean Error: ${np.mean(errors):.2f}')
plt.fill_between(dates, errors, 0, alpha=0.3, color='purple')
title_suffix = " (VOLATILE)" if USE_VOLATILE_PERIOD else ""
plt.title(f'Prediction Errors Over Time{title_suffix}\nMAE: ${mae:.2f}, RMSE: ${rmse:.2f}')
plt.xlabel('Date/Time')
plt.ylabel('Prediction Error ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'NVDA_XGBoost_predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/xgboost/nvda/NVDA_XGBoost_predicted_vs_actual.png")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print(f"\n📊 Detailed Prediction vs Actual Comparison ({period_type}):")
print("=" * 80)
comparison_df = pd.DataFrame({
    'Timestamp': test_features_clean['ds'].values,
    'Actual_Price': actual_values,
    'Predicted_Price': test_predictions,
    'Absolute_Error': np.abs(test_predictions - actual_values),
    'Percentage_Error': np.abs((test_predictions - actual_values) / actual_values) * 100
})
print(comparison_df.to_string(index=False, float_format='%.2f'))

print(f"\n📈 XGBoost Statistics ({period_type}):")
print(f"Prediction Variability (Std): ${np.std(test_predictions):.3f}")
print(f"Actual Variability (Std): ${np.std(actual_values):.3f}")
print(f"Mean Prediction: ${np.mean(test_predictions):.2f}")
print(f"Mean Actual: ${np.mean(actual_values):.2f}")
print(f"Training Data Used: {len(train_features_clean):,} samples")
print(f"Model Type: XGBoost")
print(f"Model Parameters: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}")
print(f"Features Used: {len(feature_columns)}")
print(f"Lookback Window: {LOOKBACK_WINDOW} periods")
print(f"Test Period Type: {'HIGH VOLATILITY (Period 43)' if USE_VOLATILE_PERIOD else 'RECENT (Period 0)'}")

print(f"\n🎯 All NVDA XGBoost plots saved to 'plots/xgboost/nvda/' directory:")
print("   - NVDA_XGBoost_main_analysis.png")
print("   - NVDA_XGBoost_feature_analysis.png")
print("   - NVDA_XGBoost_directional_analysis.png")
print("   - NVDA_XGBoost_predicted_vs_actual.png")

print(f"\n🚀 Analysis complete! Used {len(y_test)} samples for testing with XGBoost.")

# Show comparison if using volatile period
if USE_VOLATILE_PERIOD:
    print(f"\n📊 COMPARISON: VOLATILE vs FLAT TEST PERIODS")
    print("="*60)
    print(f"VOLATILE Period (Period 43) - CURRENT:")
    print(f"   Price range: ${test_range:.2f} (${test_prices.min():.2f} - ${test_prices.max():.2f})")
    print(f"   Price std: ${test_std:.3f}")
    print(f"   Returns std: {test_returns_std:.3f}%")
    print(f"   Date: {test_df['ds'].min()} to {test_df['ds'].max()}")
    print()
    print(f"FLAT Period (Period 0) - ORIGINAL:")
    print(f"   Price range: $1.07 ($134.94 - $136.01)")
    print(f"   Price std: $0.246")
    print(f"   Returns std: 0.106%")
    print(f"   Date: 2025-05-16 17:05:00 to 2025-05-16 20:00:00")
    print()
    print(f"IMPROVEMENT: {test_range/1.07:.1f}x more volatile test period!")
    print(f"\n💡 To use the original flat period, set USE_VOLATILE_PERIOD = False") 