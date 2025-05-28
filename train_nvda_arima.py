import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import shutil
from datasets import load_dataset
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Try to import auto_arima, fallback if not available
try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False
    print("⚠️  pmdarima not available - using manual ARIMA selection")

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
TRAINING_SIZE = 8000  # Number of samples for training
TEST_SAMPLES = 36     # Number of samples for testing

# Use a more volatile test period (Period 43 from analysis) - can be toggled
USE_VOLATILE_PERIOD = True
VOLATILE_TEST_START_IDX = 99022
VOLATILE_TEST_END_IDX = 99058

# Clean up and create organized directory structure for plots
if os.path.exists('plots/arima'):
    shutil.rmtree('plots/arima')
    print("🗑️  Deleted existing ARIMA plots folder")

plot_dir = 'plots/arima/nvda'
os.makedirs(plot_dir, exist_ok=True)
print(f"📁 Created directory structure: {plot_dir}")

print("🚀 Downloading NVDA dataset from Hugging Face...")
print("Dataset: matthewchung74/nvda_5_min_bars")

# Load the NVDA dataset from Hugging Face
dataset = load_dataset("matthewchung74/nvda_5_min_bars")
df_raw = dataset['train'].to_pandas()
print("✅ Successfully downloaded NVDA dataset from Hugging Face")
print(f"📊 Dataset shape: {df_raw.shape}")
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
df['returns'] = df['price'].pct_change()

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
# ARIMA MODEL SETUP AND TRAINING
# =============================================================================

print(f"\n🤖 Setting up ARIMA model...")

# Prepare data for ARIMA (use returns for stationarity)
train_prices = train_df['price'].values
train_returns = np.diff(train_prices) / train_prices[:-1]

# Remove any infinite or NaN values
train_returns = train_returns[np.isfinite(train_returns)]

print(f"   Using price returns for stationarity")
print(f"   Training returns shape: {train_returns.shape}")
print(f"   Returns range: {train_returns.min():.6f} to {train_returns.max():.6f}")
print(f"   Returns mean: {np.mean(train_returns):.6f}")
print(f"   Returns std: {np.std(train_returns):.6f}")

# Perform stationarity test
print(f"\n📊 Stationarity Analysis:")
adf_result = adfuller(train_returns)
print(f"   ADF Statistic: {adf_result[0]:.6f}")
print(f"   p-value: {adf_result[1]:.6f}")
print(f"   Critical Values:")
for key, value in adf_result[4].items():
    print(f"     {key}: {value:.6f}")

if adf_result[1] <= 0.05:
    print("   ✅ Series is stationary (p-value ≤ 0.05)")
else:
    print("   ⚠️  Series may not be stationary (p-value > 0.05)")

# Automatic ARIMA model selection
print(f"\n🔍 Performing automatic ARIMA model selection...")
print("   This may take a moment...")

if AUTO_ARIMA_AVAILABLE:
    try:
        # Use auto_arima for automatic model selection
        auto_model = auto_arima(
            train_returns,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=False,  # No seasonality for returns
            stepwise=True,   # Faster stepwise search
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        
        print(f"✅ Automatic model selection completed!")
        print(f"   Selected model: {auto_model.order}")
        
        # Fit the selected model
        model = auto_model
        model_order = auto_model.order
        
    except Exception as e:
        print(f"⚠️  Auto ARIMA failed: {e}")
        print("   Falling back to ARIMA(1,0,1)")
        
        # Fallback to a simple ARIMA model
        model = ARIMA(train_returns, order=(1, 0, 1))
        model = model.fit()
        model_order = (1, 0, 1)
else:
    print("   Using manual ARIMA(1,0,1) selection")
    
    # Fallback to a simple ARIMA model
    model = ARIMA(train_returns, order=(1, 0, 1))
    model = model.fit()
    model_order = (1, 0, 1)

print(f"   Final model order: {model_order}")
print(f"   AIC: {model.aic:.2f}")
print(f"   BIC: {model.bic:.2f}")

# =============================================================================
# FORECASTING AND EVALUATION
# =============================================================================

print(f"\n🔮 Generating ARIMA forecasts for {TEST_SAMPLES} periods...")

# Generate return forecasts
forecast_result = model.forecast(steps=TEST_SAMPLES)
forecast_ci = model.get_forecast(steps=TEST_SAMPLES).conf_int()

# Handle different return types from forecast
if hasattr(forecast_result, 'values'):
    forecast_returns = forecast_result.values
elif isinstance(forecast_result, np.ndarray):
    forecast_returns = forecast_result
else:
    # Single value case - expand to array
    forecast_returns = np.full(TEST_SAMPLES, forecast_result)

# Handle confidence intervals
if hasattr(forecast_ci, 'values'):
    conf_int = forecast_ci.values
else:
    conf_int = forecast_ci

print(f"   Return forecasts shape: {forecast_returns.shape}")
print(f"   Return forecasts range: {forecast_returns.min():.6f} to {forecast_returns.max():.6f}")

# Convert return forecasts back to price forecasts
last_price = train_prices[-1]
test_predictions = np.zeros(TEST_SAMPLES)
test_lower = np.zeros(TEST_SAMPLES)
test_upper = np.zeros(TEST_SAMPLES)

test_predictions[0] = last_price * (1 + forecast_returns[0])
test_lower[0] = last_price * (1 + conf_int[0, 0])
test_upper[0] = last_price * (1 + conf_int[0, 1])

# For subsequent periods, compound the returns
for i in range(1, TEST_SAMPLES):
    test_predictions[i] = test_predictions[i-1] * (1 + forecast_returns[i])
    test_lower[i] = test_lower[i-1] * (1 + conf_int[i, 0])
    test_upper[i] = test_upper[i-1] * (1 + conf_int[i, 1])

print(f"   Price predictions range: ${test_predictions.min():.2f} - ${test_predictions.max():.2f}")

# Get actual values for comparison
actual_values = test_df['price'].values

# Calculate error metrics
mae = np.mean(np.abs(test_predictions - actual_values))
mse = np.mean((test_predictions - actual_values)**2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_predictions - actual_values) / actual_values)) * 100

period_type = "VOLATILE PERIOD" if USE_VOLATILE_PERIOD else "RECENT PERIOD"
print(f"\nNVDA ARIMA Performance Metrics ({TEST_SAMPLES} sample horizon - {period_type}):")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Mean Square Error (MSE): ${mse:.2f}")

# =============================================================================
# VISUALIZATION
# =============================================================================

# Plot BEFORE fitting - Training data only
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(pd.to_datetime(train_df['ds']), train_df['price'], 'b-', label='Training Data', alpha=0.7)
plt.plot(pd.to_datetime(test_df['ds']), test_df['price'], 'g-', label='Test Data (Actual)', alpha=0.7, linewidth=3)
title_suffix = " (VOLATILE PERIOD)" if USE_VOLATILE_PERIOD else ""
plt.title(f'BEFORE Forecast - NVDA Training vs Test Data{title_suffix}\n(Training: {actual_training_size:,} samples, Test: {TEST_SAMPLES} samples)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.xticks(rotation=45)
plt.legend()

# Plot AFTER fitting - Training data + Predictions
plt.subplot(2, 2, 2)
# Show last part of training data for context
context_start = max(0, len(train_df) - 200)
plt.plot(pd.to_datetime(train_df['ds'].iloc[context_start:]), train_df['price'].iloc[context_start:], 'b-', label='Training Data (Last 200)', alpha=0.7)
plt.plot(pd.to_datetime(test_df['ds']), test_predictions, 'r-', label='ARIMA Forecast', alpha=0.8, linewidth=3)
plt.fill_between(pd.to_datetime(test_df['ds']), test_lower, test_upper, alpha=0.3, color='red', label='95% Confidence Interval')
title_suffix = " (VOLATILE)" if USE_VOLATILE_PERIOD else ""
plt.title(f'AFTER Forecast - NVDA Training Data + ARIMA{title_suffix}\n({TEST_SAMPLES} sample horizon, Order: {model_order})')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.xticks(rotation=45)
plt.legend()

# Compare predictions to actual test data
plt.subplot(2, 2, 3)
plt.plot(pd.to_datetime(test_df['ds']), actual_values, 'g-', label='Actual Test Data', linewidth=3, marker='o', markersize=6)
plt.plot(pd.to_datetime(test_df['ds']), test_predictions, 'r-', label='ARIMA Forecast', alpha=0.8, linewidth=3, marker='s', markersize=6)
plt.fill_between(pd.to_datetime(test_df['ds']), test_lower, test_upper, alpha=0.3, color='red', label='95% CI')
title_suffix = " (VOLATILE PERIOD)" if USE_VOLATILE_PERIOD else ""
plt.title(f'NVDA Prediction vs Actual{title_suffix}\n({TEST_SAMPLES} sample test period)')
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
plt.savefig(os.path.join(plot_dir, 'NVDA_ARIMA_main_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/arima/nvda/NVDA_ARIMA_main_analysis.png")

# =============================================================================
# ARIMA SPECIFIC VISUALIZATIONS
# =============================================================================

# Plot model diagnostics
plt.figure(figsize=(15, 10))

# Plot 1: Returns time series and forecasts
plt.subplot(2, 2, 1)
returns_dates = pd.to_datetime(train_df['ds'].iloc[1:])  # Skip first date due to diff
plt.plot(returns_dates[-200:], train_returns[-200:], 'b-', label='Training Returns (Last 200)', alpha=0.7)
forecast_dates = pd.to_datetime(test_df['ds'])
plt.plot(forecast_dates, forecast_returns, 'r-', label='ARIMA Return Forecasts', linewidth=3, marker='s', markersize=4)
plt.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], alpha=0.3, color='red', label='95% CI')
plt.title(f'ARIMA Return Forecasts\n(Model: {model_order})')
plt.xlabel('Date/Time')
plt.ylabel('Returns')
plt.legend()
plt.xticks(rotation=45)

# Plot 2: Residuals analysis
plt.subplot(2, 2, 2)
residuals = model.resid
plt.plot(residuals, 'b-', alpha=0.7)
plt.title('Model Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

# Plot 3: Residuals histogram
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# Plot 4: Q-Q plot for residuals
plt.subplot(2, 2, 4)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'NVDA_ARIMA_diagnostics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/arima/nvda/NVDA_ARIMA_diagnostics.png")

# =============================================================================
# DIRECTIONAL ACCURACY ANALYSIS
# =============================================================================

print(f"\n" + "="*60)
print(f"NVDA ARIMA DIRECTIONAL PREDICTION ANALYSIS - {period_type} ({TEST_SAMPLES} samples)")
print("="*60)

# Calculate actual direction changes
actual_directions = []
predicted_directions = []

# Compare with the last training point for the first prediction
last_train_price = train_df['price'].iloc[-1]
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
plt.savefig(os.path.join(plot_dir, 'NVDA_ARIMA_directional_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/arima/nvda/NVDA_ARIMA_directional_analysis.png")

# =============================================================================
# PREDICTED VS ACTUAL VALUES PLOT
# =============================================================================

plt.figure(figsize=(14, 8))

# Plot 1: Time series comparison
plt.subplot(2, 1, 1)
dates = pd.to_datetime(test_df['ds'])
plt.plot(dates, actual_values, 'g-', label='Actual Values', linewidth=3, marker='o', markersize=6, alpha=0.8)
plt.plot(dates, test_predictions, 'r-', label='ARIMA Forecast', linewidth=3, marker='s', markersize=6, alpha=0.8)
plt.fill_between(dates, test_lower, test_upper, alpha=0.3, color='red', label='95% Confidence Interval')
title_suffix = " (VOLATILE PERIOD)" if USE_VOLATILE_PERIOD else ""
plt.title(f'NVDA ARIMA: Predicted vs Actual Values{title_suffix}\n({TEST_SAMPLES} sample test period, Model: {model_order})')
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
plt.savefig(os.path.join(plot_dir, 'NVDA_ARIMA_predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/arima/nvda/NVDA_ARIMA_predicted_vs_actual.png")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print(f"\n📊 Detailed Prediction vs Actual Comparison ({period_type}):")
print("=" * 80)
comparison_df = pd.DataFrame({
    'Timestamp': test_df['ds'].values,
    'Actual_Price': actual_values,
    'Predicted_Price': test_predictions,
    'Lower_CI': test_lower,
    'Upper_CI': test_upper,
    'Absolute_Error': np.abs(test_predictions - actual_values),
    'Percentage_Error': np.abs((test_predictions - actual_values) / actual_values) * 100
})
print(comparison_df.to_string(index=False, float_format='%.2f'))

print(f"\n📈 ARIMA Statistics ({period_type}):")
print(f"Prediction Variability (Std): ${np.std(test_predictions):.3f}")
print(f"Actual Variability (Std): ${np.std(actual_values):.3f}")
print(f"Mean Prediction: ${np.mean(test_predictions):.2f}")
print(f"Mean Actual: ${np.mean(actual_values):.2f}")
print(f"Training Data Used: {actual_training_size:,} samples")
print(f"Model Type: ARIMA{model_order}")
print(f"Model AIC: {model.aic:.2f}")
print(f"Model BIC: {model.bic:.2f}")
print(f"Test Period Type: {'HIGH VOLATILITY (Period 43)' if USE_VOLATILE_PERIOD else 'RECENT (Period 0)'}")

# Calculate confidence interval coverage
coverage_count = np.sum([(test_lower[i] <= actual_values[i] <= test_upper[i]) for i in range(len(actual_values))])
coverage_percentage = coverage_count / len(actual_values) * 100
print(f"95% Confidence Interval Coverage: {coverage_percentage:.1f}% ({coverage_count}/{len(actual_values)})")

print(f"\n🎯 All NVDA ARIMA plots saved to 'plots/arima/nvda/' directory:")
print("   - NVDA_ARIMA_main_analysis.png")
print("   - NVDA_ARIMA_diagnostics.png")
print("   - NVDA_ARIMA_directional_analysis.png")
print("   - NVDA_ARIMA_predicted_vs_actual.png")

print(f"\n🚀 Analysis complete! Used {TEST_SAMPLES} samples for testing with ARIMA{model_order}.")

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