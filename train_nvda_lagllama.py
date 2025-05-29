#!/usr/bin/env python3
"""
NVDA Time Series Forecasting with Lag-Llama Foundation Model + Fine-Tuning
=========================================================================

This script implements fine-tuning of the Lag-Llama foundation model for NVDA stock price 
forecasting. Lag-Llama is the first open-source foundation model specifically designed for 
probabilistic time series forecasting.

Key Features:
- First open-source foundation model for time series forecasting
- Probabilistic forecasting with uncertainty quantification
- Lag-based tokenization approach
- Decoder-only Transformer architecture (similar to LLaMA)
- Both zero-shot and fine-tuning capabilities
- Built on GluonTS framework

Model Details:
- Purpose-built for time series (not adapted from language models)
- Supports any frequency and prediction length
- Strong zero-shot performance, better with fine-tuning
- Trained on 27 diverse time series datasets (352M tokens)

Author: AI Assistant
Date: 2024
"""

import os
import sys
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datasets import load_dataset
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATASET_NAME = "matthewchung74/nvda_5_min_bars"
TRAINING_SIZE = 8000
TEST_SIZE = 36
USE_VOLATILE_PERIOD = True
SYMBOL = "NVDA"
PREDICTION_LENGTH = 12  # 1 hour ahead (12 * 5min)
CONTEXT_LENGTH = 96     # 8 hours of 5-min data
NUM_SAMPLES = 100       # For probabilistic forecasting
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 50
DEVICE = "cpu"

def setup_lagllama():
    """Setup Lag-Llama by installing from GitHub if not available."""
    print("🔧 Setting up Lag-Llama...")
    
    try:
        # Try to import lag_llama first
        from lag_llama.gluon.estimator import LagLlamaEstimator
        print("✅ Lag-Llama already available")
        return True
    except ImportError:
        print("📦 Installing Lag-Llama from GitHub...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/time-series-foundation-models/lag-llama.git",
                "--quiet"
            ], check=True)
            print("✅ Successfully installed Lag-Llama")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing Lag-Llama: {e}")
            return False

def download_model_weights():
    """Download Lag-Llama model weights."""
    print("⬇️ Downloading Lag-Llama model weights...")
    
    model_path = "lag-llama.ckpt"
    if os.path.exists(model_path):
        print("✅ Model weights already exist")
        return model_path
    
    try:
        subprocess.run([
            "huggingface-cli", "download", 
            "time-series-foundation-models/Lag-Llama", 
            "lag-llama.ckpt", 
            "--local-dir", "."
        ], check=True, capture_output=True)
        print("✅ Successfully downloaded model weights")
        return model_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading weights: {e}")
        raise

def load_nvda_data():
    """Load and preprocess NVDA dataset from Hugging Face (same as other models)."""
    print("🚀 Downloading NVDA dataset from Hugging Face...")
    print(f"Dataset: {DATASET_NAME}")
    
    try:
        # Load dataset
        dataset = load_dataset(DATASET_NAME, split="train")
        df = dataset.to_pandas()
        print("✅ Successfully downloaded NVDA dataset from Hugging Face")
        
        # Display basic info
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Convert timestamp and set up datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'timestamp': 'ds',
            'close': 'price'  # Keep as 'price' for consistency, will rename for GluonTS later
        })
        
        # Handle timezone conversion properly
        if df['ds'].dt.tz is None:
            df['ds'] = df['ds'].dt.tz_localize('UTC')
        
        # Convert to ET for proper trading hours filtering
        df['ds'] = df['ds'].dt.tz_convert('US/Eastern')
        
        # Filter for ET trading hours (9:30 AM - 4:00 PM ET)
        df['hour'] = df['ds'].dt.hour
        df['minute'] = df['ds'].dt.minute
        df['time_decimal'] = df['hour'] + df['minute'] / 60
        
        # Keep only trading hours (9:30 AM = 9.5, 4:00 PM = 16.0)
        df = df[(df['time_decimal'] >= 9.5) & (df['time_decimal'] <= 16.0)].copy()
        df = df.drop(['hour', 'minute', 'time_decimal'], axis=1)
        df = df.reset_index(drop=True)
        
        # Fill any NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"\nDataset info:")
        print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
        print(f"Total samples: {len(df):,}")
        print(f"Sample data:")
        print(df[['ds', 'price', 'volume', 'high', 'low', 'open', 'vwap', 'trade_count']].head())
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def select_test_period(df):
    """Select test period based on volatility analysis (same as other models)."""
    total_samples = len(df)
    print(f"   Total available samples: {total_samples:,}")
    
    if USE_VOLATILE_PERIOD:
        # Use Period 43 - the volatile period used by other models
        test_start_idx = 99022
        test_end_idx = test_start_idx + TEST_SIZE
        period_name = "Period 43 (Volatile)"
        period_type = "HIGH-VOLATILITY"
        
        # Ensure we don't go beyond dataset bounds
        if test_end_idx > total_samples:
            test_start_idx = total_samples - TEST_SIZE
            test_end_idx = total_samples
    else:
        # Use Period 0 - original flat period
        test_start_idx = TRAINING_SIZE
        test_end_idx = test_start_idx + TEST_SIZE
        period_name = "Period 0"
        period_type = "FLAT"
    
    print(f"   Selected test period: {period_name}")
    print(f"   Test indices: {test_start_idx} to {test_end_idx}")
    print(f"   Period type: {period_type}")
    
    return test_start_idx, test_end_idx, period_name

def prepare_data_split(df):
    """Prepare train/test split with proper period selection (same as other models)."""
    print("\n📊 Preparing data split...")
    
    # Select test period
    test_start_idx, test_end_idx, period_name = select_test_period(df)
    
    # Create train/test split
    train_df = df.iloc[:test_start_idx].copy()
    test_df = df.iloc[test_start_idx:test_end_idx].copy()
    
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    print(f"   Test period: {period_name}")
    print(f"   Test date range: {test_df['ds'].min()} to {test_df['ds'].max()}")
    
    # Calculate test period statistics
    test_price_range = test_df['price'].max() - test_df['price'].min()
    test_price_std = test_df['price'].std()
    test_returns = test_df['price'].pct_change().dropna()
    test_returns_std = test_returns.std()
    
    print(f"   Test price range: ${test_df['price'].min():.2f} - ${test_df['price'].max():.2f} (${test_price_range:.2f})")
    print(f"   Test price std: ${test_price_std:.3f}")
    print(f"   Test returns std: {test_returns_std:.4f} ({test_returns_std*100:.3f}%)")
    
    return train_df, test_df

def prepare_data_for_gluonts(train_df):
    """Prepare data in GluonTS format for Lag-Llama."""
    print("\n📊 Preparing data for GluonTS...")
    
    try:
        import numpy as np
        
        # For Lag-Llama with financial data, we need to create a proper time series format
        # that handles non-uniform timestamps (market closures, weekends, etc.)
        
        # Create a simple dictionary format dataset
        # This avoids the timestamp uniformity issues with PandasDataset
        train_data = []
        
        # Convert the DataFrame to a simple time series format
        # GluonTS expects numpy arrays, not Python lists
        # Convert to float32 for MPS compatibility (Apple Silicon doesn't support float64)
        target_values = train_df['price'].values.astype(np.float32)  # numpy array, float32
        
        # Create a Period object with frequency information (required by GluonTS)
        start_timestamp = pd.Timestamp(train_df['ds'].iloc[0])
        start_period = pd.Period(start_timestamp, freq='5min')
        
        # Create a data entry in the format expected by GluonTS
        data_entry = {
            'target': target_values,
            'start': start_period,  # Use Period object with freq
            'item_id': 'NVDA'
        }
        
        train_data.append(data_entry)
        
        print(f"   Training samples: {len(target_values):,}")
        print(f"   Target range: ${target_values.min():.2f} - ${target_values.max():.2f}")
        print(f"   Target std: ${target_values.std():.3f}")
        print(f"   Start period: {start_period}")
        print(f"   Target shape: {target_values.shape}")
        
        return train_data
        
    except ImportError as e:
        print(f"❌ Error importing GluonTS: {e}")
        print("Make sure GluonTS is installed: pip install gluonts")
        raise

def create_lagllama_estimator(model_path):
    """Create and configure Lag-Llama estimator."""
    print("\n🏗️ Creating Lag-Llama estimator...")
    
    try:
        from lag_llama.gluon.estimator import LagLlamaEstimator
        
        # Load checkpoint to get model parameters
        ckpt = torch.load(model_path, map_location=DEVICE)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        
        print(f"   Model parameters: {estimator_args}")
        
        # Create estimator for fine-tuning
        estimator = LagLlamaEstimator(
            ckpt_path=model_path,
            prediction_length=PREDICTION_LENGTH,
            context_length=CONTEXT_LENGTH,
            
            # Fine-tuning parameters
            lr=LEARNING_RATE,
            aug_prob=0,  # No data augmentation
            nonnegative_pred_samples=True,  # Ensure positive price predictions
            
            # Model architecture (from checkpoint)
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            
            # RoPE scaling for different context lengths
            rope_scaling={
                "type": "linear",
                "factor": max(1.0, (CONTEXT_LENGTH + PREDICTION_LENGTH) / estimator_args["context_length"]),
            },
            
            # Training configuration
            batch_size=BATCH_SIZE,
            num_parallel_samples=NUM_SAMPLES,
            device=torch.device(DEVICE),
            
            # Lightning trainer arguments
            trainer_kwargs={
                "max_epochs": NUM_EPOCHS,
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "logger": False,  # Disable logging for cleaner output
                "accelerator": "cpu",  # Force CPU usage, avoid MPS issues
            }
        )
        
        print("✅ Successfully created Lag-Llama estimator")
        return estimator
        
    except Exception as e:
        print(f"❌ Error creating estimator: {e}")
        raise

def fine_tune_model(estimator, train_data):
    """Fine-tune Lag-Llama model on NVDA data (same pattern as other models)."""
    print("\n🎯 Fine-tuning Lag-Llama on NVDA data...")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Context length: {CONTEXT_LENGTH}")
    print(f"   Prediction length: {PREDICTION_LENGTH}")
    
    try:
        # Fine-tune the model
        predictor = estimator.train(
            training_data=train_data,
            cache_data=True,
            shuffle_buffer_length=1000
        )
        
        print("✅ Successfully fine-tuned Lag-Llama model")
        return predictor
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        raise

def evaluate_model(predictor, df, test_start_idx, test_end_idx):
    """Evaluate the fine-tuned model (same pattern as other models)."""
    print("\n📈 Evaluating fine-tuned Lag-Llama model...")
    
    try:
        from gluonts.evaluation import make_evaluation_predictions, Evaluator
        import numpy as np
        
        # Prepare test data - only use PREDICTION_LENGTH samples for fair comparison
        # since the model can only predict PREDICTION_LENGTH steps ahead
        available_test_samples = test_end_idx - test_start_idx
        actual_test_samples = min(PREDICTION_LENGTH, available_test_samples)
        
        print(f"   Available test samples: {available_test_samples}")
        print(f"   Using test samples: {actual_test_samples} (limited by prediction length)")
        
        # Adjust test indices to use only what we can predict
        adjusted_test_end_idx = test_start_idx + actual_test_samples
        test_df = df.iloc[test_start_idx:adjusted_test_end_idx].copy()
        
        # Create evaluation dataset using the same simple format
        eval_data = []
        
        # Include training data up to test point for context
        eval_df = df.iloc[:test_start_idx].copy()  # Only training data for context
        target_values = eval_df['price'].values.astype(np.float32)  # numpy array, float32 for MPS
        
        # Create Period object with frequency information
        start_timestamp = pd.Timestamp(eval_df['ds'].iloc[0])
        start_period = pd.Period(start_timestamp, freq='5min')
        
        data_entry = {
            'target': target_values,
            'start': start_period,  # Use Period object with freq
            'item_id': 'NVDA'
        }
        
        eval_data.append(data_entry)
        
        # Generate predictions
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=eval_data,
            predictor=predictor,
            num_samples=NUM_SAMPLES
        )
        
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        print(f"   Generated {len(forecasts)} forecasts")
        
        # Get the forecast
        forecast = forecasts[0]
        
        # Get actual vs predicted values for detailed analysis
        actual_values = test_df['price'].values
        predicted_values = forecast.mean  # Mean of probabilistic forecast
        
        # Ensure shapes match - take only the length we can compare
        min_length = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_length]
        predicted_values = predicted_values[:min_length]
        
        print(f"   Actual values shape: {actual_values.shape}")
        print(f"   Predicted values shape: {predicted_values.shape}")
        print(f"   Comparison length: {min_length}")
        
        # Calculate metrics manually to avoid GluonTS evaluation issues
        mae = np.mean(np.abs(predicted_values - actual_values))
        rmse = np.sqrt(np.mean((predicted_values - actual_values) ** 2))
        mape = np.mean(np.abs((predicted_values - actual_values) / actual_values)) * 100
        
        # Calculate additional metrics
        directional_accuracy = calculate_directional_accuracy(actual_values, predicted_values)
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'directional_accuracy': float(directional_accuracy),
            'test_samples': len(actual_values),
            'prediction_length': PREDICTION_LENGTH,
            'context_length': CONTEXT_LENGTH
        }
        
        print(f"\n📊 Evaluation Results:")
        print(f"   MAE: ${mae:.2f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
        print(f"   Test samples used: {len(actual_values)}")
        
        # Update test_df to match the actual comparison length
        test_df = test_df.iloc[:min_length].copy()
        
        return metrics, forecast, actual_values, predicted_values, test_df
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise

def calculate_directional_accuracy(actual, predicted):
    """Calculate directional accuracy (up/down prediction accuracy)."""
    if len(actual) < 2 or len(predicted) < 2:
        return 0.0
    
    actual_directions = np.diff(actual) > 0
    predicted_directions = np.diff(predicted) > 0
    
    correct_directions = actual_directions == predicted_directions
    accuracy = np.mean(correct_directions) * 100
    
    return accuracy

def setup_directories():
    """Setup output directories (same as other models)."""
    print("\n📁 Setting up directories...")
    
    plot_dir = "plots/lagllama/nvda"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"   Created directory: {plot_dir}")
    return plot_dir

def create_visualizations(metrics, forecast, actual_values, predicted_values, test_df):
    """Create comprehensive visualizations (same pattern as other models)."""
    print("\n📊 Creating visualizations...")
    
    # Setup directories
    plot_dir = setup_directories()
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Main analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'NVDA Lag-Llama Fine-tuned Model Analysis (MAE: ${metrics["MAE"]:.2f})', 
                 fontsize=16, fontweight='bold')
    
    # Subplot 1: Predictions vs Actual with confidence intervals
    ax1 = axes[0, 0]
    time_index = range(len(actual_values))
    
    ax1.plot(time_index, actual_values, 'b-', linewidth=2, label='Actual Price', alpha=0.8)
    ax1.plot(time_index, predicted_values, 'r--', linewidth=2, label='Predicted Price', alpha=0.8)
    
    # Add confidence intervals if available
    if hasattr(forecast, 'quantile'):
        try:
            q10 = forecast.quantile(0.1)
            q90 = forecast.quantile(0.9)
            ax1.fill_between(time_index, q10, q90, alpha=0.2, color='red', label='80% Confidence Interval')
        except:
            pass  # Skip if quantiles not available
    
    ax1.set_title('Price Predictions vs Actual', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps (5-min intervals)')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Prediction errors over time
    ax2 = axes[0, 1]
    errors = predicted_values - actual_values
    ax2.plot(time_index, errors, 'g-', linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.fill_between(time_index, errors, 0, alpha=0.3, color='green')
    ax2.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps (5-min intervals)')
    ax2.set_ylabel('Error ($)')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Error distribution
    ax3 = axes[1, 0]
    ax3.hist(errors, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(errors):.2f}')
    ax3.axvline(x=np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: ${np.median(errors):.2f}')
    ax3.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Prediction Error ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Model performance metrics
    ax4 = axes[1, 1]
    metrics_names = ['MAE ($)', 'RMSE ($)', 'MAPE (%)', 'Dir. Acc. (%)']
    metrics_values = [metrics['MAE'], metrics['RMSE'], metrics['MAPE'], metrics['directional_accuracy']]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/NVDA_LagLlama_main_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Directional accuracy analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('NVDA Lag-Llama Directional Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # Calculate directions
    actual_directions = np.diff(actual_values) > 0
    predicted_directions = np.diff(predicted_values) > 0
    
    # Direction comparison
    direction_labels = ['Down', 'Up']
    actual_counts = [np.sum(~actual_directions), np.sum(actual_directions)]
    predicted_counts = [np.sum(~predicted_directions), np.sum(predicted_directions)]
    
    x = np.arange(len(direction_labels))
    width = 0.35
    
    ax1.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8, color='blue')
    ax1.bar(x + width/2, predicted_counts, width, label='Predicted', alpha=0.8, color='red')
    ax1.set_title('Direction Prediction Comparison')
    ax1.set_xlabel('Direction')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(direction_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confusion matrix for directions
    correct_directions = actual_directions == predicted_directions
    confusion_data = np.array([[np.sum((~actual_directions) & (~predicted_directions)),
                               np.sum((~actual_directions) & predicted_directions)],
                              [np.sum(actual_directions & (~predicted_directions)),
                               np.sum(actual_directions & predicted_directions)]])
    
    im = ax2.imshow(confusion_data, interpolation='nearest', cmap='Blues')
    ax2.set_title('Direction Confusion Matrix')
    ax2.set_xlabel('Predicted Direction')
    ax2.set_ylabel('Actual Direction')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Down', 'Up'])
    ax2.set_yticklabels(['Down', 'Up'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, confusion_data[i, j], ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/NVDA_LagLlama_directional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed time series plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Create time labels
    time_labels = test_df['ds'].dt.strftime('%H:%M').values
    x_positions = range(len(actual_values))
    
    ax.plot(x_positions, actual_values, 'b-', linewidth=2, label='Actual Price', marker='o', markersize=4)
    ax.plot(x_positions, predicted_values, 'r--', linewidth=2, label='Predicted Price', marker='s', markersize=4)
    
    # Highlight significant errors
    errors = predicted_values - actual_values
    significant_errors = np.abs(errors) > np.std(errors)
    if np.any(significant_errors):
        ax.scatter(np.array(x_positions)[significant_errors], 
                  actual_values[significant_errors], 
                  color='orange', s=100, alpha=0.7, label='Significant Errors', zorder=5)
    
    ax.set_title(f'NVDA Price Prediction Detail (MAE: ${metrics["MAE"]:.2f}, Dir. Acc: {metrics["directional_accuracy"]:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set x-axis labels to show time
    step = max(1, len(time_labels) // 10)  # Show ~10 labels
    ax.set_xticks(x_positions[::step])
    ax.set_xticklabels(time_labels[::step], rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/NVDA_LagLlama_detailed_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {plot_dir}/")

def save_results(metrics, actual_values, predicted_values, test_df):
    """Save results to JSON file (same pattern as other models)."""
    print("\n💾 Saving results...")
    
    # Prepare results data
    results = {
        'model_name': 'Lag-Llama Fine-tuned',
        'dataset': SYMBOL,
        'test_period': 'Period 43 (Volatile)' if USE_VOLATILE_PERIOD else 'Period 0',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'prediction_length': PREDICTION_LENGTH,
            'context_length': CONTEXT_LENGTH,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'num_samples': NUM_SAMPLES
        },
        'metrics': metrics,
        'predictions': {
            'actual_values': actual_values.tolist(),
            'predicted_values': predicted_values.tolist(),
            'timestamps': test_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        },
        'data_statistics': {
            'actual_mean': float(np.mean(actual_values)),
            'actual_std': float(np.std(actual_values)),
            'actual_min': float(np.min(actual_values)),
            'actual_max': float(np.max(actual_values)),
            'predicted_mean': float(np.mean(predicted_values)),
            'predicted_std': float(np.std(predicted_values)),
            'error_mean': float(np.mean(predicted_values - actual_values)),
            'error_std': float(np.std(predicted_values - actual_values))
        }
    }
    
    # Save to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"nvda_lagllama_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {filename}")
    return filename

def main():
    """Main execution function (same pattern as other models)."""
    print("🚀 NVDA Time Series Forecasting with Lag-Llama Fine-tuning")
    print("=" * 70)
    
    try:
        # Step 1: Setup Lag-Llama
        if not setup_lagllama():
            print("❌ Failed to setup Lag-Llama")
            return
        
        # Step 2: Download model weights
        model_path = download_model_weights()
        
        # Step 3: Load and prepare data
        df = load_nvda_data()
        train_df, test_df = prepare_data_split(df)
        train_data = prepare_data_for_gluonts(train_df)
        
        # Step 4: Create estimator
        estimator = create_lagllama_estimator(model_path)
        
        # Step 5: Fine-tune model
        predictor = fine_tune_model(estimator, train_data)
        
        # Step 6: Evaluate model
        test_start_idx = len(train_df)
        test_end_idx = test_start_idx + TEST_SIZE
        metrics, forecast, actual_values, predicted_values, test_df = evaluate_model(
            predictor, df, test_start_idx, test_end_idx)
        
        # Step 7: Create visualizations
        create_visualizations(metrics, forecast, actual_values, predicted_values, test_df)
        
        # Step 8: Save results
        results_file = save_results(metrics, actual_values, predicted_values, test_df)
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 NVDA Lag-Llama Analysis Complete!")
        print("=" * 70)
        print(f"📊 Final Results:")
        print(f"   Model: Lag-Llama Fine-tuned")
        print(f"   Test Period: {'Period 43 (Volatile)' if USE_VOLATILE_PERIOD else 'Period 0'}")
        print(f"   MAE: ${metrics['MAE']:.2f}")
        print(f"   RMSE: ${metrics['RMSE']:.2f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
        print(f"   Test Samples: {metrics['test_samples']}")
        print(f"   Results saved to: {results_file}")
        print("✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 