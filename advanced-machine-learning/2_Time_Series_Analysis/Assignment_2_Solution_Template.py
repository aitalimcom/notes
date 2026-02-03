
"""
Assignment 2: Comprehensive Time Series Analysis Pipeline
------------------------------------------------------
This assignment requires the implementation of a complete time series analysis pipeline
on 3-4 different datasets.

Group Members: [Your Names Here]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf # Added kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX # For SARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression # For linear trend
import pymannkendall as mk # For Mann-Kendall test
# pip install pymannkendall if not already installed

# Set style
plt.style.use('seaborn-v0_8')
np.random.seed(42)

# ==============================================================================
# 1. Data Loading and Investigation
# ==============================================================================
def load_and_explore_datasets():
    """
    Load 3-4 datasets and perform initial investigation.
    Returns a dictionary of dataframes.
    
    Suggested datasets (available in statsmodels or easily downloadable):
    1. Mauna Loa CO2 (Trend + Seasonality)
    2. Air Passengers (Multiplicative Seasonality)
    3. Daily stock price (Random Walk / Stochastic Trend)
    4. Sunspots (Cyclical)
    """
    datasets = {}
    
    # Example 1: Air Passengers
    try:
        air_passengers = sm.datasets.get_rdataset("AirPassengers").data
        # Convert index to meaningful time index if needed
        # air_passengers['time'] = pd.date_range(start='1949-01-01', periods=len(air_passengers), freq='M')
        # air_passengers.set_index('time', inplace=True)
        datasets['AirPassengers'] = air_passengers['value'] 
    except:
        print("Could not load AirPassengers")

    # Example 2: CO2
    try:
        co2 = sm.datasets.co2.load_pandas().data
        co2 = co2.fillna(method='ffill') # Handle missing values
        datasets['CO2'] = co2['co2']
    except:
        print("Could not load CO2")
        
    # Example 3: Sunspots
    try:
        sunspots = sm.datasets.sunspots.load_pandas().data
        # sunspots['YEAR'] = pd.to_datetime(sunspots['YEAR'], format='%Y')
        # sunspots.set_index('YEAR', inplace=True)
        datasets['Sunspots'] = sunspots['SUNACTIVITY']
    except:
        print("Could not load Sunspots")

    # Add your own custom loading here
    
    # Initial Plots
    for name, data in datasets.items():
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(f"{name} - Raw Data")
        plt.show()
        
    return datasets

# ==============================================================================
# 2. Trend Detection
# ==============================================================================
def detect_trends(series, name="Series"):
    print(f"\n--- Trend Detection for {name} ---")
    
    # Method 1: Linear Model (y = b0 + b1*t + e)
    n = len(series)
    X = np.arange(n).reshape(-1, 1)
    y = series.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate stats for p-value (standard linear regression stats)
    # Using statsmodels OLS for easier p-value extraction
    X_sm = sm.add_constant(X)
    est = sm.OLS(y, X_sm)
    est2 = est.fit()
    
    slope = est2.params[1]
    p_value = est2.pvalues[1]
    r_squared = est2.rsquared
    
    print(f"1. Linear Trend Test:")
    print(f"   Slope: {slope:.4f}, p-value: {p_value:.4e}, R2: {r_squared:.4f}")
    print(f"   Significant? {'Yes' if p_value < 0.05 else 'No'}")
    
    # Method 2: Mann-Kendall Test (Non-parametric)
    try:
        mk_result = mk.original_test(series)
        print(f"2. Mann-Kendall Test:")
        print(f"   Trend: {mk_result.trend}, p-value: {mk_result.p}, Slope: {mk_result.slope:.4f}")
    except Exception as e:
        print(f"2. Mann-Kendall Test: Failed ({str(e)})")

    # Method 3: Smoothing (Visualization)
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original', alpha=0.5)
    
    # A. LOESS (using lowess from statsmodels)
    # Bandwidths = 0.1, 0.3, 0.5 (fraction)
    for frac in [0.1, 0.3, 0.5]:
        # returns (x, y)
        loess_smoothed = sm.nonparametric.lowess(series.values, np.arange(n), frac=frac)
        plt.plot(series.index if hasattr(series, 'index') else np.arange(n), 
                 loess_smoothed[:, 1], label=f'LOESS (frac={frac})', linestyle='--')
        
    # B. EWMA (Exponentially Weighted Moving Average)
    # decay factors (alpha) = 0.1, 0.3, 0.9
    for alpha in [0.1, 0.3, 0.9]:
        ewma = series.ewm(alpha=alpha).mean()
        # plt.plot(ewma, label=f'EWMA (alpha={alpha})') # Optional to plot these too
        
    plt.title(f"{name} Trend Analysis (Linear, LOESS)")
    plt.legend()
    plt.show()

# ==============================================================================
# 3. Detrending Methods
# ==============================================================================
def apply_detrending(series, name="Series"):
    """
    Apply 4 methods and return the results.
    1. Linear detrending (subtract fit)
    2. Differencing (1st and 2nd)
    3. LOESS subtraction
    """
    n = len(series)
    X = np.arange(n).reshape(-1, 1)
    
    # 1. Linear Detrending
    model = LinearRegression().fit(X, series.values)
    linear_trend = model.predict(X)
    detrended_linear = series - linear_trend
    
    # 2. Differencing
    diff_1 = series.diff().dropna()
    diff_2 = diff_1.diff().dropna()
    
    # 3. LOESS Detrending (using a reasonable frac, e.g., 0.3)
    loess_res = sm.nonparametric.lowess(series.values, np.arange(n), frac=0.3)
    loess_trend = pd.Series(loess_res[:, 1], index=series.index)
    detrended_loess = series - loess_trend
    
    return {
        'Linear': detrended_linear,
        'Diff_1': diff_1,
        'Diff_2': diff_2,
        'LOESS': detrended_loess
    }

# ==============================================================================
# 4. Seasonality Analysis
# ==============================================================================
def analyze_seasonality(series, name="Series", freq=12):
    print(f"\n--- Seasonality Analysis for {name} ---")
    
    # 1. Decomposition (Additive vs Multiplicative)
    # Check for multiplicative (variance increases with level)
    try:
        decomp_add = seasonal_decompose(series, model='additive', period=freq)
        decomp_mult = seasonal_decompose(series, model='multiplicative', period=freq)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        # Plotting code for decomposition components...
        decomp_add.plot()
        plt.suptitle(f"{name} Additive Decomposition")
        plt.show()
        
        # Determine based on extensive residuals which is better (visual inspection or residual variance)
    except Exception as e:
        print(f"Decomposition failed: {e}")

    # 2. Fourier Transform (Frequency domain)
    # Identify dominant frequencies
    fft_vals = np.fft.rfft(series.values)
    fft_freqs = np.fft.rfftfreq(len(series))
    
    # Initial index 0 is the DC component (mean), ignore it
    magnitude = np.abs(fft_vals)
    dominant_indices = np.argsort(magnitude[1:])[-3:][::-1] + 1 # Top 3 frequencies
    
    print("Dominant Periods (1/freq):")
    for idx in dominant_indices:
        f = fft_freqs[idx]
        if f > 0:
            print(f"   Period: {1/f:.2f}")

# ==============================================================================
# 5. Stationarity Tests
# ==============================================================================
def test_stationarity(series, name="Series"):
    # ADF Test
    print(f"Stationarity Tests for {name}:")
    adf_result = adfuller(series.dropna())
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    
    # KPSS Test
    # Null hypothesis: Process is trend stationary
    kpss_result = kpss(series.dropna(), regression='c', nlags="auto")
    print(f"  KPSS Statistic: {kpss_result[0]:.4f}")
    print(f"  p-value: {kpss_result[1]:.4f}") # p < 0.05 means non-stationary
    
    # Visual Inspection Code (mean/variance check)
    pass

# ==============================================================================
# 6. Modeling (ARIMA / SARIMA)
# ==============================================================================
def fit_arima_models(train, test, p_values, d_value, q_values):
    # Grid Search logic for AIC/BIC
    results_table = []
    
    for p in p_values:
        for q in q_values:
            try:
                model = ARIMA(train, order=(p, d_value, q))
                res = model.fit()
                results_table.append({
                    'order': (p, d_value, q),
                    'aic': res.aic,
                    'bic': res.bic,
                    'model': res
                })
            except:
                continue
                
    # Sort and pick best
    results_table.sort(key=lambda x: x['aic'])
    best_model_info = results_table[0]
    
    print(f"\nBest Model: ARIMA{best_model_info['order']} with AIC: {best_model_info['aic']:.2f}")
    
    # Residual Analysis for best model
    residuals = best_model_info['model'].resid
    # Plot ACF of residuals, Q-Q plot, etc.
    
    return best_model_info['model']

# ==============================================================================
# 7. RNN / Deep Learning (Optional)
# ==============================================================================
def fit_rnn_models(series):
    if len(series) < 500:
        print("Dataset too small for RNNs (<500 samples). Skipping.")
        return
        
    print("\n--- Fitting RNN/LSTM/GRU ---")
    # Data Prep: Scaling (MinMax), Sequence creation (sliding window)
    # Train/Test Split
    # Pytorch or Tensorflow/Keras implementation
    # Compare RMSE with ARIMA
    pass

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # 1. Load
    data_dict = load_and_explore_datasets()
    
    for name, series in data_dict.items():
        print(f"Processing {name}...")
        
        # 2. trends
        detect_trends(series, name)
        
        # 3. Detrending
        detrended_dict = apply_detrending(series, name)
        
        # 4. Stationarity on detrended
        # Select best detrended version (e.g., Diff_1)
        best_series = detrended_dict['Diff_1'] 
        test_stationarity(best_series, f"{name} (Diff_1)")
        
        # 5. Modeling
        # Split train/test
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        # Decide d based on differencing needs
        d = 1 # Simplified assumption
        
        # Fit models
        # fit_arima_models(train, test, range(0,3), d, range(0,3))
        
        print("="*50)

