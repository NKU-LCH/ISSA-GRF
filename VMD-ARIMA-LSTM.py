import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vmdpy import VMD  # pip install vmdpy

# Data loading and preprocessing
# Modify the file path to your Excel data path
file_path = "data.xlsx"
data = pd.read_excel(file_path)

# Convert the date to datetime format and sort by date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

# For the same LME, calculate the average of all points on the same date (for time series model)
lme_data = data.groupby(['LME', 'Date'])['MPB'].mean().reset_index()

def preprocess_data(df):
    """Sort by date and set Date as index"""
    df = df.set_index('Date')
    df = df.sort_index()
    return df

# VMD Decomposition Function
def vmd_decompose(signal, alpha=2000, tau=0, K=5, DC=0, init=1, tol=1e-7):
    """
    Decompose a 1D signal using VMD
    Parameters:
      signal: 1D array (e.g., standardized time series)
      alpha, tau, K, DC, init, tol: VMD parameters (can be adjusted based on data)
    Returns:
      u: Decomposed modes, shape (K, len(signal))
    """
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    return u

# Hybrid Model Function: VMD + ARIMA + LSTM (with in-sample performance metrics)
def hybrid_forecast_with_vmd(df, n_steps, lstm_window_size=60, lstm_epochs=10, lstm_batch_size=32, K=5):
    """
    For a single LME's time series data (df requires 'MPB' column, Date as index), build a hybrid model:
      1. Decompose the standardized data into K modes using VMD;
      2. For each mode:
           - Use ARIMA model to capture the linear trend and forecast the next n_steps;
           - Compute the ARIMA residuals and use LSTM model to fit the residuals and predict future part (iterative forecast);
           - Hybrid forecast = ARIMA forecast + LSTM forecast residuals;
      3. Combine the in-sample predictions for each IMF and calculate performance metrics;
      4. Finally, sum the future forecasts of each IMF and transform back to original scale.
    
    Returns:
      future_forecast_val: Forecasted values for future n_steps (original scale)
      in_sample_pred: In-sample hybrid predictions (original scale, for evaluation)
      metrics: Performance metrics dictionary (mse, rmse, mae, r2)
      scaler: MinMaxScaler object for standardization/inverse transformation
    """
    # Extract original data and standardize
    values = df['MPB'].values  # Original data shape: (n, )
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))
    
    # Use VMD to decompose the standardized signal into K modes
    imfs = vmd_decompose(scaled_values.flatten(), alpha=2000, tau=0, K=K, DC=0, init=1, tol=1e-7)
    
    # Initialize future forecast (standardized space)
    future_forecast_scaled = np.zeros((n_steps, 1))
    
    # Store in-sample predictions and true values (standardized)
    in_sample_list = []         # In-sample predictions for each IMF
    actual_in_sample_list = []  # True values for each IMF
    
    # Model and predict for each IMF component
    for i in range(K):
        imf = imfs[i, :].reshape(-1, 1)
        # ARIMA modeling (capture linear part)
        try:
            arima_model = ARIMA(imf, order=(5, 1, 0))
            arima_fit = arima_model.fit()
        except Exception as e:
            print(f"IMF {i+1}: ARIMA model fitting error: {e}")
            continue
        
        # ARIMA fitted values are shorter than original data (due to differencing), remember to calculate lag
        lag = len(imf) - len(arima_fit.fittedvalues)
        arima_fitted = arima_fit.fittedvalues.reshape(-1, 1)
        
        # Compute residuals and use LSTM to fit the residuals in-sample
        residuals = imf[lag:] - arima_fitted  # In-sample residuals (standardized)
        residuals = residuals.flatten()
        
        if len(residuals) <= lstm_window_size:
            print(f"IMF {i+1}: Not enough residual data to construct LSTM window (residual length: {len(residuals)}, window size: {lstm_window_size})")
            continue
        
        # Construct LSTM sliding window data
        X_res, y_res = [], []
        for j in range(lstm_window_size, len(residuals)):
            X_res.append(residuals[j - lstm_window_size:j])
            y_res.append(residuals[j])
        X_res = np.array(X_res)
        y_res = np.array(y_res)
        X_res = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))
        
        # Build and train LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=100, input_shape=(lstm_window_size, 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_res, y_res, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=0)
        
        # In-sample LSTM prediction for residuals
        lstm_in_sample_res_pred = lstm_model.predict(X_res, verbose=0)
        # In-sample hybrid prediction = ARIMA fitted values (from lstm_window_size onwards) + LSTM predicted residuals
        imf_in_sample_scaled = arima_fitted[lstm_window_size:] + lstm_in_sample_res_pred
        # Corresponding true values (from lag + lstm_window_size onwards)
        actual_imf_in_sample = imf[lag + lstm_window_size:]
        
        in_sample_list.append(imf_in_sample_scaled)
        actual_in_sample_list.append(actual_imf_in_sample)
        
        # Future n_steps prediction
        # Use ARIMA's get_forecast method to obtain the predicted mean for the next n_steps
        forecast_obj = arima_fit.get_forecast(steps=n_steps)
        forecast_arima_scaled = forecast_obj.predicted_mean.reshape(-1, 1)
        
        # LSTM residual forecast for the future (iterative approach)
        lstm_forecast_residuals = []
        last_window = residuals[-lstm_window_size:]
        last_window = last_window.reshape(1, lstm_window_size, 1)
        for _ in range(n_steps):
            pred_res = lstm_model.predict(last_window, verbose=0)
            lstm_forecast_residuals.append(pred_res[0, 0])
            last_window = np.concatenate([last_window[:, 1:, :],
                                          np.array([[[pred_res[0, 0]]]])], axis=1)
        lstm_forecast_residuals = np.array(lstm_forecast_residuals).reshape(-1, 1)
        
        # Hybrid forecast = ARIMA forecast + LSTM forecast residuals
        imf_future_forecast_scaled = forecast_arima_scaled + lstm_forecast_residuals
        
        # Sum future forecasts of all IMFs
        future_forecast_scaled += imf_future_forecast_scaled

    # Combine in-sample predictions and compute performance metrics
    if len(in_sample_list) == 0:
        print("No in-sample predictions generated!")
        return None, None, None, None

    L_min = min(arr.shape[0] for arr in in_sample_list)
    combined_in_sample_scaled = np.zeros((L_min, 1))
    combined_actual_in_sample = np.zeros((L_min, 1))
    for pred, actual in zip(in_sample_list, actual_in_sample_list):
        combined_in_sample_scaled += pred[:L_min]
        combined_actual_in_sample += actual[:L_min]
    
    # Transform back to original scale
    in_sample_pred = scaler.inverse_transform(combined_in_sample_scaled)
    actual_in_sample = scaler.inverse_transform(combined_actual_in_sample)
    future_forecast_val = scaler.inverse_transform(future_forecast_scaled)
    
    # Compute in-sample performance metrics
    mse = mean_squared_error(actual_in_sample, in_sample_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_in_sample, in_sample_pred)
    r2 = r2_score(actual_in_sample, in_sample_pred)
    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    return future_forecast_val, in_sample_pred, metrics, scaler

# Process for each LME, output performance metrics and results visualization
grouped_lme = lme_data.groupby('LME')

for lme, group in grouped_lme:
    # Sort the data by date and set index for each LME
    group = group.sort_values(by='Date').set_index('Date')
    print(f"Processing LME: {lme}")
    
    # Set forecast steps (e.g., 60 periods in the future)
    n_steps = 60
    result = hybrid_forecast_with_vmd(group, n_steps=n_steps, lstm_window_size=60,
                                      lstm_epochs=10, lstm_batch_size=32, K=5)
    if result is None:
        print(f"Model construction failed for LME {lme}.")
        continue
    # Unpack the returned results (future forecast, in-sample forecast, performance metrics, scaler)
    future_forecast_vals, in_sample_pred, metrics, scaler = result
    
    # Output in-sample performance metrics
    print(f"LME: {lme} in-sample performance metrics:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R2: {metrics['r2']:.4f}")
    
    # Compute horizontal line values:
    # Line 1: The historical average of MPB for all records of this LME in the original data
    historical_avg = data[data['LME'] == lme]['MPB'].mean()
    # Line 2: The average value of the future n_steps forecast
    forecast_avg = np.mean(future_forecast_vals)
    
    # Output specific values and growth rate
    print(f"Historical average: {historical_avg:.4f}")
    print(f"Forecasted average for next 60 periods: {forecast_avg:.4f}")
    growth_rate = (forecast_avg - historical_avg) / historical_avg * 100
    print(f"Growth rate: {growth_rate:.2f}%")
    
    # Visualization results
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    # Thicken the border lines of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Plot the historical data (color: #1963a2)
    plt.plot(group.index, group['MPB'], color="#1963a2", linewidth=2)
    # Use the last len(in_sample_pred) dates of the original series as in-sample x-axis and plot in-sample predictions (color: #864925)
    in_sample_dates = group.index[-len(in_sample_pred):]
    plt.plot(in_sample_dates, in_sample_pred, linestyle='--', linewidth=2, color="#864925")
    
    # Generate future forecast dates (assuming monthly data): From the last historical date, go forward n_steps months
    future_dates = pd.date_range(group.index[-1], periods=n_steps + 1, freq='M')[1:]
    plt.plot(future_dates, future_forecast_vals, linestyle='-.', linewidth=2, color="#b7282e")
    
    # Plot horizontal dashed lines
    plt.axhline(y=historical_avg, color='#4b0082', linestyle='--', linewidth=2,
                label=f'Historical Avg: {historical_avg:.2f}')
    plt.axhline(y=forecast_avg, color='#006400', linestyle='--', linewidth=2,
                label=f'Forecast Avg: {forecast_avg:.2f}')
    
    # Label the x-axis with LME number
    plt.xlabel(f'Date (LME {lme})', fontdict={'fontname': 'Times New Roman', 'fontsize':20, 'fontweight':'bold'})
    plt.ylabel('Microplastic concentration (items/unit)', 
               fontdict={'fontname': 'Times New Roman', 'fontsize':20, 'fontweight':'bold'})
    
    # Add dashed grid lines
    plt.grid(True, linestyle='--', linewidth=1)
    
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()
