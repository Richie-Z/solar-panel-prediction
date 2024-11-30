import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from enums import (
    DatasetColumns,
    WeatherDatasetColumns,
)

from typing import List
from diskcache import Cache
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


def check_stationarity(series: pd.Series, significance_level: float = 0.05):
    """
    Check if the time series is stationary using the Augmented Dickey-Fuller (ADF) test.

    Parameters:
    - series (pd.Series): The time series data to test.
    - significance_level (float): The significance level to reject the null hypothesis (default is 0.05).

    Returns:
    - bool: True if the series is stationary, False if not.
    """
    result = adfuller(series.dropna())
    p_value = result[1]

    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {p_value}")
    print(f"Critical Values: {result[4]}")

    if p_value < significance_level:
        print("The time series is stationary.")
        return True
    else:
        print("The time series is non-stationary.")
        return False


def find_missing_date_ranges(data: pd.Series, date_column: str):
    """
    Identify and return the missing date ranges in a time series dataset.

    Parameters:
    - data: pd.DataFrame. The dataset with a datetime index or datetime column.
    - date_column: str. The name of the column containing datetime values if the index is not datetime.

    Returns:
    - missing_ranges: List of tuples. Each tuple represents a missing date range (start_date, end_date).
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        if date_column not in data.columns:
            raise ValueError(f"Column {date_column} not found in dataset.")
        data = data.set_index(date_column)

    data = data.sort_index()
    all_dates = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")
    missing_dates = all_dates.difference(data.index)

    missing_ranges = []
    current_start = None

    for date in missing_dates:
        if current_start is None:
            current_start = date
        elif (date - prev_date).days > 1:
            missing_ranges.append((current_start, prev_date))
            current_start = date
        prev_date = date

    if current_start is not None:
        missing_ranges.append((current_start, prev_date))

    return missing_ranges[0]


def find_best_sarima_params(
    data: pd.Series,
    seasonal_period: int,
    p_values,
    d_values,
    q_values,
    P_values,
    D_values,
    Q_values,
) -> tuple:
    """
    Tunes the SARIMA model parameters to find the best combination based on AIC score.

    Parameters:
    - data: pd.Series. The time series data to model.
    - seasonal_period: int. The number of observations in one season.
    - p_values: list of int. The possible values for the non-seasonal autoregressive order.
    - d_values: list of int. The possible values for the non-seasonal differencing order.
    - q_values: list of int. The possible values for the non-seasonal moving average order.
    - P_values: list of int. The possible values for the seasonal autoregressive order.
    - D_values: list of int. The possible values for the seasonal differencing order.
    - Q_values: list of int. The possible values for the seasonal moving average order.

    Returns:
    - best_params: tuple. The best combination of parameters (p, d, q, P, D, Q) based on AIC score.
    - best_score: float. The best AIC score obtained.
    """
    best_score = float("inf")
    best_params = None

    for params in product(p_values, d_values, q_values):
        for seasonal_params in product(P_values, D_values, Q_values):
            try:
                model = SARIMAX(
                    data,
                    order=params,
                    seasonal_order=seasonal_params + (seasonal_period,),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
                # Calculate AIC (Akaike Information Criterion)
                aic = result.aic
                if aic < best_score:
                    best_score = aic
                    best_params = (params, seasonal_params)
            except Exception:
                continue

    return best_params, best_score


def find_best_sarima_cached(data, seasonal_period, p, d, q, P, D, Q):
    """
    Finds the best SARIMA parameters for the given data by searching over all combinations of p, d, q, P, D, and Q, and returns the best parameters and the associated AIC score. The search is done with a cache: if the same parameters have been searched before, the cache is used instead of recomputing the best parameters.

    Parameters:
    - data: pd.Series. The time series data to model.
    - seasonal_period: int. The number of observations in one season.
    - p_values: list of int. The possible values for the non-seasonal autoregressive order.
    - d_values: list of int. The possible values for the non-seasonal differencing order.
    - q_values: list of int. The possible values for the non-seasonal moving average order.
    - P_values: list of int. The possible values for the seasonal autoregressive order.
    - D_values: list of int. The possible values for the seasonal differencing order.
    - Q_values: list of int. The possible values for the seasonal moving average order.

    Returns:
    - best_params: tuple. The best combination of parameters (p, d, q, P, D, Q) based on AIC score.
    - best_score: float. The best AIC score obtained.
    """
    cache = Cache("sarima_cache")
    cache_key = f"sarima_params_{hash((seasonal_period, tuple(p), tuple(d), tuple(q), tuple(P), tuple(D), tuple(Q)))}"
    if cache_key in cache:
        return cache[cache_key]
    best_params, best_aic = find_best_sarima_params(
        data, seasonal_period, p, d, q, P, D, Q
    )
    cache[cache_key] = (best_params, best_aic)
    return best_params, best_aic


def present_base_dataset(
    original_data: pd.DataFrame, weather_data: pd.DataFrame, weather_features: List[str]
) -> None:
    """
    Present the base dataset.

    Parameters:
    - original_data: pd.DataFrame. The original dataset.
    - weather_data: pd.DataFrame. The weather dataset.
    - weather_features: List[str]. The weather features to display.

    Returns:
    - None
    """
    # Plot PV Yield and Inverter Yield over Time
    plt.figure(figsize=(12, 6))
    plt.plot(
        original_data.index,
        original_data[DatasetColumns.PV_YIELD.value],
        label="PV Yield",
        color="tab:blue",
    )
    plt.plot(
        original_data.index,
        original_data[DatasetColumns.INVERTER_YIELD.value],
        label="Inverter Yield",
        color="tab:orange",
    )
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.title("PV Yield and Inverter Yield over Time")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Battery Charge and Discharge over Time
    plt.figure(figsize=(12, 6))
    plt.plot(
        original_data.index,
        original_data[DatasetColumns.CHARGE.value],
        label="Charge (kWh)",
        color="tab:green",
    )
    plt.plot(
        original_data.index,
        original_data[DatasetColumns.DISCHARGE.value],
        label="Discharge (kWh)",
        color="tab:red",
    )
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.title("Battery Charge and Discharge over Time")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Weather Data
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # Temperature and Dew Point
    axes[0].plot(
        weather_data.index,
        weather_data["Temperature_C"],
        label="Temperature (°C)",
        color="red",
    )
    axes[0].plot(
        weather_data.index,
        weather_data["Dew_Point_C"],
        label="Dew Point (°C)",
        color="blue",
    )
    axes[0].set_ylabel("°C")
    axes[0].set_title("Temperature and Dew Point")
    axes[0].legend()

    # Humidity and Wind Speed
    axes[1].plot(
        weather_data.index,
        weather_data["Humidity_%"],
        label="Humidity (%)",
        color="green",
    )
    axes[1].plot(
        weather_data.index,
        weather_data["Speed_kmh"],
        label="Wind Speed (km/h)",
        color="purple",
    )
    axes[1].set_ylabel("Percentage / Speed (km/h)")
    axes[1].set_title("Humidity and Wind Speed")
    axes[1].legend()

    # Pressure and Precipitation Rate
    axes[2].plot(
        weather_data.index,
        weather_data["Pressure_hPa"],
        label="Pressure (hPa)",
        color="orange",
    )
    axes[2].plot(
        weather_data.index,
        weather_data["Precip_Rate_mm"],
        label="Precipitation Rate (mm)",
        color="cyan",
    )
    axes[2].set_ylabel("Pressure (hPa) / Precipitation (mm)")
    axes[2].set_title("Pressure and Precipitation Rate")
    axes[2].legend()

    # UV Index and Solar Radiation
    axes[3].plot(weather_data.index, weather_data["UV"], label="UV Index", color="pink")
    axes[3].plot(
        weather_data.index,
        weather_data["Solar_w/m2"],
        label="Solar Radiation (W/m²)",
        color="brown",
    )
    axes[3].set_ylabel("UV Index / Solar (W/m²)")
    axes[3].set_title("UV Index and Solar Radiation")
    axes[3].legend()

    plt.tight_layout()
    plt.xlabel("Datetime")
    plt.show()

    # Plot Missing Weather Data
    sns.heatmap(weather_data[weather_features].isna(), cbar=False)
    plt.title("Missing Weather Data Heatmap")
    plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_predictions(actual, predicted, fill_value=0):
    """
    Evaluate predictions using MAE, MSE, and RMSE.

    Parameters:
    - actual (pd.Series or np.array): The actual values.
    - predicted (pd.Series or np.array): The predicted values.
    - fill_value (float): Value to fill NaNs in the actual data. Default is 0.

    Returns:
    - metrics (dict): A dictionary containing MAE, MSE, and RMSE.
    """
    # Fill NaNs in actual and predicted
    actual = (
        actual.fillna(fill_value)
        if hasattr(actual, "fillna")
        else np.nan_to_num(actual, nan=fill_value)
    )
    predicted = (
        predicted.fillna(fill_value)
        if hasattr(predicted, "fillna")
        else np.nan_to_num(predicted, nan=fill_value)
    )

    # Calculate metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse}
