from enum import Enum
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class DatasetColumns(Enum):
    STATISTICAL_PERIOD = "Statistical Period"
    GLOBAL_IRRADIATION = "Global Irradiation (kWh/㎡)"
    AVERAGE_TEMPERATURE = "Average Temperature(°C)"
    THEORETICAL_YIELD = "Theoretical Yield (kWh)"
    PV_YIELD = "PV Yield (kWh)"
    INVERTER_YIELD = "Inverter Yield (kWh)"
    EXPORT = "Export (kWh)"
    IMPORT = "Import (kWh)"
    LOSS_EXPORT_LIMITATION_KWH = "Loss Due to Export Limitation (kWh)"
    LOSS_EXPORT_LIMITATION_RP = "Loss Due to Export Limitation(Rp)"
    CHARGE = "Charge (kWh)"
    DISCHARGE = "Discharge (kWh)"
    REVENUE = "Revenue (Rp)"


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

    # Ensure the index is sorted
    data = data.sort_index()

    # Generate the complete date range based on the min and max date
    all_dates = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")

    # Find missing dates
    missing_dates = all_dates.difference(data.index)

    # Group missing dates into continuous ranges
    missing_ranges = []
    current_start = None

    for date in missing_dates:
        if current_start is None:
            current_start = date
        elif (date - prev_date).days > 1:
            missing_ranges.append((current_start, prev_date))
            current_start = date
        prev_date = date

    # Add the last range if there is an open interval
    if current_start is not None:
        missing_ranges.append((current_start, prev_date))

    return missing_ranges[0]
