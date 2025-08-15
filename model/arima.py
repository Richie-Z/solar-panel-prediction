import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers import (
    find_best_sarima_cached,
    find_missing_date_ranges,
    present_base_dataset,
    evaluate_predictions,
    compare_prediction,
)
from enums import DatasetColumns, WeatherDatasetColumns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA


FILE_NAME = "dataset.csv"
WEATHER_DATASET = "dataset_weather.csv"


original_data = pd.read_csv(
    FILE_NAME,
    parse_dates=[DatasetColumns.STATISTICAL_PERIOD.value],
    index_col=DatasetColumns.STATISTICAL_PERIOD.value,
)

weather_data = pd.read_csv(
    WEATHER_DATASET,
    parse_dates=[WeatherDatasetColumns.DATETIME.value],
    index_col=WeatherDatasetColumns.DATETIME.value,
).asfreq("h")

weather_features = [
    WeatherDatasetColumns.TEMPERATURE_C.value,
    WeatherDatasetColumns.HUMIDITY_PERCENT.value,
]


gap_start, gap_end = find_missing_date_ranges(
    original_data, DatasetColumns.STATISTICAL_PERIOD.value
)
gap_dates = pd.date_range(start=gap_start, end=gap_end, freq="h")


pre_gap_data = original_data[original_data.index < gap_start].asfreq("h")
pre_gap_train_size = int(len(pre_gap_data) * 0.8)
pre_gap_train = pre_gap_data.iloc[:pre_gap_train_size].copy()
pre_gap_test = pre_gap_data.iloc[pre_gap_train_size:]


post_gap_data = original_data[original_data.index >= gap_end].asfreq("h")

pre_weather_data = weather_data[weather_data.index < gap_start].bfill()
gap_weather_data = weather_data.reindex(gap_dates).ffill()
post_weather_data = weather_data[weather_data.index >= gap_end].bfill()

pre_gap_train.loc[:, DatasetColumns.PV_YIELD.value] = pre_gap_train[
    DatasetColumns.PV_YIELD.value
].interpolate(method="linear")

pre_weather_data = pre_weather_data.drop(
    pre_weather_data.index.difference(pre_gap_data.index)
)

p = d = q = range(0, 3)
P = D = Q = range(0, 2)
seasonal_period = 24

best_params, best_aic = find_best_sarima_cached(
    pre_gap_train[DatasetColumns.PV_YIELD.value],
    seasonal_period,
    p,
    d,
    q,
    P,
    D,
    Q,
)

(order_params, seasonal_params) = best_params
print(f"Best Parameters: {best_params}, AIC: {best_aic}")


def train_arima_model():
    arima_model = ARIMA(
        pre_gap_train[DatasetColumns.PV_YIELD.value],
        order=order_params,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    arima_result = arima_model.fit()
    preds = arima_result.predict(
        start=pre_gap_test.index[0], end=pre_gap_test.index[-1]
    )

    evaluate_predictions(pre_gap_test[DatasetColumns.PV_YIELD.value], preds)

    compare_prediction(
        "ARIMA Predictions vs Actual",
        pre_gap_train[DatasetColumns.PV_YIELD.value],
        pre_gap_test[DatasetColumns.PV_YIELD.value],
        preds,
    )


train_arima_model()


def train_sarima_model():
    sarimax_model = SARIMAX(
        pre_gap_train[DatasetColumns.PV_YIELD.value],
        order=order_params,
        seasonal_order=seasonal_params + (seasonal_period,),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    sarimax_result = sarimax_model.fit()
    preds = sarimax_result.predict(
        start=pre_gap_test.index[0], end=pre_gap_test.index[-1]
    )

    evaluate_predictions(pre_gap_test[DatasetColumns.PV_YIELD.value], preds)

    compare_prediction(
        "SARIMA without Exogenous Predictions vs Actual",
        pre_gap_train[DatasetColumns.PV_YIELD.value],
        pre_gap_test[DatasetColumns.PV_YIELD.value],
        preds,
    )


train_sarima_model()


def train_sarima_with_exogenous():
    merged_train = pre_gap_train.merge(
        pre_weather_data[weather_features],
        left_index=True,
        right_index=True,
        how="inner",
    )
    merged_test = pre_gap_test.merge(
        pre_weather_data[weather_features],
        left_index=True,
        right_index=True,
        how="inner",
    )

    exog_train = merged_train[weather_features].ffill()
    exog_test = merged_test[weather_features].ffill()

    sarimax_model_exog = SARIMAX(
        merged_train[DatasetColumns.PV_YIELD.value],
        exog=exog_train,
        order=order_params,
        seasonal_order=seasonal_params + (seasonal_period,),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    sarimax_result_exog = sarimax_model_exog.fit()

    preds_exog = sarimax_result_exog.predict(
        start=merged_test.index[0],
        end=merged_test.index[-1],
        exog=exog_test,
    )

    evaluate_predictions(merged_test[DatasetColumns.PV_YIELD.value], preds_exog)

    compare_prediction(
        "SARIMAX Predictions vs Actual",
        test_data=merged_test[DatasetColumns.PV_YIELD.value],
        predicted=preds_exog,
    )
    return sarimax_result_exog


trained_model = train_sarima_with_exogenous()
trained_model.save("sarimax_model_results.pkl")


def train_sarima_with_exogenous_cv():
    merged_data = pre_gap_train.merge(
        pre_weather_data[weather_features],
        left_index=True,
        right_index=True,
        how="inner",
    )

    exog_data = merged_data[weather_features]
    ts_cv = TimeSeriesSplit(n_splits=5)

    model = None

    for i, (train_idx, test_idx) in enumerate(ts_cv.split(merged_data)):
        train_data, test_data = merged_data.iloc[train_idx], merged_data.iloc[test_idx]
        exog_train, exog_test = exog_data.iloc[train_idx], exog_data.iloc[test_idx]

        sarimax_model_exog = SARIMAX(
            train_data[DatasetColumns.PV_YIELD.value],
            exog=exog_train,
            order=order_params,
            seasonal_order=seasonal_params + (seasonal_period,),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        sarimax_result_exog = sarimax_model_exog.fit()

        preds_exog = sarimax_result_exog.predict(
            start=test_data.index[0],
            end=test_data.index[-1],
            exog=exog_test,
        )

        if i == ts_cv.n_splits - 1:
            evaluate_predictions(test_data[DatasetColumns.PV_YIELD.value], preds_exog)
            compare_prediction(
                f"SARIMAX with Time Series Predictions vs Actual",
                test_data=test_data[DatasetColumns.PV_YIELD.value],
                predicted=preds_exog,
            )

            model = sarimax_result_exog

    return model


trained_model_cv = train_sarima_with_exogenous_cv()
trained_model_cv.save("sarimax_cv_model_results.pkl")


exog_gap = gap_weather_data[weather_features]
gap_predictions = trained_model_cv.forecast(steps=len(gap_dates), exog=exog_gap)
gap_predictions = gap_predictions.clip(lower=0)
gap_predictions_df = gap_predictions.to_frame(name=DatasetColumns.PV_YIELD.value)
gap_predictions_df.index = gap_dates

combined_data = pd.concat([pre_gap_data, gap_predictions_df, post_gap_data]).asfreq("h")
combined_data.to_csv("combined_data.csv", index=True)
combined_data_train_size = int(len(combined_data) * 0.8)
combined_data_train = combined_data.iloc[:combined_data_train_size].copy()
combined_data_test = combined_data.iloc[combined_data_train_size:]
