from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from helpers import find_missing_date_ranges
from enums import DatasetColumns, WeatherDatasetColumns
from sklearn.preprocessing import MinMaxScaler


def generate_predictions_from_loaded(loaded_generator, weather_features, scaler):
    weather_features = tf.cast(weather_features, tf.float32)
    batch_size = tf.shape(weather_features)[0]

    noise = tf.random.normal([batch_size, 100], dtype=tf.float32)
    predictions_scaled = loaded_generator([noise, weather_features], training=False)

    predictions_with_weather = np.concatenate(
        [predictions_scaled.numpy(), weather_features.numpy()], axis=1
    )
    predictions = scaler.inverse_transform(predictions_with_weather)[:, 0]

    return predictions


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
post_gap_data = original_data[original_data.index >= gap_end].asfreq("h")

pre_gap_train_size = int(len(pre_gap_data) * 0.8)
pre_gap_train = pre_gap_data.iloc[:pre_gap_train_size].copy()
pre_gap_test = pre_gap_data.iloc[pre_gap_train_size:]

pre_gap_train.loc[:, DatasetColumns.PV_YIELD.value] = pre_gap_train[
    DatasetColumns.PV_YIELD.value
].interpolate(method="linear")


pre_weather_data = weather_data[weather_data.index < gap_start].bfill()
pre_weather_data = pre_weather_data.reindex(pre_gap_data.index)
pre_weather_data_test = pre_weather_data.reindex(pre_gap_test.index)


gap_weather_data = weather_data.reindex(gap_dates).ffill()
post_weather_data = weather_data[weather_data.index >= gap_end].bfill()

pre_gap_train_combined = pre_gap_train.join(
    pre_weather_data[weather_features], how="inner"
)
pre_gap_test_combined = pre_gap_test.join(
    pre_weather_data_test[weather_features], how="inner"
)


sarimax_model = SARIMAXResults.load("sarimax_model_results.pkl")
sarimax_cv_model = SARIMAXResults.load("sarimax_cv_model_results.pkl")
loaded_generator = tf.keras.models.load_model("solar_gan_generator.h5")
with open("scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)


exog_gap = gap_weather_data[weather_features]

sarimax_prediction = sarimax_model.forecast(steps=len(gap_dates), exog=exog_gap)
sarimax_prediction = sarimax_prediction.clip(lower=0)

sarimax_cv_prediction = sarimax_cv_model.forecast(steps=len(gap_dates), exog=exog_gap)
sarimax_cv_prediction = sarimax_cv_prediction.clip(lower=0)


scaler_weather = MinMaxScaler()
scaler_weather.min_, scaler_weather.scale_ = (
    loaded_scaler.min_[1:],
    loaded_scaler.scale_[1:],
)

gap_weather_scaled = scaler_weather.transform(gap_weather_data[weather_features])

gap_weather_tensor = tf.convert_to_tensor(gap_weather_scaled, dtype=tf.float32)
predictions = generate_predictions_from_loaded(
    loaded_generator, gap_weather_tensor, loaded_scaler
)


plt.figure(figsize=(12, 6))

plt.plot(gap_dates, sarimax_prediction, label="SARIMAX", color="green")
plt.plot(gap_dates, sarimax_cv_prediction, label="SARIMAX with CV", color="blue")

plt.plot(gap_dates, predictions, label="GAN Prediction", color="red")

plt.legend()
plt.title("SARIMAX vs GAN Predictions on gap dates range")
plt.ylabel("PV Yield")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
