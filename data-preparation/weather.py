import pandas as pd

FILE_NAME = "./model/weathers.csv"
data = pd.read_csv(FILE_NAME)
data["datetime"] = pd.to_datetime(
    data["Date"] + " " + data["Time"], format="%Y/%m/%d %I:%M %p"
)
data.set_index("datetime", inplace=True)
data.drop(["Date", "Time"], axis=1, inplace=True)

# Convert columns to numeric, coercing errors to NaN
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

hourly_data = data.resample("h").mean()
hourly_data = hourly_data.round(2)
hourly_data.reset_index(inplace=True)
hourly_data.to_csv("dataset_weather.csv", index=False)

print("Hourly data saved to dataset_weather.csv")
