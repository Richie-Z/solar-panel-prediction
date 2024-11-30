from enum import Enum


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


class WeatherDatasetColumns(Enum):
    DATETIME = "datetime"
    TEMPERATURE_C = "Temperature_C"
    DEW_POINT_C = "Dew_Point_C"
    HUMIDITY_PERCENT = "Humidity_%"
    WIND = "Wind"
    SPEED_KMH = "Speed_kmh"
    GUST_KMH = "Gust_kmh"
    PRESSURE_HPA = "Pressure_hPa"
    PRECIP_RATE_MM = "Precip_Rate_mm"
    PRECIP_ACCUM_MM = "Precip_Accum_mm"
    UV = "UV"
    SOLAR_W_M2 = "Solar_w/m2"
