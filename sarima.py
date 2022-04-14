from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("Cleaned_data.csv")
subset = data[["Unnamed: 0", "ActivePower", "WindSpeed", "WindDirection"]]

# convert Datetime to pandas datetimeand set it as index
subset["Unnamed: 0"] = pd.to_datetime(subset["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S")
datetime_index = pd.DatetimeIndex(subset["Unnamed: 0"].values)
subset = subset.set_index(datetime_index)
# Drop redundant column
subset = subset.drop(['Unnamed: 0'], axis=1)

# Autoregression plots
windspeeds_years = subset['WindSpeed'].resample('Y').mean().to_numpy()
windspeeds_months = subset['WindSpeed'].resample('M').mean().to_numpy()
windspeeds_days = subset['WindSpeed'].resample('D').mean().to_numpy()
windspeeds_hours = subset['WindSpeed'].to_numpy()
print(windspeeds_years.shape)
print(windspeeds_months.shape)
print(windspeeds_days.shape)
print(windspeeds_hours.shape)

order=(1,1,1)
sorder = (1,1,1,30)
model = SARIMAX(windspeeds_days, order=order, seasonal_order=sorder, enforce_stationarity=False)
