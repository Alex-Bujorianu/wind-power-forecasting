from itertools import product
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook

data = pd.read_csv("Cleaned_data.csv")
subset = data[["Unnamed: 0", "ActivePower", "WindSpeed", "WindDirection"]]

# convert Datetime to pandas datetimeand set it as index
subset["Unnamed: 0"] = pd.to_datetime(subset["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S")
datetime_index = pd.DatetimeIndex(subset["Unnamed: 0"].values)
subset = subset.set_index(datetime_index)
# Drop redundant column
subset = subset.drop(['Unnamed: 0'], axis=1)

print(subset.isna().sum())

powers_hours = subset['ActivePower'].to_numpy()
train = powers_hours[0:len(powers_hours)-24]

model = AutoReg(train, lags=40)
model_fit = model.fit()
