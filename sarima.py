from itertools import product
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
from tqdm.notebook import tqdm_notebook

data = pd.read_csv("Cleaned_data.csv")
subset = data[["Unnamed: 0", "ActivePower", "WindSpeed", "WindDirection"]]

# convert Datetime to pandas datetimeand set it as index
subset["Unnamed: 0"] = pd.to_datetime(subset["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S")
datetime_index = pd.DatetimeIndex(subset["Unnamed: 0"].values)
subset = subset.set_index(datetime_index)
# Drop redundant column
subset = subset.drop(['Unnamed: 0'], axis=1)

# Autoregression plots
windspeeds_days = subset['WindSpeed'].resample('D').mean()
windspeeds_weeks = subset['WindSpeed'].resample('W').mean()
windspeeds_days = windspeeds_days.dropna()
windspeeds_weeks = windspeeds_weeks.dropna()
windspeeds_hours = subset['WindSpeed']

# Autoregression plots
plot_acf(windspeeds_weeks )
plot_pacf(windspeeds_weeks)
plt.show()

# Dickley Fuller test
results_windspeeds_day = adfuller(windspeeds_days, autolag="AIC")
results_windspeeds_week = adfuller(windspeeds_weeks, autolag="AIC")
print(results_windspeeds_day)
print(results_windspeeds_week)
# So as expected, the windspeed is seasonal for the months, but not when taking into account
print("Windspeed is (day) stationary?", results_windspeeds_day[0] < results_windspeeds_day[4]['1%'])
print("Windspeed is (weeks) stationary?", results_windspeeds_week[0] < results_windspeeds_week[4]['1%'])

plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(windspeeds_days)
plt.title('Windspeed per Day')
plt.ylabel('Windspeeds (m/s)')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(windspeeds_weeks)
plt.title('Windspeed per Week')
plt.ylabel('Windspeeds (m/s)')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# There is a period of 12 months/1 year before a new windy season starts
# windspeeds_weeks = windspeeds_weeks.diff(4)
# print(windspeeds_weeks)
# data = windspeeds_weeks.drop(index=windspeeds_weeks.index[:4],
#         axis=0,
#         inplace=True)
# print(data.shape)

results_windspeeds_df = adfuller(windspeeds_weeks, autolag="AIC")
print(results_windspeeds_df)
print("Windspeed is (months) stationary?", results_windspeeds_df[0] < results_windspeeds_df[4]['5%'])


def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """

    results = []

    for param in tqdm_notebook(parameters_list):
        try:
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue

        aic = model.aic
        results.append([param, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df

# Note that we will only test different values for the parameters p, P, q and Q.
# We know that both seasonal and non-seasonal integration parameters should be 1,
# and that the length of the season is 24 hours.
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 1
Q = range(0, 4, 1)
s = 4
parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))

result_df = optimize_SARIMA(parameters_list, 1, 1, 365, windspeeds_weeks)
result_df
