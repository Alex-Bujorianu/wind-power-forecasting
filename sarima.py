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
from pmdarima.arima import auto_arima

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook

data = pd.read_csv("Cleaned_data.csv")
subset = data[["Unnamed: 0", "ActivePower", "WindSpeed", "WindDirection"]]

# convert Datetime to pandas datetimeand set it as index
subset["Unnamed: 0"] = pd.to_datetime(subset["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S")
datetime_index = pd.DatetimeIndex(subset["Unnamed: 0"].values)
subset = subset.set_index(datetime_index)
# Drop redundant column
#subset = subset.drop(['Unnamed: 0'], axis=1)

# Autoregression plots
windspeeds_days = subset['WindSpeed'].resample('D').mean()
windspeeds_weeks = subset['WindSpeed'].resample('W').mean()
windspeeds_months = subset['WindSpeed'].resample('M').mean()

windspeeds_days = windspeeds_days.dropna()
windspeeds_weeks = windspeeds_weeks.dropna()
windspeeds_months.dropna()
windspeeds_hours = subset['WindSpeed']

# Autoregression plots
# plot_acf(windspeeds_hours)
# plot_pacf(windspeeds_hours )
# plt.show()

# Dickley Fuller test
# results_windspeeds_hours = adfuller(windspeeds_hours, autolag="AIC")
# results_windspeeds_day = adfuller(windspeeds_days, autolag="AIC")
# results_windspeeds_week = adfuller(windspeeds_weeks, autolag="AIC")
# results_windspeeds_months = adfuller(windspeeds_months, autolag="AIC")
# print(results_windspeeds_hours)
# print(results_windspeeds_day)
# print(results_windspeeds_week)
# print(results_windspeeds_months)
# So as expected, the windspeed is seasonal for the months, but not when taking into account
# print("Windspeed is (hours) stationary?", results_windspeeds_hours[0] < results_windspeeds_hours[4]['1%'])
# print("Windspeed is (day) stationary?", results_windspeeds_day[0] < results_windspeeds_day[4]['1%'])
# print("Windspeed is (weeks) stationary?", results_windspeeds_week[0] < results_windspeeds_week[4]['1%'])
# print("Windspeed is (months) stationary?", results_windspeeds_months[0] < results_windspeeds_months[4]['1%'])


plot_acf(windspeeds_months, lags=12)
plot_pacf(windspeeds_months, lags=12)
plt.show()
data = windspeeds_weeks.drop(index=windspeeds_weeks.index[:4],
        axis=0,
        inplace=True)
#print(data.shape)

results_windspeeds_df = adfuller(windspeeds_weeks, autolag="AIC")
print(results_windspeeds_df)
print("Windspeed is (months) stationary?", results_windspeeds_df[0] < results_windspeeds_df[4]['1%'])

#m = 12 as yearly periods
#result = auto_arima(windspeeds_months, seasonal=True, m=12, start_p=0, start_q=0, max_P=5, max_Q=5, max_D=5).summary()
#print(result)

train = windspeeds_months[0:len(windspeeds_months)-12]
test = windspeeds_months[len(windspeeds_months)-12:]
print(train.shape, test.shape, windspeeds_months.shape)

arima_model = SARIMAX(train, order=(2, 0, 0), seasonal_order=(1, 0, 0, 12))
arima_result = arima_model.fit()
arima_result.summary()
pred = arima_result.predict(start=len(train), end=len(windspeeds_months)-1, type="levels").rename("SARIMA")
windspeeds_months.plot(legend=True)
pred.plot(legend=True)
plt.show()
# def optimize_SARIMA(parameters_list, d, D, s, exog):
#     """
#         Return dataframe with parameters, corresponding AIC and SSE
#
#         parameters_list - list with (p, q, P, Q) tuples
#         d - integration order
#         D - seasonal integration order
#         s - length of season
#         exog - the exogenous variable
#     """
#
#     results = []
#     print(len(parameters_list))
#     exception_count = 0
#     for param in parameters_list:
#         print("Testing with following paramaters: " + str(param))
#         try:
#             model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s), enforce_stationarity=True, enforce_invertibility=True)
#             model_fit = model.fit()
#         except:
#             print("An exception occured with the SARIMAX model")
#             exception_count += 1
#             continue
#
#         aic = model_fit.aic
#         results.append([param, aic])
#
#     print("Number of exceptions: ", exception_count)
#     result_df = pd.DataFrame(results)
#     print(result_df)
#     result_df.columns = ['(p,q)x(P,Q)', 'AIC']
#     # Sort in ascending order, lower AIC is better
#     result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
#
#     return result_df
#
# # Note that we will only test different values for the parameters p, P, q and Q.
# # We know that both seasonal and non-seasonal integration parameters should be 1,
# # and that the length of the season is 24 hours.
# p = range(0, 1, 1)
# d = 1
# q = range(0, 1, 1)
# P = range(0, 1, 1)
# D = 1
# Q = range(0, 1, 1)
# s = 4
# parameters = product(p, q, P, Q)
# parameters_list = list(parameters)
# print(len(parameters_list))
#
# result_df = optimize_SARIMA(parameters_list, 1, 1, 12, windspeeds_months)
# result_df

