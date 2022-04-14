import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from numpy import array

data = pd.read_csv("Cleaned_data.csv")
subset = data[["Unnamed: 0", "ActivePower", "WindSpeed", "WindDirection"]]
print(subset.head())
plt.scatter(subset['WindSpeed'], subset['ActivePower'])
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power (kW)")
plt.show()
# Effect of wind direction on power, given a wind speed
# 5, 10 and 15m/s
windspeeds = subset['WindSpeed'].tolist()
wind_directions = subset["WindDirection"].tolist()
powers = subset['ActivePower'].tolist()
results = []
for i in range(len(windspeeds)):
    if (windspeeds[i] > 4.5) and (windspeeds[i] < 5.5):
        results.append({"Wind speed": 5,
                        "Wind direction": wind_directions[i],
                        "Power": powers[i]})
    elif (windspeeds[i] > 9.5) and (windspeeds[i] < 10.5):
        results.append({"Wind speed": 10,
                        "Wind direction": wind_directions[i],
                        "Power": powers[i]})
    else:
        continue

plt.plot([x["Wind direction"] for x in results if x["Wind speed"]==5],
            [x["Power"] for x in results if x["Wind speed"]==5], 'o', label="5m/s")
plt.plot([x["Wind direction"] for x in results if x["Wind speed"]==10],
            [x["Power"] for x in results if x["Wind speed"]==10], 'o', label="10m/s")
plt.xlabel("Wind Direction (degrees)")
plt.ylabel("Power (kW)")
plt.legend()
plt.show()


#convert Datetime to pandas datetimeand set it as index
subset["Unnamed: 0"] = pd.to_datetime(subset["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S%z")
datetime_index = pd.DatetimeIndex(subset["Unnamed: 0"].values)
subset = subset.set_index(datetime_index)
#Drop redundant column
subset = subset.drop(['Unnamed: 0'], axis=1)
print(subset.shape)
#Resampling to hoursand taking the average of each hour

# Autoregression plots
windspeeds_months = subset['WindSpeed'].resample('M').mean()
print(windspeeds_months.shape)
windspeeds_months = windspeeds_months.to_numpy()
plot_acf(windspeeds_months, lags=12)
plot_pacf(windspeeds_months, lags=12)
plt.show()

# Dickley Fuller test
windspeeds_days = subset['WindSpeed']
results_windspeeds_days = adfuller(windspeeds_days, autolag="AIC")
results_windspeeds_df = adfuller(windspeeds_months, autolag="AIC")
print(results_windspeeds_df)
print(results_windspeeds_days)
# So as expected, the windspeed is seasonal for the months, but not when taking into account
print("Windspeed is (months) stationary?", results_windspeeds_df[0] < results_windspeeds_df[4]['1%'])
print("Windspeed is (days) stationary?", results_windspeeds_days[0] < results_windspeeds_days[4]['1%'])