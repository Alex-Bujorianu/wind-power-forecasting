import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Cleaned_data.csv")
subset = data[["ActivePower", "WindSpeed", "WindDirection"]]
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
    print(windspeeds[i])
    if (windspeeds[i] > 4.5) and (windspeeds[i] < 5.5):
        results.append({"Wind speed": 5,
                        "Wind direction": wind_directions[i],
                        "Power": powers[i]})

plt.plot([x["Wind direction"] for x in results if x["Wind speed"]==5],
            [x["Power"] for x in results if x["Wind speed"]==5], label="5m/s")
plt.xlabel("Wind Direction (degrees)")
plt.ylabel("Power (kW)")
plt.legend()
plt.show()
