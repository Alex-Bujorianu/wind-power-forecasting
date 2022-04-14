import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# First get the data
data = pd.read_csv("Cleaned_data.csv")
x = np.array(data['WindSpeed'].tolist())
y = np.array(data['ActivePower'].tolist())
x_train = x[0:int(0.8*len(x))]
y_train = y[0:int(0.8*len(x))]
fit = np.polyfit(x=x_train, y=y_train, deg=3)
model = np.poly1d(fit)
myline = np.linspace(start=0, stop=20, num=100)

plt.scatter(x_train, y_train)
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power (kW)")
plt.plot(myline, model(myline), color='red', label='Dumb polynomial regression')


def polynomial_prediction(x: np.ndarray) -> y:
    """
    @:param x: The x values (wind speed) based on which the prediction should be made.
    :return: the predicted output as a numpy array
    """
    return model(x)

def polynomial_prediction_smarter(x: np.ndarray) -> y:
    """
    A smarter version of polynomial_prediction that takes into account threshold values
    :param x: The input (wind speed) to use for prediction
    :return: the predicted output as a numpy array
    """
    to_return = []
    for wind_speed in x:
        if wind_speed >= 10:
            to_return.append(1750)
        elif wind_speed <= 1:
            to_return.append(0)
        else:
            to_return.append(model(np.array([wind_speed]))[0])
    return np.array(to_return)

plt.plot(myline, polynomial_prediction_smarter(myline), color='yellow', label='Smart polynomial regression')
plt.legend()
plt.show()
