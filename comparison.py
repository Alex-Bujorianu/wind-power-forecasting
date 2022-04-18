from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polynomial_regression import polynomial_prediction, polynomial_prediction_smarter
from neural_network import neural_network

def max(x: list) -> float:
    max = 0
    for num in x:
        if num > max:
            max = num
    return max

def min(x: list) -> float:
    min = 100000
    for num in x:
        if num < min:
            min = num
    return min

data = pd.read_csv("Cleaned_data.csv")
x = np.array(data['WindSpeed'].tolist())
y = np.array(data['ActivePower'].tolist())
print(len(x), len(y))
X_train, x_test = train_test_split(x, test_size=0.20, random_state=101)
Y_train, y_test = train_test_split(y, test_size=0.20, random_state=101)
print(x_test, y_test)
print("Max of x_test is: ", max(x_test), "Min is: ", min(x_test))

print("Lengths of x_test and y_test: ", len(x_test), len(y_test))

mse_dumb_polynomial = mean_squared_error(y_test, polynomial_prediction(x_test))
mse_smart_polynomial = mean_squared_error(y_test, polynomial_prediction_smarter(x_test))
mse_neural_network = mean_squared_error(y_test, neural_network.predict(x_test.reshape(-1, 1)))
rmse_dumb_polynomial = sqrt(mse_dumb_polynomial)
rmse_smart_polynomial = sqrt(mse_smart_polynomial)
rmse_neural_network = sqrt(mse_neural_network)
print("MSE scores: \n", "Dumb polynomial: ", mse_dumb_polynomial, "Smart polynomial: ", mse_smart_polynomial)
print("RMSE of dumb polynomial model: ", rmse_dumb_polynomial)
print("RMSE of smart polynomial model: ", rmse_smart_polynomial)
print("RMSE of neural network: ", rmse_neural_network)

#Last 24 hours
y_24 = y[len(y)-24:-1]
x_24 = x[len(y)-24:-1]
rmse_24_dumb_polynomial = sqrt(mean_squared_error(y_24, polynomial_prediction(x_24)))
rmse_24_smart_polynomial = sqrt(mean_squared_error(y_24, polynomial_prediction_smarter(x_24)))
rmse_24_nn = sqrt(mean_squared_error(y_24, neural_network.predict(x_24.reshape(-1, 1))))
plt.plot(x_24, y_24, label="actual")
plt.plot(x_24, neural_network.predict(x_24.reshape(-1, 1)), label="neural network")
plt.plot(x_24, polynomial_prediction_smarter(x_24), label="smart polynomial")
plt.legend()
plt.show()
print("RMSEs: \n", "Dumb polynomial: ", rmse_24_dumb_polynomial, "\n",
      "Smart polynomial: ", rmse_24_smart_polynomial, "\n",
      "NN: ", rmse_24_nn)