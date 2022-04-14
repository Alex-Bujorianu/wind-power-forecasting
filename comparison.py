from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polynomial_regression import polynomial_prediction, polynomial_prediction_smarter

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
rmse_dumb_polynomial = sqrt(mse_dumb_polynomial)
rmse_smart_polynomial = sqrt(mse_smart_polynomial)
print("MSE scores: \n", "Dumb polynomial: ", mse_dumb_polynomial, "Smart polynomial: ", mse_smart_polynomial)
print("RMSE of dumb polynomial model: ", rmse_dumb_polynomial)
print("RMSE of smart polynomial model: ", rmse_smart_polynomial)