from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
from polynomial_regression import polynomial_prediction, polynomial_prediction_smarter

data = pd.read_csv("Cleaned_data.csv")
x = np.array(data['WindSpeed'].tolist())
y = np.array(data['ActivePower'].tolist())
x_test = x[int(0.8*len(x)):]
y_test = y[int(0.8*len(y)):]
print(x_test, y_test)
mse_dumb_polynomial = mean_squared_error(y_test, polynomial_prediction(x_test))
mse_smart_polynomial = mean_squared_error(y_test, polynomial_prediction_smarter(x_test))
rmse_dumb_polynomial = sqrt(mse_dumb_polynomial)
rmse_smart_polynomial = sqrt(mse_smart_polynomial)
print("MSE scores: \n", "Dumb polynomial: ", mse_dumb_polynomial, "Smart polynomial: ", mse_smart_polynomial)
print("RMSE of dumb polynomial model: ", rmse_dumb_polynomial)
print("RMSE of smart polynomial model: ", rmse_smart_polynomial)