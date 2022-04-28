from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Cleaned_data.csv")
x = np.array(data['WindSpeed'].tolist())
y = np.array(data['ActivePower'].tolist())
x_train, x_test = train_test_split(x, test_size=0.20, random_state=101)
y_train, y_test = train_test_split(y, test_size=0.20, random_state=101)
x_test = x_test.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)
print(x_train, y_train)
neural_network = MLPRegressor(solver="adam", learning_rate_init=0.01).fit(x_train, np.ravel(y_train))
print(neural_network.score(x_test, y_test))

C1 = data['WindSpeed'].values
C2 = data['AmbientTemperatue'].values
C = np.column_stack((C1,C2))
x_bivariate_train, x_bivariate_test = train_test_split(C, test_size=0.20, random_state=101)
# The bivariate NN takes longer to converge with more weights.
# Increase learning rate and iterations.
bivariate_neural_network = MLPRegressor(solver="adam", max_iter=500,
                                        learning_rate_init=0.016).\
    fit(x_bivariate_train, np.ravel(y_train))
print("Bivariate NN score: ", bivariate_neural_network.score(x_bivariate_test, y_test))

plt.scatter(x_test, y_test, label="Actual")
plt.scatter(x_test, neural_network.predict(x_test), label="Predicted (univariate)")
plt.scatter(x_test, bivariate_neural_network.predict(x_bivariate_test),
            label="Predicted (bivariate)")
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power (kW)")
plt.legend()
plt.show()