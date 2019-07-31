import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as LinearReg
from linearRegression import LinearRegression

np.random.seed(123)
data_frame = pd.DataFrame(np.random.randint(0, 10, size=(1000, 2)), columns=["X", "Y"])

LR = LinearRegression()

Xtrain, Ytrain, Xtest, Ytest = LR.get_data(data_frame, "Y", ["X"], test_split = 0.2)
LR.train_linear_model(Ytrain, Xtrain)

print("\n")
print("************************************************")
print("Results from current model:")
print("************************************************")
print("RMSE: ", LR.rmse(Ytest, Xtest))
print("coefficients: ", LR.weights[:-1])
print("intercept: ", LR.weights[-1])
model = LinearReg()

model.fit(Xtrain, Ytrain)

print("\n")
print("************************************************")
print("Results from scikitlearn LinearRegression model:")
print("************************************************")
Ypred = model.predict(Xtest)
print("RMSE: ", sqrt(mean_squared_error(Ytest, Ypred)))
print("coefficients: ", model.coef_)
print("intercept: ", model.intercept_)
