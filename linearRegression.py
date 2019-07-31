import numpy as np
import pandas as pd
from math import sqrt
import matplotlib as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LinearReg

class LinearRegression():

    weights = list()
    learning_rate = 0.01
   
    def get_data(self, data_frame, Y, X, test_split):
        features = pd.DataFrame()

        predictor = data_frame[Y]
        for index in range(0, len(X)):
            features[X[index]] = data_frame[X[index]]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,predictor,test_size=test_split,random_state=0)
        
        return Xtrain, Ytrain, Xtest, Ytest

    def optimize_weights(self, predictor, features):
        iterations = 10000
        features["ones"] = np.ones(features.shape[0], dtype=int)
        while(iterations):
            predicted_target = features.dot(self.weights)
            actual_target = predictor
           
            # Calculating X.T * (Hypothesis - Actual) 
            partial_diff = features.T.dot(predicted_target - actual_target)/features.shape[0]

            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - self.learning_rate * partial_diff[i]
            #print(self.weights)
            iterations -=1
   
    ## Take learning rate also as an argument 
    def train_linear_model(self, predictor, features, learning_rate=0.01):
        print("Training the linear model...")
        self.weights = list(np.random.randn(features.shape[1] + 1))
        self.learning_rate = learning_rate

        self.optimize_weights(predictor, features)
    
    def get_predictions(self, features):
        features["ones"] = np.ones(features.shape[0], dtype=int)
        predictions = features.dot(self.weights)
 
        return predictions
        
    def rmse(self, predictor, features):
        predicted_values = self.get_predictions(features)
        actual_values = predictor

        rmse = (predicted_values-actual_values)**2
        rmse = sqrt(rmse.sum()/features.shape[0])
        return rmse
