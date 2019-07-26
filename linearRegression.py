import numpy as np
import pandas as pd
import matplotlib as plt

class LinearRegression():

    weights = list()
    alpha = 0.000001
   
    def get_data(self, data_frame, Y, X):
        features = pd.DataFrame()

        predictor = data_frame[Y]
        for index in range(0, len(X)):
            features[X[index]] = data_frame[X[index]]

        return predictor, features

    def optimize_weights(self, predictor, features):
        diff = 10
        iterations = 10000
        while(iterations):
            total_sum = np.zeros(len(features.index))
            feature_sum = 0
            for index in range(0, len(self.weights)-1):
                total_sum += features[features.columns[index]] * self.weights[index]
            total_sum += np.ones(len(features.index)) * self.weights[-1]
            error_diff = predictor - total_sum
            
            for index in range(0, len(self.weights)-1):
                partial_diff = error_diff.dot(features[features.columns[index]])
                partial_diff_sum = partial_diff.sum()
                partial_diff = partial_diff_sum/float(len(features.index))
                self.weights[index] = self.weights[index] - self.alpha*partial_diff_sum

            self.weights[-1] = self.weights[-1] - self.alpha*(error_diff.sum()/float(len(features.index)))
            print(self.weights)
            iterations -= 1
    
    def train_linear_model(self, df, Y, X):
        predictor, features = self.get_data(df, Y, X) 
        self.weights = list(np.random.randn(len(X)+1))

        self.optimize_weights(predictor, features)
        #self.optimize_weights(predictor, explanator)
    
    def get_predictions(self):
        print("getting individual data points")
        
    def get_individual_predictions(self, kwargs):
        result = 0
        
        for index in range(0, len(kwargs)):
            result += weights[index]*kwargs[index]
        
        result += weights[-1]
        
        return result

data_frame = pd.DataFrame(np.random.randint(0, 100, size=(100, 2)), columns=["X", "Y"])

LR = LinearRegression()
LR.train_linear_model(data_frame, "Y", ["X"])
