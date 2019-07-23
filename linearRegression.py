import numpy as np
import pandas as pd

class LinearRegression():

    weights = list()
    alpha = 0.01
   
    def get_data(self, data_frame, Y, X):
        explanators = pd.DataFrame()

        predictor = data_frame[Y]
        for index in range(0, len(X)):
            explanators[X[index]] = data_frame[X[index]]

        return predictor, explanator

    def optimise_weights(self, observations, predictor, explanators):
        diff = 10
        iterations = 10000
        while(iterations):
            for index in range(0, len(weights)):
                partial_derivative = 
            prev_coeff = coeff
            coeff = prev_coeff - alpha*partial_derivative(error_func)
            diff = prev_coeff - coeff
            iteration -= 1
    
    def train_linear_model(self, df, Y, X):
        predictor, explanator = self.get_data(df, Y, X) 
        weights = list(np.random.randn(len(kwargs)))

        self.optimize_weights(predictor, explanator)
    
    def get_predictions(self):
        
        
    def get_individual_predictions(self, kwargs):
        result = 0
        
        for index in range(0, len(kwargs)):
            result += weights[index]*kwargs[index]
        
        result += weights[-1]
        
        return result


LR = LinearRegression()
LR.train_linear_model()
