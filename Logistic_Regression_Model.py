import numpy as np

class Logistic_Regression:
    
    # declaring learning rate and number of iterations(hyperparameters)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations


    # fit function to train the model to some dataset
    def fit(self, X, Y):
        self.m, self.n = X.shape    #[m=rows, n=cols]

        self.w = np.zeros(self.n)   # for the total number of features creating empty weights
        self.b = 0                  # bias - initially 0
        self.X = X                  # initiating features
        self.Y = Y                  # intiating target

        # implementing gradient descent for optimization of solution
        for i in range(self.no_of_iterations):
            self.update_weights()