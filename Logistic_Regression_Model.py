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
    
    def update_weights(self):
        # y_cap formula(sigmoid formula)
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))  #z = wx+b

        # Now build derivatives
        dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m)*np.sum(Y_hat - self.Y)

        # updating weights and bias using gradient descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db