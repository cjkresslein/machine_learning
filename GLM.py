import numpy as np
from scipy.special import expit

class GLM:

    def __init__(self):
        self.lr = None
        self.w = None
        self.grad_norm = None
        self.train_mean = None
        self.train_std = None

    def link_function(self):
        pass

    def normalize(self, X_train):
        if self.train_mean is None:
            self.train_mean = X_train.mean(axis=0)  # mean of each column
            std = X_train.std(axis=0)               # std dev of each column
            std[std == 0] = 1
            self.train_std = std

        return (X_train - self.train_mean) / self.train_std
        
    def gradient_ascent(self, X_train, Y_train, lr = 0.01,  grad_norm=0.1, normalize=False):
        self.lr = lr
        self.grad_norm = grad_norm

        Y_train = Y_train.flatten()

        if(normalize):
            X_train = self.normalize(X_train)

        # number of input parameters
        x_size = np.size(X_train, axis=1)

        # initial parameters
        self.w = np.zeros(x_size)

        i = 0
        while(i<500):

            # find the difference between prediction and y_train
            prediction = self.link_function(np.dot(X_train, self.w))
            
            # find the gradient of likelihood (wrt each component in w)
            gradient =  X_train.T @ (Y_train-prediction)

            # check if the gradient is 0 (or near 0)
            if np.linalg.norm(gradient) < self.grad_norm:
                break
            
            # gradient ascent
            self.w = self.w + self.lr * gradient
            i += 1

        
    def predict(self, x):
        x = self.normalize(x)
        return self.link_function(x @ self.w)
    



    

    