import numpy as np
from scipy.special import expit

class GLM:

    def __init__(self, lr=0.01, grad_norm=0.01):
        self.lr = lr
        self.w = None
        self.grad_norm = grad_norm
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
        
    def gradient_ascent(self, X_train, Y_train):

        Y_train = Y_train.flatten()

        X_train = self.normalize(X_train)

        # number of input parameters
        x_size = np.size(X_train, axis=1)

        # initial parameters
        self.w = np.zeros(x_size)

        i = 0
        while(i<1000):

            # find the difference between prediction and y_train
            prediction = self.link_function(np.dot(X_train, self.w))
            
            # find the gradient of likelihood (wrt each component in w)
            gradient =  X_train.T @ (Y_train-prediction)
            # print(np.linalg.norm(prediction-Y_train))
            print(np.linalg.norm(gradient))

            # check if the gradient is 0 (or near 0)
            if np.linalg.norm(gradient) < self.grad_norm:
                break
            
            # gradient ascent
            self.w = self.w + self.lr * gradient
            i += 1
        print(i)

        
    def predict(self, x):
        x = self.normalize(x)
        return self.link_function(x @ self.w)
    


class LogisticRegression(GLM):

    def __init__(self, lr=0.01, grad_norm=0.01):
        super().__init__(lr, grad_norm)
        self.b = None

    def link_function(self, x):
        x = x.astype(float)
        return expit(x)

    def add_intercept(self, X_train):
        # add column of ones to x to train intercept
        b = np.ones(np.size(X_train, axis=0)).reshape(-1, 1)
        X_train = np.concatenate((X_train, b), axis=1)
    
    def gradient_ascent(self, X_train, Y_train):
        # add intercept to X_train
        self.add_intercept(X_train)

        # call gradient ascent from GLM class
        return super().gradient_ascent(X_train, Y_train)
    
    def predict(self, x):
        self.add_intercept(x)
        return super().predict(x)
    
    def tp_fp(self, x, y, threshold):
        samples = x.size(axis=0)
        tp = fp = tn = fn = 0
        for i in range(samples):
            # predict the sample
            prediction = self.predict(x[i])

            # classify each prediction as tp, fp, tn, fn
            if prediction >= threshold:
                if y[i]:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i]:
                    fn += 1
                else:
                    tn += 1

        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        return [tpr, fpr]
    
    def roc(self, x, y):
        tpr_fpr = []
        for i in range(100):
            tpr_fpr.append(self.tp_fp(x, y, i))
        return tpr_fpr

    def auc(self, x, y):
        auc = 0
        roc = self.roc(x, y)
        for rates in roc:
            auc += rates[0]
        return auc
    
    def accuracy(self, predictions, y_test, threshold):
        y_test = y_test.flatten()
        test_size = len(y_test)
        correct = 0
        for p in predictions:
            if p > threshold:
                correct += 1
        return correct/test_size
        
        

    

    