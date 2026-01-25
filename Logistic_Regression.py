from .GLM import GLM
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

class LogisticRegression(GLM):

    def __init__(self):
        super().__init__()
        self.b = None

    def link_function(self, x):
        if type(x) is float:
            return expit(x)
        x = x.astype(float)
        return expit(x)

    # add column of ones to x to train intercept
    def add_intercept(self, X_train):
        # dimension == 1
        if X_train.ndim == 1:
            X_train = np.append(X_train, 1)
            return X_train
        
        # dimension > 1
        b = np.ones(np.size(X_train, axis=0)).reshape(-1, 1)
        X_train = np.concatenate((X_train, b), axis=1)
    
    def gradient_ascent(self, X_train, Y_train, lr=0.01, grad_norm=0.1, normalize=False):
        # add intercept to X_train
        self.add_intercept(X_train)

        # call gradient ascent from GLM class
        return super().gradient_ascent(X_train, Y_train, lr, grad_norm, normalize)
    
    def predict(self, x):
        self.add_intercept(x)
        return super().predict(x)
    
    def tp_fp(self, x, y, threshold):
        samples = len(x)
        tp = fp = tn = fn = 0
        for i in range(samples):
            # predict the sample
            prediction = self.predict(x[i])

            # classify each prediction as tp, fp, tn, fn
            if prediction >= threshold/100:
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
        y = y.flatten()
        tpr_fpr = []
        for i in range(100):
            rates = self.tp_fp(x, y, i)
            tpr_fpr.append(rates)
        return tpr_fpr

    def auc(self, x, y):
        auc = 0
        roc = self.roc(x, y)
        for rates in roc:
            auc += rates[0]
        return auc/100
    
    def plot_roc(self, x, y):
        roc = self.roc(x, y)
        
        tpr = [tpr_fpr[0] for tpr_fpr in roc]
        fpr = [tpr_fpr[1] for tpr_fpr in roc]
        thresholds = [i for i in range(1, 100)]

        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")  
        plt.title("ROC")

        # add threshold labels
        for xi, yi, label in zip(fpr, tpr, thresholds):
            plt.text(xi, yi, label, fontsize=9, ha='right', va='bottom')
        plt.show()   

    
    def accuracy(self, predictions, y_test, threshold):
        y_test = y_test.flatten()
        test_size = len(y_test)
        correct = 0
        for i in range(len(predictions)):
            if (predictions[i] >= threshold):
                if (y_test[i] == 1):
                    correct += 1
            else:
                if y_test[i] == 0:
                    correct += 1
        return correct/test_size
        
        
