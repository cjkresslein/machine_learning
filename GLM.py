import numpy as np

class GLM:

    def __init__(self, lr=0.01, grad_norm=0.01):
        self.lr = lr
        self.w = None
        self.grad_norm = grad_norm

    def link_function(self):
        pass


    def gradient_ascent(self, X_train, Y_train):

        # number of input parameters
        x_size = np.size(X_train, axis=1)

        # initial parameters
        self.w = np.zeros(x_size)

        while(True):

            # find the difference between prediction and y_train
            prediction = self.link_function(X_train @ self.w)
            
            # find the gradient
            gradient = (Y_train - prediction) @ X_train

            # check if the gradient is 0 (or near 0)
            if np.linalg.norm(gradient) < self.grad_norm:
                break
            
            # gradient ascent
            self.w += self.lr * gradient
        
    def predict(self, x):
        return self.link_function(x @ self.w)
    


class LogisticRegression(GLM):

    def __init__(self, lr=0.01, grad_norm=0.01):
        super().__init__(lr, grad_norm)
        self.b = None

    def link_function(self, x):
        return 1 / 1 + np.exp(-x)
    
    def gradient_ascent(self, X_train, Y_train):
        # add column of ones to x to train intercept
        b = np.ones(np.size(X_train, axis=0)).reshape(-1, 1)
        X_train = np.concatenate((X_train, b), axis=1)

        # call gradient ascent from GLM class
        return super().gradient_ascent(X_train, Y_train)
    
    def predict(self, x):
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
        




    


x = np.array([
    [1, 2, 3],
    [2, 3, 4]
])

y = np.array([7, 8, 9])
logr = LogisticRegression()
logr.gradient_ascent(x, y)

    

    