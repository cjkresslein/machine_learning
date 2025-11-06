import numpy as np

class LinearRegression:

    def __init__(self, lr):
        self.alpha = lr
        self.w = None

    # fit model to data
    def fit(self, X_train, Y_train):
        # dimension of x
        x_size = np.size(X_train, axis=0)
        dimension = np.size(X_train, axis=1)

        # initial parameters
        w = np.zeros(x_size)

        # percent difference
        pd = 1

        # find the initial loss
        results = X_train @ w
        loss = 0.5 * np.sum((Y_train - results) ** 2)
        print(f"loss 0: {loss}")
        
        while(pd > 0.05):
            # get sum of differences
            difference = results - Y_train

            # update each component of w
            for i in range(dimension):
                grad = np.sum(difference * X_data[:, i])
                w[i] -= self.alpha * np.sum(grad)

            # find the loss with updated paramaters
            results = X_train @ w
            new_loss = 0.5 * np.sum((Y_train - results) ** 2)
            pd = (loss - new_loss)/loss
            loss = new_loss
            print(f"loss {i+1}: " + str(new_loss) + "\n")

        print(f"final loss: {loss}")
        self.w = w

    def predict(self, x):
        return self.w @ x

            

