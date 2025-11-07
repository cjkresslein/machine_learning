import numpy as np

class LinearRegression:

    def __init__(self, lr=0.01, terminate_function="lossChange", threshold=0.01):
        self.alpha = lr
        self.w = None
        self.threshold = threshold

        terminate_funcs = {
            "lossThreshold": self.loss_terminate,
            "lossChange": self.loss_change_terminate
        }

        if terminate_function in terminate_funcs.keys():
            self.terminate_function = terminate_funcs[terminate_function]
        else:
            raise "Error: undefined termination functions"

    # terminate gradient descent based on desired loss
    def loss_terminate(self, *args):
        loss = args[0]
        desired_loss = args[1]
        print(f"loss is less than {desired_loss}")
        return (loss <= desired_loss)
    
    # terminate gradient descent based on percentage change in loss
    def loss_change_terminate(self, *args):
        loss = args[0]
        change_minimum = args[1]
        old_loss = args[2]
        return (abs(loss-old_loss)/old_loss) <= change_minimum

    # fit model to data with linear regression
    def fit(self, X_train, Y_train):
        # number of training samples
        x_size = np.size(X_train, axis=0)

        # initial parameters
        w = np.zeros(x_size)


        # find the initial loss
        results = X_train @ w
        loss = 0.5 * np.sum((Y_train - results) ** 2)
        print(f"loss 0: {loss}")
        
        i=1
        while(True):
            # get sum of differences
            difference = results - Y_train

            # update each component of w
            grad = difference @ X_train
            w -= self.alpha * grad

            # find the loss with updated paramaters
            results = X_train @ w
            new_loss = 0.5 * np.sum((Y_train - results) ** 2)

            # check if gradient descent should terminate
            if self.terminate_function(new_loss, self.threshold, loss):
                break

            # display loss during training itertions
            loss = new_loss
            print(f"loss {i+1}: " + str(loss) + "\n")
            i += 1

        print(f"final loss: {loss}")
        self.w = w

    # fit model to data with Normal equation
    def fit_normal(self, X_train, Y_train):
        
        # generate a square matrix with x
        x_matrix = X_train.T @ X_train
        
        # check if the matrix is singluar (determinant near 0)
        tolerance = 1e-9
        if np.linalg.det(x_matrix) < tolerance:
            raise "cannot use normal equation with singluar matrix"
        
        inv_x = np.linalg.inv(x_matrix)
        xty = X_train.T @ Y_train
        self.w = inv_x @ xty


    def predict(self, x):
        return self.w @ x

            

x = np.array([
    [1, 1000, 2],
    [1, 1500, 3],
    [1, 2000, 3],
    [1, 2500, 4]
])

y = np.array([180, 240, 310, 375])

model = LinearRegression()

model.fit_normal(x, y)
print(model.w)

print(model.predict(np.array([1, 1800, 3])))