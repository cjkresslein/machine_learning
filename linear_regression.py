import numpy as np

class LinearRegression:

    def __init__(self, lr, terminate_function="lossChange", threshold=0.01):
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

    # fit model to data
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

    def predict(self, x):
        return self.w @ x

            

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

y = np.array([6, 15, 24])

model = LinearRegression(lr=0.001, terminate_function="lossThreshold", threshold=0.05)

model.fit(x, y)

print(model.predict(np.array([10, 11, 12])))