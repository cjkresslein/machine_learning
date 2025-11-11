import pandas as pd
import numpy as np

# assign integer values to columns with discrete values
def labelEncode(data, column):
    # data is pd.DataFrame
    encode_column = data[column]
    unique_values = np.sort(encode_column.unique())

    # assign an integer value to each unique value in column
    mapper = {}
    for i in range(len(unique_values)):
        mapper[unique_values[i]] = i

    # replace each value in column with its integer value
    for i in range(len(encode_column)):
        current_val = encode_column[i]
        integer_value = mapper[current_val]
        data.loc[i, column] = integer_value

# split data into x and y
def xy_split(data, y_column):
    y = pd.DataFrame(data[y_column])
    x = data.drop(columns=y_column)
    return x, y

# separate x and y into training and testing data
def train_test_split(x, y, test_size=0.2):
    x_test = pd.DataFrame(columns=x.columns)
    y_test = pd.DataFrame(columns=y.columns)

    # select random row indices for test
    dataset_size = x.size
    population = np.arange(dataset_size)
    test_indices = np.random.choice(population, int(dataset_size*test_size), replace=False)

    for index in test_indices:
        x_test.loc[len(x_test)] = x.iloc[index].copy()
        y_test.loc[len(y_test)] = y.iloc[index].copy()

        index_to_drop = x.iloc[index].name # Get the index label of the 1st row
        x = x.drop(index_to_drop)
        y = y.drop(index_to_drop)


    return (x, x_test, y, y_test)


data = {
    "column1": [1, 2, 3, 4, 5],
    "column2": ["val1", "val2", "val2", "val1", "val1"]
}

df = pd.DataFrame(data)
labelEncode(df, "column2")
x, y = xy_split(df, "column2")
x_train, x_test, y_train, y_test = train_test_split(x, y, 0.2)

print(x_test)