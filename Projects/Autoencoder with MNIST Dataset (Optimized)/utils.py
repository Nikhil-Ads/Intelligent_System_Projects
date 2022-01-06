import os.path

import pandas as pd
import numpy as np


def get_df_where(df: pd.DataFrame, condition):
    """Returns a copy of the dataframe with the given select condition."""
    return df.copy()[condition]


def are_sets_available(sets_path):
    for set_path in sets_path:
        if not os.path.exists(set_path):
            return False
    return True


def generate_train_test_sets(df, x_columns, y_columns, axis=0, ignore_index=True, frac=1):
    train_set = None
    test_set = None

    rows = df[0].shape[0]
    train_size = int(rows * frac)
    if type(df) == list:
        d = []
        for frame in df:
            d.append(frame[:train_size])
        train_set = pd.concat(d, axis=axis, ignore_index=ignore_index).sample(frac=1)

        d.clear()
        for frame in df:
            d.append(frame[train_size:])
        test_set = pd.concat(d, axis=axis, ignore_index=ignore_index).sample(frac=1)

    return train_set[x_columns], test_set[x_columns], train_set[y_columns], test_set[y_columns]


def generate_train_test_sets_rand(df, x_columns, y_columns, axis=0, ignore_index=True, frac=1):
    if type(df) == list:
        df = pd.concat(df, axis=axis, ignore_index=ignore_index)

    train_set = df.sample(frac=frac)
    test_set = df.sample(frac=(1 - frac))

    return train_set[x_columns], test_set[x_columns], train_set[y_columns], test_set[y_columns]


def store_to_csv(df: pd.DataFrame, path: str, sep=",", headers=None, indexes=None, overwrite=False):
    if not os.path.exists(path) or overwrite:
        dirs = os.path.dirname(path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        return df.to_csv(path, sep=sep, header=headers, index=indexes)


def get_rand_weights_tup(tup: tuple):
    return np.random.uniform(tup[0], tup[1], tup[2])


def get_rand_weights(low=0, high=1, size=1):
    return np.random.uniform(low, high, size)


def train_neuron_model(neuron, x: pd.DataFrame, y: pd.DataFrame, set_size, train, adjust_weights, epochs=40, eta=0.01,
                       tolerance=0.001):
    """This is a generic training method. It takes the number of epochs"""
    # prev_weights = neuron.weights
    for epoch in range(epochs):
        for sample in range(set_size):
            x_i = x.iloc[sample]
            y_i = y.iloc[sample]

            y_t = train(neuron, x_i, y_i)
            # print(list(x_i), '\n', list(y_t))
            adjust_weights(neuron, x_i, y_t, eta)
        # print(f"Neuron Weights, after epoch {epoch}: ", neuron.weights)


def predict_from_neuron(neuron, x: pd.DataFrame, prediction, theta=0, axis=1):
    return pd.DataFrame(x.apply(prediction, axis=axis, args=(neuron, theta)), columns=['Label'])


def load_sets_of_data(paths_to_sets, indexes_x, indexes_y):
    x_train = pd.read_csv(paths_to_sets['training_X'], sep='\\s+', header=None, names=indexes_x)
    y_train = pd.read_csv(paths_to_sets['training_Y'], sep='\\s+', header=None, names=indexes_y)
    x_test = pd.read_csv(paths_to_sets['testing_X'], sep='\\s+', header=None, names=indexes_x)
    y_test = pd.read_csv(paths_to_sets['testing_Y'], sep='\\s+', header=None, names=indexes_y)
    x_challenge = pd.read_csv(paths_to_sets['challenge_X'], sep='\\s+', header=None, names=indexes_x)
    y_challenge = pd.read_csv(paths_to_sets['challenge_Y'], sep='\\s+', header=None, names=indexes_y)

    return x_train, x_test, x_challenge, y_train, y_test, y_challenge


def convert_to_list(data, level, lvls=1):
    if level >= lvls:
        return data
    size = len(data)
    if size == 0:
        raise Exception("Data should not be empty")
    elif size == 1:
        if type(data[0]) == list:
            return data
    else:
        if type(data[0]) == list:
            return data
        else:
            return [convert_to_list(data, level + 1, lvls)]


def print_sep(sep='=', length=100):
    print(sep * length)


def plot_graph(title, x_lbl, y_lbl, x_data, y_data, labels=None, legend_loc=None):
    x_data = convert_to_list(x_data, 1, 2)
    y_data = convert_to_list(y_data, 1, 2)

    import matplotlib.pyplot as py
    py.figure()
    py.title(title)
    py.xlabel(x_lbl)
    py.ylabel(y_lbl)

    if len(x_data) == 1:
        x = x_data[0]
        if labels is None:
            labels = ['']
        for lbl, y in zip(labels, y_data):
            py.plot(x, y, label=lbl)
    elif len(y_data) == 1:
        y = y_data[0]
        if not labels:
            labels = ['']
        for lbl, x in zip(labels, x_data):
            py.plot(x, y, label=lbl)
    else:
        if len(labels) == 1:
            labels = [label for label in labels]
        for lbl, x, y in zip(labels, x_data, y_data):
            py.plot(x, y, label=lbl)
    if legend_loc:
        py.legend(loc=legend_loc)
    py.show()


def get_confusion_matrix(df_1: pd.DataFrame, df_2: pd.DataFrame, column):
    num = df_1[column].unique()
    num = np.sort(num)
    nums = num.shape[0]

    values = pd.DataFrame()
    total_i = []
    total_j = []
    for row in range(nums):
        val_in_row = []
        df_i = df_1[df_1[column] == num[row]]
        total_i.append(df_i.count())
        for col in range(nums):
            df_j = df_2[df_2[column] == num[col]]
            total_j.append(df_j.count())
            df_i_j = df_i[df_i[column] == df_j[column]]
            val_in_row.append(df_i_j.count())
        values[row] = val_in_row

    sum_i = 0
    sum_j = 0
    for i, j in zip(total_i, total_j):
        sum_i += i
        sum_j += j

    values['Actual_Total'] = total_j
    values = values.T
    values['Assigned_total'] = total_i

    return values


def create_confusion_matrix(df_1: pd.DataFrame, df_2: pd.DataFrame, column):
    nums = df_1[column].unique()
    nums = np.sort(nums)
    num = df_1.shape[0]

    mat = {i: {j: 0 for j in nums} for i in nums}
    for i in range(num):
        row = df_1.iloc[i][column]
        col = df_2.iloc[i][column]
        mat[row][col] = mat[row][col] + 1

    df = pd.DataFrame(mat)
    # df_assigned_total = df[[i for i in nums]].sum()
    # df['Assigned Total'] = df.iloc[0:num].sum()
    # df['Actual Total'] = df_assigned_total
    return df.T
