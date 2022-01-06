import numpy as np
import pandas as pd
import utils
from neurons import Neuron
import matplotlib.pyplot as py
from model import Classifier
from model import Perceptron

"""Paths being used across the code for storing and retrieving data sets"""
paths_to_sets = {"training_X": "./res/data/train/MNISTnum_Train_800_Images.txt",
                 "training_Y": "./res/data/train/MNISTnum_Train_800_Labels.txt",
                 "testing_X": "./res/data/test/MNISTnum_Test_200_Images.txt",
                 "testing_Y": "./res/data/test/MNISTnum_Test_200_Labels.txt",
                 "challenge_X": "./res/data/challenge/MNISTnum_Challenge_200_Images.txt",
                 "challenge_Y": "./res/data/challenge/MNISTnum_Challenge_200_Labels.txt",
                 "dataset_0": "./res/data/datasets/dataset_0.txt",
                 "dataset_1": "./res/data/datasets/dataset_1.txt",
                 "dataset_7": "./res/data/datasets/dataset_7.txt",
                 "dataset_9": "./res/data/datasets/dataset_9.txt",
                 "original_set_X": "./res/MNISTnumImages5000_balanced.txt",
                 "original_set_Y": "./res/MNISTnumLabels5000_balanced.txt"}


def generate_sets_of_data(paths_to_sets, index_x, index_y):
    """Generates the different data sets for training, testing and challenge with the given indexes/columns for
    X and Y datasets."""

    y_col = index_y[0]
    # Reading Images file to get data points for each feature
    df = pd.read_csv(paths_to_sets['original_set_X'], sep='\\s+', header=None, names=index_x)
    # Reading Labels file to get labels for each data point
    df[y_col] = pd.read_csv(paths_to_sets['original_set_Y'], header=None, names=[y_col])[y_col]

    # Preprocessing data. Adding bias to the features for perceptron
    df['bias'] = np.ones((df.shape[0], 1))

    # Selecting all data points with data label as 0
    df_0 = utils.get_df_where(df, df[y_col] == 0)
    utils.store_to_csv(df_0, paths_to_sets["dataset_0"], sep="\t")
    # Selecting all data points with data label as 1
    df_1 = utils.get_df_where(df, df[y_col] == 1)
    utils.store_to_csv(df_0, paths_to_sets["dataset_1"], sep="\t")
    # Selecting all data points with data label as 7
    df_7 = utils.get_df_where(df, df[y_col] == 7)
    utils.store_to_csv(df_0, paths_to_sets["dataset_7"], sep="\t")
    # Selecting all data points with data label as 9
    df_9 = utils.get_df_where(df, df[y_col] == 9)
    utils.store_to_csv(df_0, paths_to_sets["dataset_9"], sep="\t")

    """The following line of code calls my utility function which will generate training and testing data. 
        'df_0' is the set of data points for images with labels 0
        'df_1' is the set of data points for images with labels 1
        We pass them as a list. So that, data points can be picked from them sequentially.
        They are picked as a fraction of the number of rows. Here 0.8, which will take out 400 data points each and will 
        combine them in a dataframe and then shuffle/randomize them row-wise. We will then take out the columns for 
        features and labels, placing them separately in X_train and Y_train, respectively. The process is repeated for 
        obtaining the testing set"""
    # Generate Training and Testing data from the given data points
    x_train, x_test, y_train, y_test = utils.generate_train_test_sets([df_0, df_1], indexes, index_y, 0, True, 0.8)

    """The following line of code calls my utility function which will generate training and testing data. 
        'df_7' is the set of data points for images with labels 7
        'df_9' is the set of data points for images with labels 9
        We pass them as a list. So that, data points can be picked from them sequentially.
        They are picked as a fraction of the number of rows. Here 0.2, which will take out 100 data points each and will 
        combine them in a dataframe and then shuffle/randomize them row-wise. We will then take out the columns for 
        features and labels, placing them separately in X_challenge and Y_challenge, respectively."""
    # Generate Challenge data from the given data points
    x_challenge, waste_set_x, y_challenge, waste_set_y = utils.generate_train_test_sets([df_7, df_9], indexes, index_y,
                                                                                        0, True, 0.2)

    # Storing sets into files for MNIST Set
    utils.store_to_csv(x_train, paths_to_sets['training_X'], sep="\t")
    utils.store_to_csv(y_train, paths_to_sets['training_Y'], sep="\t")

    utils.store_to_csv(x_test, paths_to_sets['testing_X'], sep="\t")
    utils.store_to_csv(y_test, paths_to_sets['testing_Y'], sep="\t")

    utils.store_to_csv(x_challenge, paths_to_sets['challenge_X'], sep="\t")
    utils.store_to_csv(y_challenge, paths_to_sets['challenge_Y'], sep="\t")

    return x_train, x_test, x_challenge, y_train, y_test, y_challenge


def display_weights_image(init_weights, final_weights, function_name):
    """
    Plots a comparison of the initial weights (before training) and the final weights (after training) of the system.

    :param init_weights:
    :param final_weights:
    :param function_name:
    :return:
    """
    fig, subplots = py.subplots(1, 2)
    fig.suptitle(f"Comparison between Initial Weights and Final Weights for {function_name}")
    fig.supylabel("Features of the Image")
    subplots[0].set_xlabel("Initial Weights")
    subplots[0].imshow(init_weights)
    subplots[1].set_xlabel("Final Weights")
    subplots[1].imshow(final_weights)
    py.show()


def calculate_performance_values(y_pred,  y_actual):
    # Taking out those values which were correctly classified and wrongly classified
    correctly_class = y_pred[y_pred['Label'] == y_actual['Label']]
    wrongly_class = y_pred[y_pred['Label'] != y_actual['Label']]

    # Taking out the true_positive and true_negative
    true_pos = correctly_class['Label'][correctly_class['Label'] == 1].count()
    true_neg = correctly_class['Label'][correctly_class['Label'] == 0].count()

    # Taking out the false_positive and false_negative
    false_pos = wrongly_class['Label'][wrongly_class['Label'] == 1].count()
    false_neg = wrongly_class['Label'][wrongly_class['Label'] == 0].count()

    return true_pos, true_neg, false_pos, false_neg


def calculate_performance_on_model(w, args):
    performance = args["performance"]
    key = args['when']

    model = args["model"]
    x_test = args["x_test_set"]
    y_test = args["y_test_set"]
    y_pred = model.predict(x_test, w, {})
    true_pos, true_neg, false_pos, false_neg = calculate_performance_values(y_pred, y_test)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    performance[key] = (precision, recall, f1)


def calculate_error_on_train_test(epoch, w, args):
    error_frac = args['error_frac']

    x_train = args['x_train_set']
    y_train = args['y_train_set']
    y_train_pred = args['model'].predict(x_train, w, {})
    train_err_frac = calculate_error_frac(y_train_pred, y_train)

    x_test = args['x_test_set']
    y_test = args['y_test_set']
    y_test_pred = args['model'].predict(x_test, w, {})
    test_err_frac = calculate_error_frac(y_test_pred, y_test)

    error_frac[epoch] = (train_err_frac, test_err_frac)


def calculate_error_frac(y_pred, y_actual):
    return y_pred['Label'][y_pred['Label'] != y_actual['Label']].count() / y_pred.shape[0]


def train_neuron(neuron: Neuron, x_train, y_train, train_met, update_w_met, epochs, eta, suffix_func):
    if utils.are_sets_available([paths_to_sets["initial_weights"]+suffix_func,
                                 paths_to_sets["final_weights"]+suffix_func]):
        init_weights = pd.read_csv(paths_to_sets["initial_weights"]+suffix_func, sep='\\s+', header=None)
        final_weights = pd.read_csv(paths_to_sets["final_weights"]+suffix_func, sep='\\s+', header=None)
    else:
        init_weights = np.transpose(neuron.weights.copy().values.reshape(28, 28))
        utils.store_to_csv(neuron.weights, paths_to_sets["initial_weights"]+suffix_func, sep="\t")

        # Train the neuron, using the given method

        utils.train_neuron_model(neuron, x_train, y_train, x_train.shape[0], train_met,
                                 update_w_met, epochs=epochs, eta=eta)

        final_weights = np.transpose(neuron.weights.copy().values.reshape(28, 28))
        utils.store_to_csv(neuron.weights, paths_to_sets["final_weights"]+suffix_func, sep="\t")

    return init_weights, final_weights


def perform_neuron_operations(neuron: Neuron, datasets, model: Classifier, function_name, suffix,
                              epochs, eta):
    """Performs Neuron Operations as per problem question description. I have tried to keep comments separating out
    all the steps of the problem. Kindly read the instructions to have a better understanding. """

    """Loading Datasets 
        Separating out the datasets loaded from the various files."""
    x_train, x_test, x_challenge, y_train, y_test, y_challenge = datasets[0], datasets[1], datasets[2], datasets[3],\
                                                                 datasets[4], datasets[5]

    # Arguments to be provided to the model for training:

    """Arguments for calculating performance of the model, before training
       These are passed to the model to calculate the performance, before training as a pre-process"""
    performance = {}
    pre_args = {"x_test_set": x_test, "y_test_set": y_test, "model": model, "performance": performance,
                "when": "before"}

    """ Arguments for calculating performance of the model, after training
        These are passed to the model to calculate the performance, after training as a post-process"""
    post_args = {"x_test_set": x_test, "y_test_set": y_test, "model": model, "performance": performance,
                 "when": "after"}

    """ Arguments for Post-Epoch Operations:
        Arguments being set to calculate the error_fraction, after every epoch."""
    error_frac = {}
    epoch_post_args = {"error_frac": error_frac, "x_train_set": x_train, "y_train_set": y_train,
                       "x_test_set": x_test, "y_test_set": y_test, "model": model}

    # Storing initial weights of the neuron
    init_weights = np.transpose(neuron.weights[:784].copy().values.reshape(28, 28))

    # Calling the model's function 'train' to train the neuron
    neuron.weights = model.train(x_train, y_train, neuron.weights.copy(), epochs=epochs, eta=eta,
                                 pre_process=calculate_performance_on_model,
                                 pre_process_args=pre_args,
                                 epoch_post_process=calculate_error_on_train_test,
                                 epoch_post_process_args=epoch_post_args,
                                 post_process=calculate_performance_on_model,
                                 post_process_args=post_args)

    # Storing final weights of the neuron
    final_weights = np.transpose(neuron.weights[:784].copy().values.reshape(28, 28))

    # Uncomment the following code to see a graph of the weights set [Additional Information]
    # py.figure()
    # py.title("Weights of the Neuron")
    # py.xlabel("Feature Labels")
    # py.ylabel("Feature Weights")
    # py.plot(indexes, neuron.weights)
    # py.legend(loc="upper right")
    # py.show()

    # Plotting Error Fraction of the Neuron, through training as epochs pass
    error_on_training = []
    error_on_testing = []
    for error in error_frac.values():
        error_on_training.append(error[0])
        error_on_testing.append(error[1])

    utils.plot_graph(title="Error Fraction on training and testing set, through training",
                     x_lbl="Epochs", y_lbl="Error Fraction",
                     x_data=list(error_frac.keys()),
                     y_data=[error_on_training, error_on_testing],
                     labels=["Error on training set", "Error on testing set"],
                     legend_loc="upper right")

    # Plotting Precision, Recall and F1 Score values, before and after training.
    labels = ["Before Training", "After Training"]

    width = 0.2
    x = np.arange(len(labels))
    py.figure()
    py.title("Precision, Recall and F1 Scores before and after training")
    py.xlabel("Precision, Recall and F1 Scores")
    precision = py.bar(x - width, [performance['before'][0], performance['after'][0]], width, label='Precision')
    recall = py.bar(x, [performance['before'][1], performance['after'][1]], width, label='Recall')
    f1 = py.bar(x + width, [performance['before'][2], performance['after'][2]], width, label='F1 Score')

    py.xticks(ticks=x, labels=labels)
    py.bar_label(precision, padding=3)
    py.bar_label(recall, padding=3)
    py.bar_label(f1, padding=3)
    py.ylim(0, 1.5)
    py.legend(loc="upper right")
    py.show()

    # Displays images of the initial weights and final weights of the neuron
    display_weights_image(init_weights, final_weights, function_name)

    # Predict values for Challenge Set containing labels: 7 and 9
    y_t = model.predict(x_challenge, neuron.weights, {})

    q_1 = y_t[y_t['Label'] == 1]
    q_0 = y_t[y_t['Label'] == 0]

    # Taking out the true_positive and true_negative
    q_1_7 = q_1['Label'][y_challenge['Label'] == 7].count()
    q_1_9 = q_1['Label'][y_challenge['Label'] == 9].count()

    # Taking out the false_positive and false_negative
    q_0_7 = q_0['Label'][y_challenge['Label'] == 7].count()
    q_0_9 = q_0['Label'][y_challenge['Label'] == 9].count()

    # Printing Prediction Results for the challenge set containing labels: 7 and 9
    print(f"Challenge Set Classification Results for '{function_name}':")
    print('|{:<10}-{:<10}-{:<10}-{:<10}|'.format("-" * 10, "-" * 10, "-" * 10, "-" * 10))
    print('|{:<10}|{:<10}|{:<10}|{:<10}|'.format("Q_17", "Q_19", "Q_07", "Q_09"))
    print('|{:<10}|{:<10}|{:<10}|{:<10}|'.format("-" * 10, "-" * 10, "-" * 10, "-" * 10))
    print('|{:<10}|{:<10}|{:<10}|{:<10}|'.format(q_1_7, q_1_9, q_0_7, q_0_9))
    print('|{:<10}-{:<10}-{:<10}-{:<10}|'.format("-" * 10, "-" * 10, "-" * 10, "-" * 10))


if __name__ == '__main__':
    # Setting up constant values for execution
    features = 784
    max_theta = 40

    epochs = 15
    eta = 0.01

    # Preparing indexes/column names for each feature in the feature space
    indexes = ['f'+str(i) for i in range(1, features + 1)]
    indexes.append('bias')

    if utils.are_sets_available(paths_to_sets.values()):
        datasets = utils.load_sets_of_data(paths_to_sets, indexes, ['Label'])
    else:
        datasets = generate_sets_of_data(paths_to_sets, indexes, ['Label'])

    # Preprocessing data. Adding bias to the features for perceptron
    datasets[0]['bias'] = np.ones((datasets[0].shape[0], 1))
    datasets[2]['bias'] = np.ones((datasets[2].shape[0], 1))
    datasets[4]['bias'] = np.ones((datasets[4].shape[0], 1))

    # Show Image [Additional Image]
    # Uncomment the following code to show image
    # from matplotlib import pyplot as plt
    # plt.imshow(datasets[0].iloc[0].copy().values.reshape(28, 28), interpolation='nearest')
    # plt.show()

    # Implementing Perceptron from here on
    # Initializing a Neuron with its weights
    neuron = Neuron(pd.Series(utils.get_rand_weights(0, 0.5, features+1), index=indexes))
    model = Perceptron()
    perform_neuron_operations(neuron, datasets, model, "Neuron trained using Perceptron",
                              "_perceptron.txt", epochs=epochs, eta=eta)

