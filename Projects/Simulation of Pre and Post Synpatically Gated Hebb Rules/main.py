import numpy as np
import pandas as pd
import utils
from neurons import Neuron
from neurons import BinaryThresholdNeuron
import matplotlib.pyplot as py

paths_to_sets = {"training_X": "../res/data/train/MNISTnum_Train_800_Images.txt",
                 "training_Y": "../res/data/train/MNISTnum_Train_800_Labels.txt",
                 "testing_X": "../res/data/test/MNISTnum_Test_200_Images.txt",
                 "testing_Y": "../res/data/test/MNISTnum_Test_200_Labels.txt",
                 "challenge_X": "../res/data/challenge/MNISTnum_Challenge_200_Images.txt",
                 "challenge_Y": "../res/data/challenge/MNISTnum_Challenge_200_Labels.txt",
                 "dataset_0": "../res/data/datasets/dataset_0.txt",
                 "dataset_1": "../res/data/datasets/dataset_1.txt",
                 "dataset_7": "../res/data/datasets/dataset_7.txt",
                 "dataset_9": "../res/data/datasets/dataset_9.txt",
                 "initial_weights": "../res/data/weights/initial_weights",
                 "final_weights": "../res/data/weights/final_weights",
                 "original_set_X": "../res/MNISTnumImages5000_balanced.txt",
                 "original_set_Y": "../res/MNISTnumLabels5000_balanced.txt"}


def generate_sets_of_data(paths_to_sets, index_x, index_y):
    """Generates the different data sets for training, testing and challenge with the given indexes/columns for
    X and Y datasets."""

    y_col = index_y[0]
    # Reading Images file to get data points for each feature
    df = pd.read_csv(paths_to_sets['original_set_X'], sep='\\s+', header=None, names=index_x)
    # Reading Labels file to get labels for each data point
    df[y_col] = pd.read_csv(paths_to_sets['original_set_Y'], header=None, names=[y_col])[y_col]

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
    X_train, X_test, Y_train, Y_test = utils.generate_train_test_sets([df_0, df_1], indexes, index_y, 0, True, 0.8)

    """The following line of code calls my utility function which will generate training and testing data. 
        'df_7' is the set of data points for images with labels 7
        'df_9' is the set of data points for images with labels 9
        We pass them as a list. So that, data points can be picked from them sequentially.
        They are picked as a fraction of the number of rows. Here 0.2, which will take out 100 data points each and will 
        combine them in a dataframe and then shuffle/randomize them row-wise. We will then take out the columns for 
        features and labels, placing them separately in X_challenge and Y_challenge, respectively."""
    # Generate Challenge data from the given data points
    X_challenge, waste_set_X, Y_challenge, waste_set_Y = utils.generate_train_test_sets([df_7, df_9], indexes, index_y,
                                                                                        0, True, 0.2)

    # Storing sets into files for MNIST Set
    utils.store_to_csv(X_train, paths_to_sets['training_X'], sep="\t")
    utils.store_to_csv(Y_train, paths_to_sets['training_Y'], sep="\t")

    utils.store_to_csv(X_test, paths_to_sets['testing_X'], sep="\t")
    utils.store_to_csv(Y_test, paths_to_sets['testing_Y'], sep="\t")

    utils.store_to_csv(X_challenge, paths_to_sets['challenge_X'], sep="\t")
    utils.store_to_csv(Y_challenge, paths_to_sets['challenge_Y'], sep="\t")

    return X_train, X_test, X_challenge, Y_train, Y_test, Y_challenge


def train_simple_threshold_neuron(neuron: Neuron, x_i, z_t):
    """Trains a simple threshold neuron by determining the value of y^ that should be given by the neuron.
    <Specific Implementation to a given neuron model>"""
    # The value of s(t) is not used as we are relying on the value of teaching input: z(t)[z_t]
    # s_t = np.dot(x_i, neuron.weights)
    return z_t['Label']


def update_neuron_weights_post_hebb(neuron: Neuron, x_t, y_t, eta):
    """Updates the weights of the given neuron with post-synaptically gated hebb rule.
       The formula is given by:
        w_j(t) = w_j(t-1) + n x y(t) x [x_j(t) - w_j(t-1)]
    <Specific Implementation to a given neuron model>"""
    weights = neuron.weights
    weights = weights + (eta * y_t * (x_t - weights))
    neuron.weights = weights


def update_neuron_weights_pre_hebb(neuron: Neuron, x_t, y_t, eta):
    """Updates the weights of the given neuron with pre-synaptically gated hebb rule.
       The formula is given by:
        w_j(t) = w_j(t-1) + n x x_j(t) x [y(t) - w_j(t-1)]
    <Specific Implementation to a given neuron model>"""
    weights = neuron.weights
    weights = weights + (eta * x_t * (y_t - weights))
    neuron.weights = weights


def predict(x_i, neuron: Neuron, theta):
    """Predicts the value that the neuron will give, as a result of it's training.
    This function takes a set of values (representing the X dataset containing values of all features for the dataset),
    a neuron (whose weights have already been trained), and a theta value (based on which the binary neuron model will
    decide whether the data point belongs class 0 or 1).
    <Implementation Specific to Simple Neuron model>
    """
    # Net Input
    s_t = np.dot(x_i, neuron.weights)
    if s_t > theta:
        return 1
    else:
        return 0


def run_prediction_for_theta(x, y, max_theta):
    predicted_y = {}
    precisions = []
    recalls = []
    f1s = []
    tps = []
    fps = []
    fns = []
    tns = []
    thetas = []
    for theta in range(0, max_theta+1):
        y_t = utils.predict_from_neuron(neuron=binary_neuron, x=x, prediction=predict, theta=theta)
        # Taking out those values which were correctly classified and wrongly classified
        correctly_class = y_t[y_t['Label'] == y['Label']]
        wrongly_class = y_t[y_t['Label'] != y['Label']]

        # Taking out the true_positive and true_negative
        true_pos = correctly_class['Label'][correctly_class['Label'] == 1].count()
        true_neg = correctly_class['Label'][correctly_class['Label'] == 0].count()

        # Taking out the false_positive and false_negative
        false_pos = wrongly_class['Label'][wrongly_class['Label'] == 1].count()
        false_neg = wrongly_class['Label'][wrongly_class['Label'] == 0].count()

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        # Uncomment following lines to check the values at particular theta
        # print(f"At Theta value: {theta}")
        # print(true_pos, true_neg, false_pos, false_neg)
        # print("Precision: ", precision)
        # print("Recall: ", recall)

        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0

        predicted_y[theta] = (y_t, true_pos, true_neg, false_pos, false_neg, precision, recall, f1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        tps.append(true_pos)
        fps.append(false_pos)
        tns.append(true_neg)
        fns.append(false_neg)
        thetas.append(theta)
    return predicted_y, precisions, recalls, f1s, tps, fps, tns, fns, thetas


def get_best_val_theta(fps, tps, max_theta):
    fps_det = [fps[0]]
    for i in range(0, len(fps)-1):
        fps_det.append(fps[i+1] - fps[i])

    tps_det = [tps[0]]
    for i in range(0, len(tps)-1):
        tps_det.append(tps[i+1] - tps[i])

    max_val = [0, 0]
    for fp, tp, theta in zip(fps_det, tps_det, range(0, max_theta)):
        if fp != 0:
            slope = tp/fp
            if 1 > slope > max_val[0]:
                max_val[0] = slope
                max_val[1] = theta
    return max_val


def display_weights_image(init_weights, final_weights, function_name):
    fig, subplots = py.subplots(1, 2)
    fig.suptitle(f"Comparison between Initial Weights and Final Weights for {function_name}")
    fig.supylabel("Features of the Image")
    subplots[0].set_xlabel("Initial Weights")
    subplots[0].imshow(init_weights)
    subplots[1].set_xlabel("Final Weights")
    subplots[1].imshow(final_weights)
    py.show()


def train_neuron(neuron: Neuron, x_train, y_train, train_met, update_w_met, epochs, eta, suffix_func):
    """
    This method is used to train the given neuron on the given training sets: (x_train, y_train) for the given number of
    epochs, eta, which are passed to a utility function which will call the given training and update_weights function,
    during training.
    :param neuron: the neuron to be trained
    :param x_train: the training set containing the features of the dataset.
    :param y_train: the corresponding labels of the training set.
    :param train_met: the reference to the method to be used for training
    :param update_w_met: the reference to the update weights method to be used for updating the weights.
    :param epochs: the max number of epochs for which training will happen.
    :param eta: the learning rate (hyper-parameter) to be applied for training
    :param suffix_func: Additional parameter to be used for storing information for specific cases, while making
    implementation generic.
    :return: a tuple of initial and final weights
    """
    if utils.are_sets_available([paths_to_sets["initial_weights"]+suffix_func,
                                 paths_to_sets["final_weights"]+suffix_func]):
        init_weights = pd.read_csv(paths_to_sets["initial_weights"]+suffix_func, sep='\\s+', header=None)
        final_weights = pd.read_csv(paths_to_sets["final_weights"]+suffix_func, sep='\\s+', header=None)
    else:
        init_weights = np.transpose(neuron.weights.copy().values.reshape(28, 28))
        utils.store_to_csv(neuron.weights, paths_to_sets["initial_weights"]+suffix_func, sep="\t")

        # Train the neuron, using Simple Neuron model
        utils.train_neuron_model(neuron, x_train, y_train, x_train.shape[0], train_met,
                                 update_w_met, epochs=epochs, eta=eta)

        final_weights = np.transpose(neuron.weights.copy().values.reshape(28, 28))
        utils.store_to_csv(neuron.weights, paths_to_sets["final_weights"]+suffix_func, sep="\t")

    return init_weights, final_weights


def perform_neuron_operations(neuron: Neuron, datasets, train_method, update_weights_method, function_name, suffix,
                              epochs, eta):
    """
    Performs the required operations on the neuron passed. This includes training, prediction, calculating performance
    on testing set and then plotting them
    :param neuron: the neuron to be used for operations
    :param datasets: a tuple containing the training and testing datasets.
    :param train_method: a reference to the method to be used for training
    :param update_weights_method: a reference to the method to be used for updating weights
    :param function_name: the name of the function, used for making method generic.
    :param suffix: the suffix of the function, used for making method generic.
    :param epochs: the max number of epochs, to be used for training
    :param eta: the learning rate to be used for training
    :return: None
    """
    x_train, x_test, x_challenge, y_train, y_test, y_challenge = datasets[0], datasets[1], datasets[2], datasets[3],\
                                                                 datasets[4], datasets[5]

    init_weights, final_weights = train_neuron(neuron, x_train, y_train, train_method,
                                               update_weights_method, epochs, eta, suffix)

    # Uncomment the following code to see a graph of the weights set
    # py.figure()
    # py.title("Weights of the Neuron")
    # py.xlabel("Feature Labels")
    # py.ylabel("Feature Weights")
    # py.plot(indexes, binary_neuron.weights)
    # py.legend(loc="upper right")
    # py.show()

    # Predicting y_values
    # Storing values for true positive, true negative, false positive and false negative
    predicted_y, precisions, recalls, f1s, tps, fps, tns, fns, thetas = run_prediction_for_theta(x_test, y_test,
                                                                                                 max_theta)

    # Plots Recall Curve
    utils.plot_graph(title=f"Precision, Recall and F1 Scores Plot V/S Theta for '{function_name}'",
                     x_lbl="Theta values", y_lbl="Precision/ Recall/ F1 Values",
                     x_data=thetas, y_data=[precisions, recalls, f1s],
                     labels=["Precision", "Recall", "F1(s)"], legend_loc="upper right")

    # Plots ROC Curve
    utils.plot_graph(title=f"ROC Curve for '{function_name}'",
                     x_lbl="False Positive", y_lbl="True Positive",
                     x_data=fps, y_data=tps)

    # Gets best value of theta
    best_theta = get_best_val_theta(fps, tps, max_theta)

    # Displays images of the initial weights and final weights of the neuron
    display_weights_image(init_weights, final_weights, function_name)

    # Predict values for Challenge Set containing labels: 7 and 9
    y_t = utils.predict_from_neuron(neuron=binary_neuron, x=x_challenge, prediction=predict, theta=best_theta[1])

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

    epochs = 40
    eta = 0.01

    # Preparing indexes/column names for each feature in the feature space
    indexes = ['f'+str(i) for i in range(1, features + 1)]

    if utils.are_sets_available(paths_to_sets.values()):
        datasets = utils.load_sets_of_data(paths_to_sets, indexes, ['Label'])
    else:
        datasets = generate_sets_of_data(paths_to_sets, indexes, ['Label'])

    # Show Image
    # Uncomment the following code to show image
    # from matplotlib import pyplot as plt
    # plt.imshow(X_train.iloc[0].copy().values.reshape(28, 28), interpolation='nearest')
    # plt.show()

    # Implementing Post-Hebb Rule from here on
    # Initializing BinaryThresholdNeuron with its weights
    binary_neuron = BinaryThresholdNeuron(pd.Series(utils.get_rand_weights(0, 0.5, features), index=indexes))
    perform_neuron_operations(binary_neuron, datasets, train_simple_threshold_neuron, update_neuron_weights_post_hebb,
                              "Neuron trained with Post-Synaptically Gated Hebb Rule", "_post.txt", epochs=epochs, eta=eta)

    # Implementing Pre-Hebb Rule from here on
    # Initializing BinaryThresholdNeuron with its weights
    binary_neuron = BinaryThresholdNeuron(pd.Series(utils.get_rand_weights(0, 0.5, features), index=indexes))
    perform_neuron_operations(binary_neuron, datasets, train_simple_threshold_neuron, update_neuron_weights_pre_hebb,
                              "Neuron trained with Pre-Synaptically Gated Hebb Rule", "_pre.txt", epochs=epochs, eta=eta)
