import datetime

import matplotlib.pyplot as plt
import pandas as pd
from model import FeedForwardNetwork
import utils
import numpy as np

paths_to_sets = {"training_X": "./res/data/train/MNISTnum_Train_Images.txt",
                 "training_Y": "./res/data/train/MNISTnum_Train_Labels.txt",
                 "testing_X": "./res/data/test/MNISTnum_Test_Images.txt",
                 "testing_Y": "./res/data/test/MNISTnum_Test_Labels.txt",
                 "dataset_0": "./res/data/datasets/dataset_0.txt",
                 "dataset_1": "./res/data/datasets/dataset_1.txt",
                 "dataset_2": "./res/data/datasets/dataset_2.txt",
                 "dataset_3": "./res/data/datasets/dataset_3.txt",
                 "dataset_4": "./res/data/datasets/dataset_4.txt",
                 "dataset_5": "./res/data/datasets/dataset_5.txt",
                 "dataset_6": "./res/data/datasets/dataset_6.txt",
                 "dataset_7": "./res/data/datasets/dataset_7.txt",
                 "dataset_8": "./res/data/datasets/dataset_8.txt",
                 "dataset_9": "./res/data/datasets/dataset_9.txt",
                 "original_set_X": "./res/MNISTnumImages5000_balanced.txt",
                 "original_set_Y": "./res/MNISTnumLabels5000_balanced.txt"}

paths_to_other_files = {"weights_of_layer": "./res/data/weights/final_weights_layer_",
                        "neurons_plotted": "./res/data/plots/neuron_features.csv"
                        }

training_error = 1.0
print_error = False


def create_or_get_datasets(index_x, index_y, axis=0, ignore_index=True, train_set_frac=1.0):
    if utils.are_sets_available(paths_to_sets):
        datasets = utils.load_sets_of_data(paths_to_sets, index_x, index_y)
    else:
        df_x = pd.read_csv(paths_to_sets['original_set_X'], sep='\\s+', header=None, names=index_x)
        df_y = pd.read_csv(paths_to_sets['original_set_Y'], header=None, names=index_y)

        df = df_x.copy()
        df['Label'] = df_y['Label']

        df_i = []
        for i in range(10):
            df_ = df[df['Label'] == i]
            utils.store_to_csv(df_, paths_to_sets[f'dataset_{i}'], sep='\t')
            df_i.append(df_)

        x_train, x_test, y_train, y_test = utils.generate_train_test_sets(df_i, index_x, index_y, axis, ignore_index,
                                                                          train_set_frac)

        utils.store_to_csv(x_train, paths_to_sets['training_X'], sep='\t')
        utils.store_to_csv(y_train, paths_to_sets['training_Y'], sep='\t')
        utils.store_to_csv(x_test, paths_to_sets['testing_X'], sep='\t')
        utils.store_to_csv(y_test, paths_to_sets['testing_Y'], sep='\t')

        return x_train, x_test, y_train, y_test


def before_epoch(epoch, args):
    model = {}
    utils.print_sep()
    print(f"Epoch {epoch + 1}:")
    x = args['x_train']
    y = args['y_train']

    x_index = args['x_index']
    y_index = args['y_index']

    train_set = x.sample(frac=args['frac'])
    model['x_train'] = train_set
    model['y_train'] = train_set
    model['set_size'] = train_set.shape[0]
    model['start_time'] = datetime.datetime.now()
    return model


def __winner_takes_all_output(y, index_y=None):
    if type(y) == pd.Series:
        axis = 0
    else:
        axis = 1
    result = y.apply(FeedForwardNetwork.__get_index_of_highest_output__, axis=axis)
    result = pd.DataFrame({'Label': result.to_list()}, index=index_y)
    return result


def __store_training_error(train_error: float):
    global training_error
    training_error = train_error


def stop_training_condition(epoch, args):
    global print_error
    condition = True if training_error <= float(args['min_train_error']) and epoch + 1 >= args['min-epochs'] else False
    if condition:
        print_error = True
    return condition


def pre_process(args):
    args['loss'] = np.array([])
    calculate_error_on_train_test_bt(args)


def calculate_error_on_train_test_bt(args):
    return calculate_error_on_train_test(-1, args)


def calculate_error_on_train_test(epoch, args):
    global print_error
    model = args
    args = args['epoch_post_process_args']
    epochs_interval = args['epochs_interval']
    error_frac = args['error_frac']

    if epoch+1 > 0:
        print(f"Time taken in Epoch {epoch+1}: ", ((datetime.datetime.now() - model['start_time']).seconds % 3600) // 60
              , ' min')
    x_train = model['x_train']
    y_train = model['y_train']
    if 'y_pred' in model:
        y_train_pred = pd.DataFrame(model['y_pred'][epoch], columns=y_train.columns)
    else:
        y_train_pred = args['model'].predict(x_train, {})

    train_err_frac = lms(y_train_pred, y_train)
    __store_training_error(train_err_frac)
    model['loss'] = np.append(model['loss'], train_err_frac)

    if epoch+1 > 0:
        error_frac[epoch] = train_err_frac

    if epoch == 0 or (epoch+1) % epochs_interval == 0 or print_error:
        print("J2 Loss over training set: ", train_err_frac)


def lms(y_predicted: pd.DataFrame, y_actual: pd.DataFrame):
    return np.sum(np.power(np.subtract(y_predicted.to_numpy(), y_actual.to_numpy()), 2))/2


def store_weights(weights: dict, n_layers):
    common_path = paths_to_other_files['weights_of_layer']
    for layer in range(n_layers):
        columns = len(weights[layer][0])
        df = pd.DataFrame(weights[layer], columns=[i for i in range(columns)])
        utils.store_to_csv(df, common_path+f"{layer}.csv", sep='\t', overwrite=True)


def plot_features_of_neurons(w, layer, neurons_n_layer, n_neurons, plot_rows=1, plot_cols=1, n=None):
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols)
    fig.suptitle(f"Features of {n_neurons} from the hidden layer")
    fig.supxlabel("Features")
    fig.supylabel("Features")
    if n is not None:
        for i in range(plot_rows):
            for j in range(plot_cols):
                ax[i][j].imshow(w[layer][n[i+j]].reshape(28, 28), cmap='gray')
    else:
        n = []
        for i in range(plot_rows):
            for j in range(plot_cols):
                neuron_i = np.random.randint(0, neurons_n_layer)
                ax[i][j].imshow(w[layer][neuron_i].reshape(28, 28), cmap='gray')
                n.append(neuron_i)
    plt.show()
    return n


def plot_outputs_of_net(x_input, x_output, n_rows=1, n_cols=1):
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.suptitle(f"Comparison of {x_input.shape[0]} Inputs and Outputs to the AutoEncoder")
    for i in range(n_rows//2):
        for j in range(n_cols):
            ax[i][j].imshow(x_input.iloc[j].to_numpy().reshape(28, 28))
        for j in range(n_cols):
            ax[i+1][j].imshow(x_output.iloc[j].to_numpy().reshape(28, 28))
    plt.show()


if __name__ == '__main__':
    features = 784
    indexes = [i for i in range(features)]
    # indexes.append('bias')

    nn = r""" |\   | ______ |     | |
              | \  | |      |     | |
              |  \ |  ===   |     | |
              |   \| |_____ \____/  |"""

    """Gets or Creates the training and testing datasets from the stored locations."""
    X_train, X_test, Y_train, Y_test = create_or_get_datasets(indexes, ['Label'], train_set_frac=0.8)

    """Taking inputs for creating Neural networks"""
    utils.print_sep()
    print("Inputs to initialize the neural network:")
    layers = int(input("Enter the number of layers for neural network: "))
    neurons_in_layers = []
    for i in range(1, layers + 1):
        if i < layers:
            neurons_in_layers.append(int(input(f"Enter the number of neurons for hidden-layer {i}: ")))
        else:
            neurons_in_layers.append(int(input(f"Enter the number of neurons for output-layer: ")))
    utils.print_sep()

    print("Initializing Feed-Forward Neural Network ...")

    weights = None
    train = True
    if utils.are_sets_available([paths_to_other_files['weights_of_layer']+f"{layer}.csv" for layer in range(layers)]):
        read_previous_weights = input("Previously stored weights found. Do you want to use these weights? ")
        if read_previous_weights.lower() in ['yes', 'y', 'true']:
            train = False
            weights = []
            for layer in range(layers):
                df = pd.read_csv(paths_to_other_files['weights_of_layer']+f"{layer}.csv", sep='\\s+', header=None)\
                    .to_numpy()
                if neurons_in_layers[layer] != df.shape[0]:
                    print("Neurons mismatch. The model will continue to train.")
                    train = True
                    break
                else:
                    weights.append(df)
            if not train:
                weights = np.array(weights, dtype=object)
                train = False
            utils.print_sep()

    """Arguments for model to create weights from, while initializing the network.
        The 'w_gen_args' defines the range within which the weights have to be initialized. 
        However, these are only the default values. Internally, the network uses a function to determine the range,
        as discussed in class."""
    model_args = {'w_gen': utils.get_rand_weights_tup, 'w_gen_args': (0, 0.25)}

    """Initializing the Feed-Forward network with the given inputs."""
    feedFwdNwk = FeedForwardNetwork(layers=layers, neurons_in_layers=neurons_in_layers,
                                    model_args=model_args, inputs=features,
                                    eta=0.01, alpha=0.01, weights=weights)

    """Arguments for calculating performance of the model, before training
       These are passed to the model to calculate the performance, before training as a pre-process"""
    performance = {}
    pre_args = {"x_test_set": X_test, "y_test_set": X_test, "model": feedFwdNwk, "performance": performance,
                "when": "before"}

    """ Arguments for Post-Epoch Operations:
        Arguments being set to calculate the error_fraction, after every epoch."""
    error_frac = {}
    epoch_post_args = {"error_frac": error_frac, "x_train": X_train, "y_train": X_train,
                       "x_test": X_test, "y_test": X_test, "model": feedFwdNwk, "epochs_interval": 1}

    """Performs training on the feed-forward neural network."""
    if train:
        print("Training Feed-Forward Neural Network ...")
        print("Initial J2 Loss over Training Set: Before Training")
        weights = feedFwdNwk.train(X_train, X_train, epochs=300,
                                   pre_process=pre_process,
                                   pre_process_args=epoch_post_args,
                                   epoch_pre_process=before_epoch, epoch_pre_process_args={'frac': 0.25,
                                                                                           'x_train': X_train,
                                                                                           'x_index': indexes,
                                                                                           'y_train': X_train,
                                                                                           'y_index': indexes},
                                   epoch_post_process=calculate_error_on_train_test,
                                   epoch_post_process_args=epoch_post_args,
                                   stop_criterion=stop_training_condition,
                                   stop_criterion_args={'min_train_error': 1000, 'min-epochs': 30},
                                   stop_process=calculate_error_on_train_test)
        store_weights(weights, layers)
        utils.print_sep()

        print("Plotting Loss Function of the Network")
        # Plotting loss value of the network, through training as epochs pass
        error_on_training = error_frac.values()

        plt.figure()
        plt.title("Loss Function on training set, through training")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function over training")
        plt.plot(list(error_frac.keys()), error_on_training, label="Loss on training set")
        plt.legend(loc="upper right")
        plt.show()

    utils.print_sep()
    print("Predicting loss over training and testing set:")
    """Predicting on the feed-forward network after training."""
    # Predicting over training set

    # Loss over training set
    lms_sum_over_train = 0
    lms_i_over_train = []
    for i in range(10):
        x_train_label = X_train.iloc[Y_train[Y_train['Label'] == i].index.to_list()]
        y_pred = feedFwdNwk.predict(x_train_label, {})
        lms_loss = lms(y_pred, x_train_label)
        lms_i_over_train.append(lms_loss / x_train_label.count().values[0])
        lms_sum_over_train += lms_loss
    lms_sum_over_train /= X_train.count().values[0]
    print("Average J2 Loss over the training set, after training: ", lms_sum_over_train)

    # Loss over test set
    lms_sum_over_test = 0
    lms_i_over_test = []
    for i in range(10):
        x_test_label = X_test.iloc[Y_test[Y_test['Label'] == i].index.to_list()]
        y_pred = feedFwdNwk.predict(x_test_label, {})
        lms_loss = lms(y_pred, x_test_label)
        lms_i_over_test.append(lms_loss / x_test_label.count().values[0])
        lms_sum_over_test += lms_loss
    lms_sum_over_test /= X_test.count().values[0]
    print("Average J2 Loss over the testing set, after training: ", lms_sum_over_test)
    utils.print_sep()

    print("Plotting average loss on Training and Testing set, after Training")
    labels = ["Average Loss over Training", "Average Loss over Testing"]

    width = 0.2
    x = np.arange(len(labels))
    plt.figure()
    plt.title("Average Loss over Training and Testing sets, after training")
    lms_plot = plt.bar(x, [lms_sum_over_train, lms_sum_over_test], width, label='Loss')

    plt.xticks(ticks=x, labels=labels)
    plt.bar_label(lms_plot, padding=3)
    plt.legend(loc="upper right")
    plt.show()
    utils.print_sep()

    print("Plotting average loss on digits, over Training and Testing set (after Training)")
    # Plotting loss over digits
    labels = [f"Digit {i}" for i in range(10)]

    width = 0.2
    x = np.arange(len(labels))
    plt.figure()
    plt.title("Average Loss on Digits over Training and Testing sets, after training")
    train = plt.bar(x - width, lms_i_over_train, width, label='Train')
    test = plt.bar(x + width, lms_i_over_test, width, label='Test')

    plt.xticks(ticks=x, labels=labels)
    plt.bar_label(train, padding=3)
    plt.bar_label(test, padding=3)
    plt.legend(loc="upper right")
    plt.show()
    utils.print_sep()

    print("Plotting Features of randomly-selected hidden neurons")
    # Plot features of neurons and store there indices for future use
    neurons = None
    if utils.are_sets_available([paths_to_other_files['neurons_plotted']]):
        neurons = pd.read_csv(paths_to_other_files['neurons_plotted'], sep="\\s+", header=None)[0].to_list()

    neurons_plotted = plot_features_of_neurons(feedFwdNwk.get_weights(), 0, neurons_in_layers[0],
                                               20, 4, 5, neurons)
    if neurons is None:
        utils.store_to_csv(pd.DataFrame({'Index of Neuron': neurons_plotted, 'Layer': [0 for i in range(20)]}),
                           paths_to_other_files['neurons_plotted'], sep="\t")
    utils.print_sep()

    print("Plotting Inputs and Outputs of AutoEncoder, for randomly-selected data points from the testing set.")
    # Plot outputs of the network
    no_samples = 8
    x_test_inputs = X_test.iloc[np.random.randint(0, X_test.shape[0], no_samples)]
    x_test_outputs = feedFwdNwk.predict(x_test_inputs, {})
    plot_outputs_of_net(x_test_inputs, x_test_outputs, 2, no_samples)
    utils.print_sep()


