import pandas as pd
from model import FeedForwardNetwork, AutoEncoderClassifier
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
                        "confusion_mat_train": './res/data/confusion_matrices/confusion_mat_train.csv',
                        "confusion_mat_test": './res/data/confusion_matrices/confusion_mat_test.csv',
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

    x_train, x_waste, y_train, y_waste = utils.generate_train_test_sets([x, y], x_columns=x_index, y_columns=y_index,
                                                                        axis=1, ignore_index=False, frac=args['frac'])

    model['x_train'] = x_train
    model['y_train'] = y_train
    model['set_size'] = x_train.shape[0]
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


def calculate_error_on_train_test_bt(args):
    return calculate_error_on_train_test(0, args)


def calculate_error_on_train_test(epoch, args):
    global print_error
    model = args
    args = args['epoch_post_process_args']
    epochs_interval = args['epochs_interval']

    error_frac = args['error_frac']

    x_train = model['x_train']
    y_train = model['y_train']
    if 'y_pred' in model:
        y_pred = pd.Series(model['y_pred'][epoch]).apply(lambda y: pd.Series(y[0]))
        y_train_pred = y_pred.apply(lambda y:
                                    y.apply(FeedForwardNetwork.__j2_loss_function__,
                                            args=(args['low'], args['high'])), axis=1)
    else:
        y_pred = args['model'].predict(x_train, {})
        y_train_pred = y_pred.apply(lambda y:
                                    pd.Series([FeedForwardNetwork.__j2_loss_function__(j, args['low'], args['high'])
                                               for j in y[0]]))

    y_train_out = y_train.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
    train_err_frac = calculate_error_frac_train(y_train_pred, y_train_out)

    x_test = args['x_test']
    y_test = args['y_test']

    y_test_pred = __winner_takes_all_output(args['model'].predict(x_test, {}).apply(lambda y: pd.Series(y[0])),
                                            y_test.index)

    y_test_out = y_test['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
    y_test_out = __winner_takes_all_output(y_test_out, y_test_out.index)
    test_err_frac = calculate_error_frac_test(y_test_pred, y_test_out)

    error_frac[epoch] = (train_err_frac, test_err_frac)
    __store_training_error(train_err_frac)

    if epoch == 0 or (epoch + 1) % epochs_interval == 0 or print_error:
        print("Error Fraction on training: ", train_err_frac)
        print("Error Fraction on testing: ", test_err_frac)


def calculate_error_frac_train(y_predicted, y_actual):
    total = y_predicted.shape[0]
    count = 0
    for index in range(total):
        match = True
        for i in range(y_predicted.shape[1]):
            if y_predicted.iloc[index][i] != y_actual.iloc[index][i]:
                match = False
                break
        if not match:
            count += 1
    return count / total


def calculate_error_frac_test(y_predicted, y_actual):
    return y_predicted['Label'][y_predicted['Label'] != y_actual['Label']].count() / y_predicted.shape[0]


def after_epoch(epoch, args):
    network = args['net']
    network.print_weights()


def store_weights(weights: dict, n_layers):
    common_path = paths_to_other_files['weights_of_layer']
    for layer in range(n_layers):
        columns = len(weights[layer][0])
        df = pd.DataFrame(weights[layer], columns=[i for i in range(columns)])
        utils.store_to_csv(df, common_path + f"{layer}.csv", sep='\t', overwrite=True)


if __name__ == '__main__':
    features = 784
    indexes = [i for i in range(features)]

    nn = r""" |\   | ______ |     | |
              | \  | |      |     | |
              |  \ |  ===   |     | |
              |   \| |_____ \____/  |"""

    """Gets or Creates the training and testing datasets from the stored locations."""
    X_train, X_test, Y_train, Y_test = create_or_get_datasets(indexes, ['Label'], train_set_frac=0.8)

    layers = 2
    neurons_in_layers = [50, 10]
    low = 0.25
    high = 0.75
    eta, alpha = 0.1, 0.1

    # Initializing for networks with AutoEncoder
    weights = []
    df = pd.read_csv(paths_to_other_files['weights_of_layer'] + "0.csv", sep='\\s+', header=None)
    weights.append(df.to_numpy())
    weights = np.array(weights)

    """Case I"""
    utils.print_sep()
    print("Case I: Using AutoEncoder as hidden-layer and training the classifier layer")
    utils.print_sep()
    print("Initializing the Feed-Forward Neural Network ...")

    """Arguments for model to create weights from, while initializing the network.
        The 'w_gen_args' defines the range within which the weights have to be initialized. 
        However, these are only the default values. Internally, the network uses a function to determine the range,
        as discussed in class."""
    model_args = {'w_gen': utils.get_rand_weights_tup, 'w_gen_args': (0, 0.25)}

    """Initializing the Feed-Forward network with the given inputs."""
    model_classifier = AutoEncoderClassifier(layers=layers, neurons_in_layers=neurons_in_layers,
                                             model_args=model_args, inputs=features, low=low, high=high,
                                             eta=eta, alpha=alpha, weights=weights)

    """Arguments for calculating performance of the model, before training
       These are passed to the model to calculate the performance, before training as a pre-process"""
    performance = {}
    pre_args = {"x_test_set": X_test, "y_test_set": Y_test, "model": model_classifier, "performance": performance,
                "when": "before"}

    """ Arguments for Post-Epoch Operations:
        Arguments being set to calculate the error_fraction, after every epoch."""
    error_frac = {}
    epoch_post_args = {"error_frac": error_frac, "x_train": X_train, "y_train": Y_train,
                       "x_test": X_test, "y_test": Y_test, "model": model_classifier, "epochs_interval": 10,
                       'low': low, 'high': high}

    print("Training Feed-Forward Neural Network ...")
    utils.print_sep()
    print("Error on Training and Testing Set: Before Training")
    """Performs training on the feed-forward neural network."""
    model_classifier.train(X_train, Y_train, epochs=100,
                           pre_process=calculate_error_on_train_test_bt,
                           pre_process_args=epoch_post_args,
                           epoch_pre_process=before_epoch, epoch_pre_process_args={'frac': 0.5,
                                                                                   'x_train': X_train,
                                                                                   'x_index': indexes,
                                                                                   'y_train': Y_train,
                                                                                   'y_index': ['Label']},
                           epoch_post_process=calculate_error_on_train_test,
                           epoch_post_process_args=epoch_post_args,
                           stop_criterion=stop_training_condition,
                           stop_criterion_args={'min_train_error': 0.01, 'min-epochs': 30},
                           stop_process=calculate_error_on_train_test)
    utils.print_sep()

    print("Plotting Error Fraction of the Network")
    # Plotting Error Fraction of the Network, through training as epochs pass
    error_on_training = []
    error_on_testing = []
    for error in error_frac.values():
        error_on_training.append(error[0])
        error_on_testing.append(error[1])

    utils.plot_graph(title="Case I: Error Fraction on training and testing set, through training",
                     x_lbl="Epochs", y_lbl="Error Fraction",
                     x_data=list(error_frac.keys()),
                     y_data=[error_on_training, error_on_testing],
                     labels=["Error on training set", "Error on testing set"],
                     legend_loc="upper right")

    utils.print_sep()
    # Loss over training set
    error_i_train = []
    for i in range(10):
        y_train = Y_train[Y_train['Label'] == i]
        x_train = X_train.loc[y_train.index.to_list()]
        y_pred = model_classifier.predict(x_train, {}).apply(lambda y: pd.Series(y[0]))
        y_pred = __winner_takes_all_output(y_pred, index_y=y_train.index)
        y_train_out = y_train.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
        y_train_out = __winner_takes_all_output(y_train_out, y_train_out.index)
        error_i = calculate_error_frac_train(y_pred, y_train_out)
        error_i_train.append(error_i)

    # Loss over test set
    error_i_test = []
    for i in range(10):
        y_test = Y_test[Y_test['Label'] == i]
        x_test = X_test.loc[y_test.index.to_list()]
        y_pred = model_classifier.predict(x_test, {}).apply(lambda y: pd.Series(y[0]))
        y_pred = __winner_takes_all_output(y_pred, y_test.index)
        y_test_out = y_test.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
        y_test_out = __winner_takes_all_output(y_test_out, y_test_out.index)
        error_i = calculate_error_frac_train(y_pred, y_test_out)
        error_i_test.append(error_i)
    utils.print_sep()

    y_pred = model_classifier.predict(X_train, {})
    y_pred = y_pred.apply(lambda y: pd.Series([FeedForwardNetwork.__j2_loss_function__(j, low, high)
                                               for j in y[0]]))
    y_train_out = Y_train.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
    error_train = calculate_error_frac_train(y_pred, y_train_out)
    print("Average Error over the training set, after training: ", error_train)

    y_pred = model_classifier.predict(X_test, {})
    y_pred = y_pred.apply(lambda y: pd.Series([FeedForwardNetwork.__j2_loss_function__(j, low, high)
                                               for j in y[0]]))
    y_train_out = Y_test.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
    error_test = calculate_error_frac_train(y_pred, y_train_out)
    print("Average Error over the training set, after training: ", error_test)
    print("Plotting average error on Training and Testing set, after Training")
    import matplotlib.pyplot as plt

    labels = ["Average Error over Training", "Average Error over Testing"]

    width = 0.2
    x = np.arange(len(labels))
    plt.figure()
    plt.title("Average Error over Training and Testing sets, after training")
    lms_plot = plt.bar(x, [error_train, error_test], width, label='Error Fraction')

    plt.xticks(ticks=x, labels=labels)
    plt.bar_label(lms_plot, padding=3)
    plt.legend(loc="upper right")
    plt.show()
    utils.print_sep()

    print("Plotting average error on digits, over Training and Testing set (after Training)")
    # Plotting loss over digits
    labels = [f"Digit {i}" for i in range(10)]

    width = 0.2
    x = np.arange(len(labels))
    plt.figure()
    plt.title("Average Error on Digits over Training and Testing sets, after training")
    train = plt.bar(x - width, error_i_train, width, label='Train')
    test = plt.bar(x + width, error_i_test, width, label='Test')

    plt.xticks(ticks=x, labels=labels)
    plt.bar_label(train, padding=3)
    plt.bar_label(test, padding=3)
    plt.legend(loc="upper right")
    plt.show()
    utils.print_sep()

    print("Predicting over training and testing set:")
    """Predicting on the feed-forward network after training."""
    # Predicting over training set
    y_pred = model_classifier.predict(X_train, None).apply(lambda y: pd.Series(y[0]))
    y_pred_train_results = __winner_takes_all_output(y_pred, index_y=Y_train.index)
    # print(y_pred_train_results)
    confusion_mat_train = utils.create_confusion_matrix(Y_train, y_pred_train_results, 'Label')

    # Predicting over testing set
    y_pred = model_classifier.predict(X_test, None).apply(lambda y: pd.Series(y[0]))
    y_pred_test_results = __winner_takes_all_output(y_pred, index_y=Y_test.index)
    # print(y_pred_test_results)
    confusion_mat_test = utils.create_confusion_matrix(Y_test, y_pred_test_results, 'Label')

    utils.print_sep()
    print("Confusion Matrix on Training Set: ")
    print(confusion_mat_train)
    utils.store_to_csv(confusion_mat_train, paths_to_other_files['confusion_mat_train'], sep="\t")

    utils.print_sep()
    print("Confusion Matrix on Testing Set: ")
    print(confusion_mat_test)
    utils.store_to_csv(confusion_mat_train, paths_to_other_files['confusion_mat_test'], sep="\t")

    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(50, 50))
    fig.suptitle("Case I: Confusion Matrix for Training Set")
    ConfusionMatrixDisplay.from_predictions(Y_train, y_pred_train_results, ax=ax)

    fig, ax = plt.subplots(figsize=(50, 50))
    fig.suptitle("Case I: Confusion Matrix for Testing Set")
    ConfusionMatrixDisplay.from_predictions(Y_test, y_pred_test_results, ax=ax)
    plt.show()

    """Case II"""
    utils.print_sep()
    print("Case II: Using AutoEncoder as hidden-layer and training the entire network")
    utils.print_sep()
    print("Initializing the Feed-Forward Neural Network ...")

    """Arguments for model to create weights from, while initializing the network.
        The 'w_gen_args' defines the range within which the weights have to be initialized. 
        However, these are only the default values. Internally, the network uses a function to determine the range,
        as discussed in class."""
    model_args = {'w_gen': utils.get_rand_weights_tup, 'w_gen_args': (0, 0.25)}

    """Initializing the Feed-Forward network with the given inputs."""
    model_classifier = FeedForwardNetwork(layers=layers, neurons_in_layers=neurons_in_layers,
                                          model_args=model_args, inputs=features, low=low, high=high,
                                          eta=eta, alpha=alpha, weights=weights)

    """Arguments for calculating performance of the model, before training
       These are passed to the model to calculate the performance, before training as a pre-process"""
    performance = {}
    pre_args = {"x_test_set": X_test, "y_test_set": Y_test, "model": model_classifier, "performance": performance,
                "when": "before"}

    """ Arguments for Post-Epoch Operations:
        Arguments being set to calculate the error_fraction, after every epoch."""
    error_frac = {}
    epoch_post_args = {"error_frac": error_frac, "x_train": X_train, "y_train": Y_train,
                       "x_test": X_test, "y_test": Y_test, "model": model_classifier, "epochs_interval": 10,
                       'low': low, 'high': high}

    print("Training Feed-Forward Neural Network ...")
    utils.print_sep()
    print("Error on Training and Testing Set: Before Training")
    """Performs training on the feed-forward neural network."""
    model_classifier.train(X_train, Y_train, epochs=100,
                           pre_process=calculate_error_on_train_test_bt,
                           pre_process_args=epoch_post_args,
                           epoch_pre_process=before_epoch, epoch_pre_process_args={'frac': 0.5,
                                                                                   'x_train': X_train,
                                                                                   'x_index': indexes,
                                                                                   'y_train': Y_train,
                                                                                   'y_index': ['Label']},
                           epoch_post_process=calculate_error_on_train_test,
                           epoch_post_process_args=epoch_post_args,
                           stop_criterion=stop_training_condition,
                           stop_criterion_args={'min_train_error': 0.01, 'min-epochs': 30},
                           stop_process=calculate_error_on_train_test)
    utils.print_sep()

    print("Plotting Error Fraction of the Network")
    # Plotting Error Fraction of the Network, through training as epochs pass
    error_on_training = []
    error_on_testing = []
    for error in error_frac.values():
        error_on_training.append(error[0])
        error_on_testing.append(error[1])

    utils.plot_graph(title="Case II: Error Fraction on training and testing set, through training",
                     x_lbl="Epochs", y_lbl="Error Fraction",
                     x_data=list(error_frac.keys()),
                     y_data=[error_on_training, error_on_testing],
                     labels=["Error on training set", "Error on testing set"],
                     legend_loc="upper right")

    utils.print_sep()
    # Loss over training set
    error_i_train = []
    for i in range(10):
        y_train = Y_train[Y_train['Label'] == i]
        x_train = X_train.loc[y_train.index.to_list()]
        y_pred = model_classifier.predict(x_train, {}).apply(lambda y: pd.Series(y[0]))
        y_pred = __winner_takes_all_output(y_pred, index_y=y_train.index)
        y_train_out = y_train.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
        y_train_out = __winner_takes_all_output(y_train_out, y_train_out.index)
        error_i = calculate_error_frac_train(y_pred, y_train_out)
        error_i_train.append(error_i)

    # Loss over test set
    error_i_test = []
    for i in range(10):
        y_test = Y_test[Y_test['Label'] == i]
        x_test = X_test.loc[y_test.index.to_list()]
        y_pred = model_classifier.predict(x_test, {}).apply(lambda y: pd.Series(y[0]))
        y_pred = __winner_takes_all_output(y_pred, y_test.index)
        y_test_out = y_test.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
        y_test_out = __winner_takes_all_output(y_test_out, y_test_out.index)
        error_i = calculate_error_frac_train(y_pred, y_test_out)
        error_i_test.append(error_i)
    utils.print_sep()

    y_pred = model_classifier.predict(X_train, {})
    y_pred = y_pred.apply(lambda y: pd.Series([FeedForwardNetwork.__j2_loss_function__(j, low, high)
                                               for j in y[0]]))
    y_train_out = Y_train.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
    error_train = calculate_error_frac_train(y_pred, y_train_out)
    print("Average Error over the training set, after training: ", error_train)

    y_pred = model_classifier.predict(X_test, {})
    y_pred = y_pred.apply(lambda y: pd.Series([FeedForwardNetwork.__j2_loss_function__(j, low, high)
                                               for j in y[0]]))
    y_train_out = Y_test.copy()['Label'].apply(FeedForwardNetwork.__convert_label_to_result__)
    error_test = calculate_error_frac_train(y_pred, y_train_out)
    print("Average Error over the testing set, after training: ", error_test)
    print("Plotting average error on Training and Testing set, after Training")
    import matplotlib.pyplot as plt

    labels = ["Average Error over Training", "Average Error over Testing"]

    width = 0.2
    x = np.arange(len(labels))
    plt.figure()
    plt.title("Average Error over Training and Testing sets, after training")
    lms_plot = plt.bar(x, [error_train, error_test], width, label='Error Fraction')

    plt.xticks(ticks=x, labels=labels)
    plt.bar_label(lms_plot, padding=3)
    plt.legend(loc="upper center")
    plt.show()
    utils.print_sep()

    print("Plotting average error on digits, over Training and Testing set (after Training)")
    # Plotting loss over digits
    labels = [f"Digit {i}" for i in range(10)]

    width = 0.2
    x = np.arange(len(labels))
    plt.figure()
    plt.title("Average Error on Digits over Training and Testing sets, after training")
    train = plt.bar(x - width, error_i_train, width, label='Train')
    test = plt.bar(x + width, error_i_test, width, label='Test')

    plt.xticks(ticks=x, labels=labels)
    plt.bar_label(train, padding=3)
    plt.bar_label(test, padding=3)
    plt.legend(loc="upper center")
    plt.show()
    utils.print_sep()

    print("Predicting over training and testing set:")
    """Predicting on the feed-forward network after training."""
    # Predicting over training set
    y_pred = model_classifier.predict(X_train, None).apply(lambda y: pd.Series(y[0]))
    y_pred_train_results = __winner_takes_all_output(y_pred, index_y=Y_train.index)
    # print(y_pred_train_results)
    confusion_mat_train = utils.create_confusion_matrix(Y_train, y_pred_train_results, 'Label')

    # Predicting over testing set
    y_pred = model_classifier.predict(X_test, None).apply(lambda y: pd.Series(y[0]))
    y_pred_test_results = __winner_takes_all_output(y_pred, index_y=Y_test.index)
    # print(y_pred_test_results)
    confusion_mat_test = utils.create_confusion_matrix(Y_test, y_pred_test_results, 'Label')

    utils.print_sep()
    print("Confusion Matrix on Training Set: ")
    print(confusion_mat_train)
    utils.store_to_csv(confusion_mat_train, paths_to_other_files['confusion_mat_train'], sep="\t")

    utils.print_sep()
    print("Confusion Matrix on Testing Set: ")
    print(confusion_mat_test)
    utils.store_to_csv(confusion_mat_train, paths_to_other_files['confusion_mat_test'], sep="\t")

    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(50, 50))
    fig.suptitle("Case II: Confusion Matrix for Training Set")
    ConfusionMatrixDisplay.from_predictions(Y_train, y_pred_train_results, ax=ax)

    fig, ax = plt.subplots(figsize=(50, 50))
    fig.suptitle("Case II: Confusion Matrix for Testing Set")
    ConfusionMatrixDisplay.from_predictions(Y_test, y_pred_test_results, ax=ax)
    plt.show()
