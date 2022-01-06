import numpy
import numpy as np
import pandas as pd


def parse_kwargs(x, y, kwargs):
    model = {'epochs': kwargs['epochs'] if 'epochs' in kwargs else 10,
             'set_size': kwargs['set_size'] if 'set_size' in kwargs else x.shape[0],
             'eta': kwargs['eta'] if 'eta' in kwargs else 0.01,
             'train_args': kwargs['train_args'] if 'train_args' in kwargs else {},
             'pre_process': kwargs['pre_process'] if 'pre_process' in kwargs else None,
             'pre_process_args': kwargs['pre_process_args'] if 'pre_process_args' in kwargs else {},
             'post_process': kwargs['post_process'] if 'post_process' in kwargs else None,
             'post_process_args': kwargs['post_process_args'] if 'post_process_args' in kwargs else {},
             'epoch_pre_process': kwargs['epoch_pre_process'] if 'epoch_pre_process' in kwargs else None,
             'epoch_pre_process_args': kwargs['epoch_pre_process_args'] if 'epoch_pre_process_args' in kwargs else {},
             'epoch_post_process': kwargs['epoch_post_process'] if 'epoch_post_process' in kwargs else None,
             'epoch_post_process_args': kwargs[
                 'epoch_post_process_args'] if 'epoch_post_process_args' in kwargs else {},
             'stop_criterion': kwargs['stop_criterion'] if 'stop_criterion' in kwargs else None,
             'stop_criterion_args': kwargs['stop_criterion_args'] if 'stop_criterion_args' in kwargs else {},
             'stop_process': kwargs['stop_process'] if 'stop_process' in kwargs else None,
             'stop_process_args': kwargs['stop_process_args'] if 'stop_process_args' in kwargs else {}
             }

    model['update_w_args'] = kwargs['update_w_args'] if 'update_w_args' in kwargs else {'eta': model['eta']}
    return model


def store__args(model, args):
    if args:
        for arg in args:
            model[arg] = args[arg]


class Classifier:
    """ The base class for any Classifier. It is a generic class and provides the general methods to train and
    predict on the model.
    """
    def __train__(self, x, y, args):
        """[Internal Method]
            Provides implementation to train the model. Each classifier must override this function to provide it's
            own training way."""
        return y

    def train(self, x, y, **kwargs):
        """[External Method]
            The generic method to train the model. It takes x-training set<Input>, y-training set<Output>, and
            set of weights to be used to train the model. This method will generally go through the given number of
            epochs and call the __train__ and __update_weights method in each epoch, to train the method.

            Additionally, this method offers execution points where additional pieces of code can be executed, while
            training. It supports the following execution points for further complex operations:

            1. Pre-process: Before epochs start executing.
            2. Pre-process-in-epoch: Executed once for each epoch, before training in that epoch.
            3. Post-process-in-epoch: Executed once for each epoch, after training in that epoch.
            4. Post-process: After training is completed.

            Users of this method can utilize these execution points to perform or record certain results available at
            that point of execution. Users must provide their own implementations for these methods and may provide
            their own arguments to be passed. These can be set as follows:

            1. epochs       - sets the number of epochs,
            2. set_size     - sets the size of the training set,
            3. train_args   - arguments to be passed to your classifier implementation of train method
                              as additional values,
            4. update_w_args- arguments to be passed to your classifier implementation of update_weights method
                              as additional values,
            5. pre_process  - can be used to set the function to be called pre-processing, before training.
                              It is passed [weights, pre_process_args] as fixed arguments,
            6. pre_process_args - used for setting arguments to be as additional arguments to your implementation of
                              pre-processing function,
            7. epoch_pre_process- can be used to set the function to be called for pre-processing in every epoch,
            epoch_pre_process_args,
            epoch_post_process,
            epoch_post_process_args,
            post_process,
            post_process_args """
        train_model = {'x': x, 'y': y, 'x_train': x, 'y_train': y}
        parsed_args = parse_kwargs(x, y, kwargs)
        store__args(train_model, parsed_args)

        if train_model['pre_process']:
            store__args(train_model, train_model['pre_process'](train_model))

        train_model['y_pred'] = {}
        train_model['weights'] = {}

        for epoch in range(train_model['epochs']):
            train_model['y_pred'][epoch] = []

            if train_model['epoch_pre_process']:
                store__args(train_model, train_model['epoch_pre_process'](epoch, train_model['epoch_pre_process_args']))

            for sample in range(train_model['set_size']):
                x_i = train_model['x_train'].iloc[sample]
                y_i = train_model['y_train'].iloc[sample]['Label']

                results = self.__train__(x_i, y_i, train_model['train_args'])
                train_model['y_pred'][epoch].append(results)
                w = self.__update_weights__(x_i, y_i, results, train_model['update_w_args'])

            train_model['weights'][epoch] = w

            if train_model['epoch_post_process']:
                store__args(train_model,
                            train_model['epoch_post_process'](epoch, train_model))

            if train_model['stop_criterion']:
                if train_model['stop_criterion'](epoch, train_model['stop_criterion_args']):
                    if train_model['stop_process']:
                        store__args(train_model,
                                    train_model['stop_process'](epoch, train_model))
                    break

        if train_model['post_process']:
            store__args(train_model, train_model['post_process'](train_model['post_process_args']))

        return w

    def predict(self, x, args):
        pass

    def __update_weights__(self, x, y, results, args):
        pass


class Perceptron(Classifier):

    @staticmethod
    def __activation_function__(x, w):
        sig = 1 / (1 + np.exp(- np.dot(x, w)))
        return sig

    @staticmethod
    def __deriv_activation_function__(x, w):
        f_u = Perceptron.__activation_function__(x, w)
        return f_u * (1 - f_u)

    @staticmethod
    def __get_weights_range(args):
        a = np.sqrt(3 / args[2])
        return -a, a, args[2]

    def __init__(self, index, eta, alpha, args):
        self.index = index
        self.layer = args['layer']
        self.max_layer = args['max_layers']
        self.eta = eta
        self.alpha = alpha
        if 'weights' in args:
            self.w = args['weights']
        else:
            self.w = args['w_gen'](Perceptron.__get_weights_range(args['w_gen_args']))
        self.prev_delta = 0
        self.act_func = args['act_func'] if 'act_func' in args else Perceptron.__activation_function__
        self.deriv_act_func = args[
            'deriv_act_func'] if 'deriv_act_func' in args else Perceptron.__deriv_activation_function__
        self.input = None
        self.weighted_delta = []

    def __train__(self, x, y, args):
        self.input = x
        s_t = self.act_func(x, self.w)
        return s_t

    def predict(self, x, args):
        return Perceptron.__activation_function__(x, self.w)

    def __update_weights__(self, x=None, y=None, results=None, args=None):
        delta = args['delta']
        delta_w = (self.eta * delta * self.input)
        new_delta = delta_w + (self.alpha * self.prev_delta)
        self.w += new_delta
        self.prev_delta = new_delta
        return self.w

    def calc_delta_func(self, args):
        delta = 0
        if args['is_hidden_layer']:
            if 'w_i_j' in args:
                w_i_j = args['w_i_j']
            else:
                raise Exception('"Weights" of the upper layer from this neuron must be provided.')

            sum_of_deltas = np.sum(w_i_j)
            if sum_of_deltas != 0:
                delta = sum_of_deltas * self.deriv_act_func(self.input, self.w)
        else:
            actual = args['actual']
            predict = args['predicted']
            delta = (actual - predict) * self.deriv_act_func(self.input, self.w)
        self.set_weighted_deltas(delta)
        return delta

    def get_neuron_index(self):
        return self.index

    def get_input(self):
        return self.input

    def get_weights(self):
        return self.w

    def get_weight_for_neuron(self, index):
        return self.w[index]

    def set_weighted_deltas(self, delta):
        self.weighted_delta = self.w * delta

    def get_weighted_delta(self, index):
        return self.weighted_delta[index]


class FeedForwardNetwork(Classifier):
    """A generic model to work with Feed-Forward networks."""

    def __init__(self, layers=2, neurons_in_layers=None, inputs=None,
                 eta=0.02, alpha=0.01,
                 act_func=None, deriv_act_func=None,
                 low=0, high=0,
                 model_for_neurons=Perceptron, model_args=None,
                 neurons=None, weights=None):
        """Creates a generic (single-type) Feed-forward network, with the given number of layers and the given number of
        neurons on each layer. These can be specified by using the parameters:

        layers: [int] - specifies the number of layers in the feed-forward network
        neurons_in_layers: [list] - Specifies the number of neurons in each layer

        Each layer of the network will have the same type of neurons, whose model can be specified and arguments
        required for instantiating the model can be passed. Additionally, it will pass the index of the neuron (in that
        layer) to your model. Thus, it is necessary for your model to accept the given arguments, i.e., it must have at
        least two arguments. These can be specified as:

        model_for_neurons: [Any] - specifies the class of the model, you want to have your neurons of.
        model_args: [Any] - specifies the arguments that your model needs to instantiate.

        Alternatively, you may pass your own list-of-lists, containing neurons of different models corresponding to each
        layer of the neural network."""
        act_func_for_layers = None
        deriv_act_func_for_layers = None
        act_func_for_neurons = None
        deriv_act_func_for_neurons = None
        if neurons_in_layers is None:
            neurons_in_layers = [1, 1]
        self.layers = layers
        self.low = low
        self.high = high
        if not neurons:
            if inputs:
                neurons = []
                model_args['max_layers'] = layers - 1
                if act_func:
                    if 'function' in str(type(act_func)):
                        model_args['act_func'] = act_func
                        model_args['deriv_act_func'] = deriv_act_func
                    elif type(act_func) == list:
                        if len(act_func) == layers:
                            if 'function' in str(type(act_func[0])):
                                for func in act_func:
                                    if not callable(func):
                                        raise Exception('"Activation Function" passed is not callable')
                                act_func_for_layers = act_func
                                deriv_act_func_for_layers = deriv_act_func
                            else:
                                for func_layer in act_func:
                                    for func in func_layer:
                                        if not callable(func):
                                            raise Exception('"Activation Function" passed is not callable')
                                act_func_for_neurons = act_func
                                deriv_act_func_for_neurons = deriv_act_func
                for neuron_layer, layer in zip(neurons_in_layers, range(len(neurons_in_layers))):
                    neuron_in_layer = []
                    args_for_layer = dict(model_args)
                    args = list(args_for_layer['w_gen_args'])
                    args.append(inputs)
                    args = tuple(args)
                    args_for_layer['w_gen_args'] = args
                    args_for_layer['layer'] = layer
                    if act_func_for_layers:
                        args_for_layer['act_func'] = act_func_for_layers[layer]
                        args_for_layer['deriv_act_func'] = deriv_act_func
                    for neuron in range(neuron_layer):
                        if act_func_for_neurons:
                            args_for_layer['act_func'] = act_func_for_neurons[layer][neuron]
                            args_for_layer['deriv_act_func'] = deriv_act_func
                        if weights is not None and len(weights) > layer:
                            args_for_layer['weights'] = weights[layer][neuron]
                        neuron_in_layer.append(model_for_neurons(neuron, eta, alpha, args_for_layer))
                    inputs = neuron_layer
                    neurons.append(neuron_in_layer)
            else:
                raise Exception('Inputs in Input_layer must be specified, to initialize weights of the first layer')
        self.neurons = pd.Series([pd.Series(neurons_in_layer) for neurons_in_layer in neurons],
                                 index=[i for i in range(1, layers + 1)])

    @staticmethod
    def __j2_loss_function__(result, low, high):
        if result <= low:
            return 0
        elif result >= high:
            return 1
        else:
            return result

    def __update_weights__(self, x, y, results, args):
        y = FeedForwardNetwork.__convert_label_to_result__(y, self.neurons[self.layers].shape[0]).to_list()
        results = results.apply(FeedForwardNetwork.__j2_loss_function__, args=(self.low, self.high))

        # Calculating deltas
        # Calculating delta for output layer
        deltas = []
        new_deltas = []
        for index, neuron in zip(range(self.neurons[self.layers].shape[0]), self.neurons[self.layers]):
            delta = neuron.calc_delta_func({"is_hidden_layer": False, 'actual': y[index], 'predicted': results[index]})
            new_deltas.append(delta)

        deltas.append(new_deltas)
        # For hidden layers
        prev_deltas = np.array(new_deltas)
        for neuron_layers, upper_layer in zip(self.neurons[-2::-1], range(self.layers, 1, -1)):
            new_deltas = []
            for neuron in neuron_layers:
                weights = np.array([self.neurons[upper_layer][i].get_weighted_delta(neuron.index)
                                    for i in range(self.neurons[upper_layer].shape[0])])
                args = {"is_hidden_layer": True, 'w_i_j': weights}
                delta = neuron.calc_delta_func(args)
                new_deltas.append(delta)
            prev_deltas = new_deltas
            deltas.append(new_deltas)
        deltas.reverse()

        # Updating weights
        weights = {}
        for layer in range(self.layers-1, -1, -1):
            weights[layer] = []
            for index, neuron in zip(range(self.neurons[layer+1].shape[0]), self.neurons[layer+1]):
                weights[layer].append(neuron.__update_weights__(args={'delta': deltas[layer][index]}))

        return weights

    @staticmethod
    def __get_index_of_highest_output__(result: pd.Series):
        return result[result == max(result.to_list())].index[0]

    @staticmethod
    def __convert_result_to_label__(result: pd.Series):
        result = result.to_list()
        for i in range(len(result)):
            if result[i] == 1:
                return i
        return None

    @staticmethod
    def __convert_label_to_result__(label: numpy.int64, len_of_res=10):
        return pd.Series([1 if i == label else 0 for i in range(len_of_res)])

    @staticmethod
    def __neuron_predict__(neuron: Classifier, input_layer, args):
        return neuron.predict(input_layer, args)

    def __predict__(self, input_layer, args=None):
        output = None
        for layer in self.neurons:
            output = layer.apply(FeedForwardNetwork.__neuron_predict__, args=(input_layer, args))
            input_layer = output
        return output

    @staticmethod
    def __convert_output_to_binary__(row):
        max_out = row.max()
        return row.apply(lambda val: 1 if val == max_out else 0)

    def predict(self, x, args):
        output = None
        axis = 1
        if not args:
            args = tuple()
        if type(x) == pd.Series:
            x = x.to_frame()
            axis = 0
        output = x.apply(self.__predict__, axis=axis, args=args)
        return output

    def __train__(self, x, y, args):
        if not args:
            args = dict()
        if type(x) == pd.Series:
            return self.__train_layers__(x, args)
        elif type(x) == pd.DataFrame:
            return x.apply(self.__train_layers__, args, axis=1)

    def __train_layers__(self, input_layer, args):
        output = None
        for layer in self.neurons:
            output = layer.apply(FeedForwardNetwork.__train_neuron__, args=(input_layer, args))
            input_layer = output
        return output

    @staticmethod
    def __train_neuron__(neuron: Classifier, input_layer, args):
        return neuron.__train__(input_layer.to_numpy(), None, args)

    def print_weights(self):
        print(f"Feed Forward Network with layers: {self.layers}")
        for layer, neuron_layer in enumerate(self.neurons):
            print(f"\nLayer {layer}:")
            for index, neuron in enumerate(neuron_layer):
                print(f"Weights of the neuron at index {index}: {neuron.get_weights()}")

    def __str__(self):
        print_str = ''
        print_str += f"Feed Forward Network with {self.layers} layer(s)" + "\n"
        print_str += f'Neurons in layers:' + '\n'
        for layer, index in zip(self.neurons, range(len(self.neurons))):
            print_str += f"\tNeurons in layer {index + 1}: {len(layer)}" + '\n'
            for neuron in layer:
                print_str += f'\t\tNeuron Type: {type(neuron)}, Neuron: {neuron}\n'
        return print_str

    def get_weights(self):
        return [[neurons_in_layer.w[:-1] for neurons_in_layer in layer] for layer in self.neurons]


class AutoEncoderClassifier(FeedForwardNetwork):
    """A specific model to work with AutoEncoder as hidden-layers and a classifier as output layer
    in Feed-Forward networks."""

    def __update_weights__(self, x, y, results, args):
        y = FeedForwardNetwork.__convert_label_to_result__(y, self.neurons[self.layers].shape[0]).to_list()
        results = results.apply(FeedForwardNetwork.__j2_loss_function__, args=(self.low, self.high))

        # Calculating deltas
        # Calculating delta for output layer
        deltas = []
        for index, neuron in zip(range(self.neurons[self.layers].shape[0]), self.neurons[self.layers]):
            delta = neuron.calc_delta_func({"is_hidden_layer": False, 'actual': y[index], 'predicted': results[index]})
            deltas.append(delta)

        # Updating weights
        weights = {}
        layer = 1
        weights[layer] = []
        for index, neuron in zip(range(self.neurons[layer+1].shape[0]), self.neurons[layer+1]):
            weights[layer].append(neuron.__update_weights__(args={'delta': deltas[index]}))
        weights[0] = [neuron.get_weights() for neuron in self.neurons[1]]
        return weights
