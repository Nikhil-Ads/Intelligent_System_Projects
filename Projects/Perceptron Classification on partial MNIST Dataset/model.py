import numpy as np
import pandas as pd


def parse_kwargs(x, y, kwargs):
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 10
    set_size = kwargs['set_size'] if 'set_size' in kwargs else x.shape[0]
    eta = kwargs['eta'] if 'eta' in kwargs else 0.01

    train_args = kwargs['train_args'] if 'train_args' in kwargs else {}
    update_w_args = kwargs['update_w_args'] if 'update_w_args' in kwargs else {'eta': eta}

    pre_process = kwargs['pre_process'] if 'pre_process' in kwargs else None
    pre_process_args = kwargs['pre_process_args'] if 'pre_process_args' in kwargs else {}

    post_process = kwargs['post_process'] if 'post_process' in kwargs else None
    post_process_args = kwargs['post_process_args'] if 'post_process_args' in kwargs else {}

    epoch_pre_process = kwargs['epoch_pre_process'] if 'epoch_pre_process' in kwargs else None
    epoch_pre_process_args = kwargs['epoch_pre_process_args'] if 'epoch_pre_process_args' in kwargs else {}

    epoch_post_process = kwargs['epoch_post_process'] if 'epoch_post_process' in kwargs else None
    epoch_post_process_args = kwargs['epoch_post_process_args'] if 'epoch_post_process_args' in kwargs else {}

    return epochs, set_size, train_args, update_w_args, \
           pre_process, pre_process_args, epoch_pre_process, epoch_pre_process_args, \
           epoch_post_process, epoch_post_process_args, post_process, post_process_args


class Classifier:
    """ The base class for any Classifier. It is a generic class and provides the general methods to train and
    predict on the model.
    """

    def __train__(self, x, y, w, args):
        """[Internal Method]
            Provides implementation to train the model. Each classifier must override this function to provide it's
            own training way."""
        return y

    def train(self, x, y, w, **kwargs):
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
        epochs, set_size, train_args, update_w_args, \
        pre_process, pre_process_args, epoch_pre_process, epoch_pre_process_args, \
        epoch_post_process, epoch_post_process_args, post_process, post_process_args = parse_kwargs(x, y, kwargs)

        if pre_process:
            pre_process(w, pre_process_args)

        for epoch in range(epochs):
            if epoch_pre_process:
                epoch_pre_process(epoch, w, epoch_pre_process_args)

            for sample in range(set_size):
                x_i = x.iloc[sample]
                y_i = y.iloc[sample]['Label']

                results = self.__train__(x_i, y_i, w, train_args)
                w = self.__update_weights__(x_i, y_i, w, results, update_w_args)

            if epoch_post_process:
                epoch_post_process(epoch, w, epoch_post_process_args)

        if post_process:
            post_process(w, post_process_args)

        return w

    def predict(self, x, w, args):
        pass

    def __update_weights__(self, x, y, w, results, args):
        return w


class Perceptron(Classifier):

    def __train__(self, x, y, w, args):
        s_t = np.sum(x * w)
        if s_t > 0:
            return 1
        else:
            return 0

    def predict(self, x, w, args):
        s_t = pd.DataFrame(np.sum(x * w, axis=1), columns=['Label'], index=x.index)
        s_t['Label'] = s_t['Label'].apply(lambda y: 1 if y > 0 else 0)
        return s_t

    def __update_weights__(self, x, y, w, results, args):
        eta = args['eta']
        w += eta * (y - results) * x
        return w
