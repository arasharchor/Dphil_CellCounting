# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations, initializations
from ..layers.core import MaskedLayer
import theano
import theano.tensor as T


class Recurrent(MaskedLayer):
    '''Abstract base class for recurrent layers.
    Do not use in a model -- it's not a functional layer!

    All recurrent layers (GRU, LSTM, SimpleRNN) also
    follow the specifications of this class and accept
    the keyword arguments listed below.

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        - if `return_sequences`: 3D tensor with shape
            `(nb_samples, timesteps, output_dim)`.
        - else, 2D tensor with shape `(nb_samples, output_dim)`.

    # Arguments
        weights: list of numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
        **Note:** for the time being, masking is only supported with Theano.

    # TensorFlow warning
        For the time being, when using the TensorFlow backend,
        the number of timesteps used must be specified in your model.
        Make sure to pass an `input_length` int argument to your
        recurrent layer (if it comes first in your model),
        or to pass a complete `input_shape` argument to the first layer
        in your model otherwise.


    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_shape=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    '''
    input_ndim = 3

    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Recurrent, self).__init__(**kwargs)

    def get_output_mask(self, train=False):
        if self.return_sequences:
            return super(Recurrent, self).get_output_mask(train)
        else:
            return None

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def step(self, x, states):
        raise NotImplementedError

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        mask = self.get_input_mask(train)

        assert K.ndim(X) == 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences.\n' +
                                'If your first layer is an Embedding, ' +
                                'make sure to pass it an "input_length" ' +
                                'argument. Otherwise, make sure ' +
                                'the first layer has ' +
                                'an "input_shape" or "batch_input_shape" ' +
                                'argument, including the time axis.')
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        last_output, outputs, states = K.rnn(self.step, X,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful}
        if self.stateful:
            config['batch_input_shape'] = self.input_shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length
            
        base_config = super(Recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(Recurrent):
    '''Fully-connected RNN where the output is to fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        super(SimpleRNN, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        # states only contains the previous output.
        assert len(states) == 1
        prev_output = states[0]
        h = K.dot(x, self.W) + self.b
        output = self.activation(h + K.dot(prev_output, self.U))
        return output, [output]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(Recurrent):
    '''Gated Recurrent Unit - Cho et al. 2014.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.

    # References
        - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(GRU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = K.zeros((self.output_dim,))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = K.zeros((self.output_dim,))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                  self.W_r, self.U_r, self.b_r,
                                  self.W_h, self.U_h, self.b_h]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        assert len(states) == 1
        x_z = K.dot(x, self.W_z) + self.b_z
        x_r = K.dot(x, self.W_r) + self.b_r
        x_h = K.dot(x, self.W_h) + self.b_h

        h_tm1 = states[0]
        z = self.inner_activation(x_z + K.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + K.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + K.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(LSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o))
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiDimRNN(Recurrent):
    '''grid_shape: define the 2D shape of the grids. Eg. there are in total 5*6 grids'''


    def __init__(self, output_dim, grid_shape , 
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):

        self.output_dim = output_dim
        self.grid_shape = tuple(grid_shape)

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(MultiDimRNN, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None for _ in range(np.sum(self.grid_shape))]
            #for state_id in range(np.sum(self.grid_shape)):
            #    self.states.append(K.zeros((input_shape[0], self.output_dim)) )

        input_dim = input_shape[2] #Nsample*timestep*input_dim
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.U_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_col = self.inner_init((self.output_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        self.params = [self.W, self.U_row, self.U_col, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            for state_id in range(np.sum(self.grid_shape)):  
                K.set_value(self.states[state_id],
                            np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = []
            for state_id in range(np.sum(self.grid_shape)):
                self.states.append(K.zeros((input_shape[0], self.output_dim)))

    def step(self, x, row_state, col_state):
        # states only contains the previous output.
        #assert len(states) == 1
        #prev_output = states[0]
        h = K.dot(x, self.W) + self.b
        output = self.activation(h + (K.dot(row_state, self.U_row) + K.dot(col_state, self.U_col))/2 )
        return output, [output]

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
        # get input, then map it to the grid_shape
        inner_data = K.reshape(X, (-1,) + self.grid_shape + (self.input_shape[2],))  # for example 16*16 *900 (16 * 16 grids and each contains 900 pixels)

        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        mask = self.get_output_mask(train)

        if mask:
            # apply mask
            X *= K.cast(K.expand_dims(mask), X.dtype)
            masking = True
        else:
            masking = False

        #grid_mask = K.reshape(mask, (-1,) +  self.grid_shape)

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X) #build an all-zero tensor of shape (samples, output_dim)

        last_output, outputs, states = self.rnn_grid(self.step, inner_data, self.grid_shape, initial_states,
                                             go_backwards=self.go_backwards,
                                             masking=masking)
                                             
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return K.reshape(outputs, (-1,) + (np.prod(self.grid_shape), self.input_shape[2]))
        else:
            return last_output

    def rnn_grid(self ,step_function, inputs, grid_shape, initial_states,
        go_backwards=False, masking=True):
        row_states = initial_states[0:grid_shape[0]]
        col_states = initial_states[grid_shape[0]:]

        inputs = inputs.dimshuffle((2, 1, 0, 3))  # col, row, nsample, dim
        #inputs = input.dimshuffle(tuple(range(1,len(grid_shape)+1))+(0, len(grid_shape)+1) )

        def _step(input, row, row_state, col_state): #[row_states] + [col_state]
            #'''it needs the left, and upper grid as it's predecessor,  
            #   it only update the col_state after process one local patch.
            #   although it needs two states to compute each step, but it actually only updates the upper column state, for the
            #   left row_state, it only use it.
            #'''
            states = [col_state]
            output, new_states = step_function(input, row_state, states[0])  # this function actually should only return one state

            if masking:
                # if all-zero input timestep, return
                # all-zero output and unchanged states
                switch = T.any(input, axis=-1, keepdims=True)
                output = T.switch(switch, output, 0. * output)
                return_states = []
                for state, new_state in zip(states, new_states):
                    return_states.append(T.switch(switch, new_state, state))
                return [output] + return_states
            else:
                return [output] + new_states


        def loop_over_col(coldata, col_ind, col_state, rows_states, rows_model):
            #'''when loops over a column, it needs the left column as row_states,
            #row_states, col_states are already concatenated to be tensors for scan to process before call this function.
            #it return the results as well as this column response as the row_states to next column.
            #it should return the last column_state
            #'''
            results , _ = theano.scan( fn = _step,
                                       sequences = [coldata, rows_model, rows_states],                                     
                                       outputs_info=[None] +  [col_state] 
                                       )
            new_row_states = results[1:] #'''list of tensor of row_size * nsample * out_dim, but we need to modify it to be list'''
            col_vals = results[0]  
           #'''tensor of row_size * nsample * out_dim'''
           #'''the length is the number of states for each single actual step'''
            single_state_num = len(new_row_states) 
            returned_row_states = []
            for timestep_id in range(self.grid_shape[0]):       
                #'''new_row_states is a list of tensor of size row_size * nsample * out_dim
                #   returned states should be [timestep_1_state_1,...,timestep_1_state_n, ...., timestep_m_state_1,...,timestep_m_state_n],
                #   n is the number of required states for each single actual step function, most time it is 1. but CWRNN has 2.
                #   m is the row_size, = grid_shape[0]
                #'''
               for state_id in range(single_state_num):
                    returned_row_states.append(T.squeeze(new_row_states[state_id][timestep_id]))
            returned_col_state = []
            for state_id in range(single_state_num):
              returned_col_state.append(T.squeeze(new_row_states[state_id][-1]))
            #'''output should be list corresponding to the outputs_info'''
            return [col_vals] + [T.stack(returned_row_states, axis = 0)] + [T.stack(returned_col_state, axis = 0)]        

        inputs_model = dict(input = inputs, taps=[0])
        rows_model = T.arange(self.grid_shape[0])
        cols_model = T.arange(self.grid_shape[1])
        #'''loop_over_col(coldata, col_ind, rows_states, col_states, rows_model)'''
        #'''Maybe I need to concatenate all states list to tensor at this stage, scan deal with list of tensor variable, not
        #list of list.'''
        results, _ = theano.scan( fn = loop_over_col,
                                  sequences = [inputs_model, cols_model, T.stack(col_states, axis = 0)],
                                #'''return grid_results, row_states, col_state'''
                                  outputs_info=[None, T.stack(row_states, axis=0), None],
                                  non_sequences = [rows_model] 
            )
        #update col_states here
        #'''results are list of [grid_tensor, row_state_tensor, col_state]'''
        outputs =  results[0]  #'''tensor type'''
        row_states_tensor  = results[1][-1]  #'''we only need the last column as the row_states, tensor type'''
        col_states_tensor  = results[2]  #'''tensor type'''


        # deal with Theano API inconsistency
        #if type(results) is list:
        #    outputs = results[0]
        #    states = results[1:]
        #else:
        #    outputs = results
        #    states = []
        outputs = T.squeeze(outputs)
        last_output = outputs[-1,-1]

        #outputs  col, row, nsample, dim
        outputs = outputs.dimshuffle((2,1,0,3))

        retuned_row_states = []
        retuned_col_states = []

        for state_id in range(len(row_states)):
            state = row_states_tensor[state_id]
            retuned_row_states.append(T.squeeze(state[-1]))
        for state_id in range(len(col_states)):
            state = col_states_tensor[state_id]
            retuned_col_states.append(T.squeeze(state[-1]))

        return last_output, outputs, retuned_row_states + retuned_col_states


    def get_config(self):
        config = {"grid_shape": self.grid_shape,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(MultiDimRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
