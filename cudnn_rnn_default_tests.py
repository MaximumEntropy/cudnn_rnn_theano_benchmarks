"""CUDNN RNN Test."""
from __future__ import absolute_import, print_function, division
import logging

from nose.plugins.skip import SkipTest
from nose_parameterized import parameterized
import numpy
from itertools import product, chain

import theano
from six import StringIO
import theano.tensor as T
import theano.tests.unittest_tools as utt
from theano.tensor.signal.pool import pool_2d, pool_3d
from theano.tensor.signal.pool import Pool, MaxPoolGrad, AveragePoolGrad
from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor

mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray').excluding('gpu')
mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpuarray')

class Model(object):
    def __init__(self, name=""):
        self.name = name
        self.layers = []
        self.params = []
        self.other_updates = {}

    def add_layer(self, layer):
        self.layers.append(layer)
        for p in layer.params:
            self.params.append(p)

        if hasattr(layer, 'other_updates'):
            for y in layer.other_updates:
                self.other_updates[y[0]] = y[1]

    def get_params(self):
        return self.params


def uniform(stdev, size):
    """uniform distribution with the given stdev and size"""
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype(theano.config.floatX)


def linear_transform_weights(input_dim, output_dim,
                             param_list=None, name=""):
    "theano shared variable given input and output dimension"
    weight_inialization = uniform(numpy.sqrt(2.0 / input_dim),
                                  (input_dim, output_dim))
    W = theano.shared(weight_inialization, name=name)

    assert(param_list is not None)

    param_list.append(W)
    return W


def bias_weights(length, param_list=None, name=""):
    "theano shared variable for bias unit, given length"
    bias_initialization = numpy.zeros(length).astype(theano.config.floatX)

    bias = theano.shared(
        bias_initialization,
        name=name
        )

    if param_list is not None:
        param_list.append(bias)

    return bias

class Layer(object):
    '''Generic Layer Template which all layers should inherit'''
    def __init__(self, name=""):
        self.name = name
        self.params = []

    def get_params(self):
        return self.params

class WrapperLayer(Layer):
    def __init__(self, X, name=""):
        self.params = []
        self.name = name
        self.X = X

    def output(self):
        return self.X

class LSTM(Layer):
    def __init__(self, input_dim, output_dim, input_layer, s0=None, c0=None,
                 name=""):
        '''Layers information'''
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.input_layer = input_layer
        self.X = input_layer.output()
        self.s0 = s0
        self.c0 = c0
        self.params = []

        '''Layers weights'''

        '''self.params is passed so that any paramters could be appended to it'''
        self.W_i = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_i")
        self.b_wi = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wi")

        self.W_f = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_f")
        self.b_wf = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wf")

        self.W_c = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_c")
        self.b_wc = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wc")

        self.W_o = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_o")
        self.b_wo = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wo")

        self.R_i = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_i")
        self.b_ri = bias_weights((output_dim,), param_list=self.params, name=name + ".b_ri")

        self.R_f = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_f")
        self.b_rf = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rf")

        self.R_c = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_c")
        self.b_rc = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rc")

        self.R_o = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_o")
        self.b_ro = bias_weights((output_dim,), param_list=self.params, name=name + ".b_ro")

        '''step through processed input to create output'''
        def step(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_i) + T.dot(h_tm1, self.R_i) + self.b_wi + self.b_ri)
            f_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_f) + T.dot(h_tm1, self.R_f) + self.b_wf + self.b_rf)
            o_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_o) + T.dot(h_tm1, self.R_o) + self.b_ro + self.b_wo)

            c_hat_t = T.tanh(
                T.dot(x_t, self.W_c) + T.dot(h_tm1, self.R_c) + self.b_wc + self.b_rc)
            c_t = f_t * c_tm1 + i_t * c_hat_t
            h_t = o_t * T.tanh(c_t)

            return h_t, c_t

        outputs_info = [self.s0, self.c0]

        states, updates = theano.scan(
            fn=step,
            sequences=[self.X],
            outputs_info=outputs_info
            )

        self.Y = states[0]
        self.C = states[1]

    def output(self):
        return self.Y


class FastLSTM(Layer):
    def __init__(self, input_dim, output_dim, input_layer, s0=None, c0=None,
                 name=""):
        '''Layers information'''
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.input_layer = input_layer
        self.X = input_layer.output()
        self.s0 = s0
        self.c0 = c0
        self.params = []

        '''Layers weights'''

        '''self.params is passed so that any paramters could be appended to it'''
        self.W_i = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_i")
        self.b_wi = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wi")

        self.W_f = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_f")
        self.b_wf = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wf")

        self.W_c = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_c")
        self.b_wc = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wc")

        self.W_o = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_o")
        self.b_wo = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wo")

        self.R_i = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_i")
        self.b_ri = bias_weights((output_dim,), param_list=self.params, name=name + ".b_ri")

        self.R_f = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_f")
        self.b_rf = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rf")

        self.R_c = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_c")
        self.b_rc = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rc")

        self.R_o = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_o")
        self.b_ro = bias_weights((output_dim,), param_list=self.params, name=name + ".b_ro")

        '''step through processed input to create output'''
        def step(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_i) + T.dot(h_tm1, self.R_i) + self.b_wi + self.b_ri)
            f_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_f) + T.dot(h_tm1, self.R_f) + self.b_wf + self.b_rf)
            o_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_o) + T.dot(h_tm1, self.R_o) + self.b_ro + self.b_wo)

            c_hat_t = T.tanh(
                T.dot(x_t, self.W_c) + T.dot(h_tm1, self.R_c) + self.b_wc + self.b_rc)
            c_t = f_t * c_tm1 + i_t * c_hat_t
            h_t = o_t * T.tanh(c_t)

            return h_t, c_t

        outputs_info = [self.s0, self.c0]

        states, updates = theano.scan(
            fn=step,
            sequences=[self.X],
            outputs_info=outputs_info
            )

        self.Y = states[0]
        self.C = states[1]

    def output(self):
        return self.Y

def test_dnn_rnn_lstm():
    # test params
    input_dim = 32
    hidden_dim = 16
    batch_size = 2
    depth = 3
    timesteps = 5

    # test code
    X = T.tensor3('X')
    Y = T.tensor3('Y')
    h0 = T.tensor3('h0')
    c0 = T.tensor3('c0')

    rnnb = dnn.RNNBlock(theano.config.floatX, hidden_dim, depth, 'lstm')
    psize = rnnb.get_param_size([batch_size, input_dim])
    params_cudnn = gpuarray_shared_constructor(
        numpy.zeros((psize,), dtype=theano.config.floatX))

    model = Model()
    last_layer = WrapperLayer(X)
    last_dim = input_dim
    for i in range(depth):
        lstm = LSTM(last_dim, hidden_dim, last_layer, s0=h0[i, :, :], c0=c0[i, :, :])
        model.add_layer(lstm)
        last_layer = lstm
        last_dim = hidden_dim
        layer_params = lstm.get_params()
        dnn_params = rnnb.split_params(params_cudnn, i,
                                       [batch_size, input_dim])
        for j, p in enumerate(dnn_params):
            p[:] = layer_params[j].get_value(borrow=True,
                                             return_internal_type=True)

    def funcs(out, params):
        fn = theano.function([X, h0, c0], out, mode=mode_with_gpu)
        cost = T.mean((Y - out)**2)
        grad = T.grad(cost, [X, h0, c0] + params)
        grad_fn = theano.function([X, Y, h0, c0], grad, mode=mode_with_gpu)
        return fn, grad_fn

    ref_fn, ref_grad_fn = funcs(last_layer.output(),
                                model.get_params())
    cudnn_fn, cudnn_grad_fn = funcs(rnnb.apply(params_cudnn, X, h0, c0)[0],
                                    [params_cudnn])

    x_val = numpy.random.random((timesteps, batch_size, input_dim)).astype(theano.config.floatX)
    y_val = numpy.random.random((timesteps, batch_size, hidden_dim)).astype(theano.config.floatX)
    h0_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)
    c0_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)

    ref_out = ref_fn(x_val, h0_val, c0_val)
    cudnn_out = cudnn_fn(x_val, h0_val, c0_val)

    utt.assert_allclose(ref_out, cudnn_out)

    ref_grads = ref_grad_fn(x_val, y_val, h0_val, c0_val)
    cudnn_grads = cudnn_grad_fn(x_val, y_val, h0_val, c0_val)

    utt.assert_allclose(ref_grads[0], cudnn_grads[0])
    utt.assert_allclose(ref_grads[1], cudnn_grads[1])
    utt.assert_allclose(ref_grads[2], cudnn_grads[2])

    ref_grads_params = ref_grads[3:]
    cudnn_grads_params = gpuarray_shared_constructor(cudnn_grads[3])

    for i in range(depth):
        cudnn_grads_layer = rnnb.split_params(cudnn_grads_params, i,
                                              [batch_size, input_dim])
        ref_grads_layer = ref_grads_params[i * len(cudnn_grads_layer):
                                           (i + 1) * len(cudnn_grads_layer)]
        for j, g in enumerate(cudnn_grads_layer):
            utt.assert_allclose(ref_grads_layer[j], g)

test_dnn_rnn_lstm()