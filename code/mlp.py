import os
import sys
import time

import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression 


# TODO: determine the state variable
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """ state is the symbol to distinguish between training and testing,
            0 for training and 1 for testing.
        """
        self.input = input

        if W is None:
#            W = numpy.asarray(
#                rng.uniform(
#                    low=-numpy.sqrt(6. / (n_in + n_out)),
#                    high=numpy.sqrt(6. / (n_in + n_out)),
#                    size=(n_in, n_out)
#                ),
#                dtype=theano.config.floatX
#            )
            mu, sigma = 0, 0.1
            W = numpy.asarray(
               rng.normal(mu, sigma, (n_in, n_out)),
               dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W *= 4

        if b is None:
            b = numpy.zeros((n_out,), dtype=theano.config.floatX)

        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)

        self.momentum_W = theano.shared(
            numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            borrow=True
        )
        self.momentum_b = theano.shared(
            numpy.zeros((n_out,), dtype=theano.config.floatX),
            borrow=True
        )

        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]

class DropoutHiddenLayer(object):
    def __init__(self, mrng, rng, input, n_in, n_out, W=None, b=None,
                 state=0, dropout_rate=0.5, activation=T.tanh):
        """ state is the symbol to distinguish between training and testing,
            0 for training and 1 for testing.
        """
        self.input = input

        if W is None:
            W = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W *= 4

        if b is None:
            b = numpy.zeros((n_out,), dtype=theano.config.floatX)

        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)

        linear_output = (T.dot(input, self.W) + self.b)
        if (state == 0):
            lin_output = linear_output * mrng.binomial(size=(n_out, ), p=dropout_rate)
        else:
            lin_output = dropout_rate * linear_output

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]
