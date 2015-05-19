import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 mu=0, sigma=0.1):
        randnumgen = numpy.random.RandomState(12345)
        # if W is not given, random generate W
        if W is None:
            self.W = theano.shared(
                value=numpy.asarray(
                    randnumgen.normal(mu, sigma, (n_in, n_out)),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = theano.shared(value=W, name='W', borrow=True)

        # if b is not given, random generate b
        if b is None:
            self.b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = theano.shared(value=b, name='b', borrow=True)

        self.momentum_W = theano.shared(
            numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            borrow=True
        )
        self.momentum_b = theano.shared(
            numpy.zeros((n_out), dtype=theano.config.floatX),
            borrow=True
        ) 

        # probability of each label
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # predicted label
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameter list
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

        # cross entropy cost
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):

        # incorrespondence
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
