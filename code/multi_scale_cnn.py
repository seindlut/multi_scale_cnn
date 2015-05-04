# TODO: 1. analysis of activation state.
#       2. use traditional LeNet5 for accuracy checking.    
#       3. add dropout by using a simple mask of (0, 1)
#       4. analysis of activation state of hidden layer.
#          3.1 finish training and save parameters
#          3.2 forward propagate training set and record hidden layer state.
#          3.3 display histogram of hidden layer state.


import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import cPickle
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import hard_sigmoid
from random import randint
from IO import unpickle
from IO import share_data
from preprocess import normalize
import pdb


from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class MyNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2), params_W=None, params_b=None):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # if params_W is not given, generate random params_W        
        if params_W == None:
            fan_in = numpy.prod(filter_shap[1:])
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                       numpy.prod(poolsize))
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = theano.shared(
                numpy.asarray(params_W,dtype=theano.config.floatX), borrow=True)

        # if params_b is not given, generate random params_b
        if params_b == None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = theano.shared(
                numpy.asarray(params_b,dtype=theano.config.floatX), borrow=True)

        # feature maps after convolution
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        
        # feature maps after pooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # output of layer, activated pooled feature maps 
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # parameters list
        self.params = [self.W, self.b]
