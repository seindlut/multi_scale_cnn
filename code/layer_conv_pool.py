import os
import sys
import time

import theano
import theano.tensor as T
import cPickle

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import hard_sigmoid

import numpy
from utils import relu
import pdb

class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, 
                 poolsize=(2,2), params_W=None, params_b=None, 
                 mu=0, sigma=0.1, bias_val=0.5, activation_mode=0):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # if params_W is not given, generate random params_W        
        if params_W == None:
            self.W = theano.shared(
                numpy.asarray(
                    rng.normal(mu, sigma, filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )    
        else:
            self.W = theano.shared(
                numpy.asarray(params_W,dtype=theano.config.floatX), borrow=True)

        # if params_b is not given, generate random params_b
        if params_b == None:
            b_values = bias_val * numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = theano.shared(
                numpy.asarray(params_b,dtype=theano.config.floatX), borrow=True
            )

        self.momentum_W = theano.shared(
            numpy.zeros(filter_shape, dtype=theano.config.floatX),
            borrow=True
        )
        self.momentum_b = theano.shared(
            numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )  

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
        if activation_mode == 0: 
            self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        elif activation_mode == 1:
            self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]
