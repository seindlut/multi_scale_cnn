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

from IO import * 
from utils import *
from layer_conv_pool import ConvPoolLayer
from layer_normalization import NormalizationLayer
from layer_multi_layer_perceptron import HiddenLayer, DropoutHiddenLayer
from layer_logistic_regression import LogisticRegression

import pdb

def test_cifar10(datapath, testset_name, param_file_name, nkerns=[32,32,64], batch_size=10000):
    """ This function is used to test the accuracy of cifar10. """
    # layer parameters
    rng = numpy.random.RandomState(23455)
    num_test_images = 10000
    num_batches = num_test_images / batch_size
    num_channels = 3                          
    original_image_rows = 32
    original_image_cols = 32
    original_image_column_width = original_image_rows * original_image_cols * num_channels

    conv_layer0_rows = original_image_rows       
    conv_layer0_cols = original_image_cols  
    conv_layer0_pixels = conv_layer0_rows * conv_layer0_cols 
    conv_layer0_column_width = conv_layer0_pixels * num_channels     
    conv_layer0_kernel_size = 5  
    conv_layer0_pool_size = 2  

    conv_layer1_rows = (conv_layer0_rows - conv_layer0_kernel_size + 1) / conv_layer0_pool_size 
    conv_layer1_cols = (conv_layer0_cols - conv_layer0_kernel_size + 1) / conv_layer0_pool_size
    conv_layer1_kernel_size = 5 
    conv_layer1_pool_size = 1
                                                                         
    conv_layer2_rows = (conv_layer1_rows - conv_layer1_kernel_size + 1) / conv_layer1_pool_size
    conv_layer2_cols = (conv_layer1_cols - conv_layer1_kernel_size + 1) / conv_layer1_pool_size 
    conv_layer2_kernel_size = 5 
    conv_layer2_pool_size = 1 

    conv_output_rows = (conv_layer2_rows - conv_layer2_kernel_size + 1) / conv_layer2_pool_size
    conv_output_cols = (conv_layer2_cols - conv_layer2_kernel_size + 1) / conv_layer2_pool_size           
    fc_layer0_hidden_nodes = 64 

    # load in test data and preprocess
    testx_list, testy_list = load_cifar10_1set(datapath, testset_name, 
                                 original_image_column_width)
    testx_list = mean_subtraction_preprocessing(testx_list)
    testx_list = unit_scaling(testx_list)
    test_x, test_y = share_data(testx_list, testy_list)

    # load in trained parameters
    parameters = unpickle(param_file_name)
    # get variable names for data and labels
    x = T.matrix('x')
    y = T.ivector('y')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... buliding the model'    

    conv_layer0_input = x.reshape((batch_size, num_channels, original_image_rows, original_image_cols))

    conv_layer0 = ConvPoolLayer(
        rng,
        input=conv_layer0_input,
        image_shape=(batch_size, num_channels, conv_layer0_rows, conv_layer0_cols),                
        filter_shape=(nkerns[0], num_channels, conv_layer0_kernel_size, conv_layer0_kernel_size),
        poolsize=(conv_layer0_pool_size, conv_layer0_pool_size),
        params_W=parameters[0].get_value(), params_b=parameters[1].get_value(),
        activation_mode=1
    )
  
    conv_layer1 = ConvPoolLayer(
        rng,
        input=conv_layer0.output,
        image_shape=(batch_size, nkerns[0], conv_layer1_rows, conv_layer1_cols),    
        filter_shape=(nkerns[1], nkerns[0], conv_layer1_kernel_size, conv_layer1_kernel_size), 
        poolsize=(conv_layer1_pool_size, conv_layer1_pool_size),
        params_W=parameters[2].get_value(), params_b=parameters[3].get_value(),
        activation_mode=1
    ) 

    conv_layer2 = ConvPoolLayer(
        rng,
        input=conv_layer1.output,
        image_shape=(batch_size, nkerns[1], conv_layer2_rows, conv_layer2_cols),
        filter_shape=(nkerns[2], nkerns[1], conv_layer2_kernel_size, conv_layer2_kernel_size),
        poolsize=(conv_layer2_pool_size, conv_layer2_pool_size),
        params_W=parameters[4].get_value(), params_b=parameters[5].get_value(),
        activation_mode=1
    )


    fc_layer0_input = conv_layer2.output.flatten(2)
    fc_layer0 = HiddenLayer(
        rng,
        input=fc_layer0_input,
	    n_in=nkerns[2] * conv_output_rows * conv_output_cols, 
        n_out=fc_layer0_hidden_nodes,
        W=parameters[6].get_value(), b=parameters[7].get_value(),
        activation=relu
    )    

    class_layer0 = LogisticRegression(
        input=fc_layer0.output, 
        n_in=fc_layer0_hidden_nodes, 
        n_out=10,
        W=parameters[8].get_value(),
        b=parameters[9].get_value()
    )

    test_model = theano.function(
        [],
        class_layer0.errors(y),
        givens={
            x: test_x,
            y: test_y
        }
    )

    errors = test_model()
    print "error = ", test_model

if __name__ == '__main__':
    datapath = '../data/cifar10/'
    testset_name = 'test_batch'
    param_file_name = 'parameters'
    test_cifar10(datapath, testset_name, param_file_name)
