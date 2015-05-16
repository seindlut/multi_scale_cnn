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

def hidden_layer_states(datapath, trainset_name, param_file_name, nkerns=[32,32,64], batch_size=10000):
    """ This function is used to test the accuracy of cifar10. """
    # layer parameters
    rng = numpy.random.RandomState(23455)
    num_train_images = 50000
    num_batches = num_train_images / batch_size
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
    trainx_list, trainy_list = load_cifar10_train(datapath, trainset_name, 
                                 original_image_column_width)
    trainx_list = mean_subtraction_preprocessing(trainx_list)
    trainx_list = unit_scaling(trainx_list)
    train_x, train_y = share_data(trainx_list, trainy_list)

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

    batch_index = T.iscalar()
    hls = theano.function(
        [batch_index],
        fc_layer0.output, 
        givens={
            x: train_x[batch_index * batch_size : (batch_index+1) * batch_size],
        }
    )

    states = []
    for i in range(num_batches):
        errors = hls(i)
        states.append(errors)

    # save hidden layer node states
    f = open('hls','wb')
    cPickle.dump(states, f)
    f.close()

if __name__ == '__main__':
    datapath = '../data/cifar10/'
    testset_name =  ['data_batch_1','data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    param_file_name = 'parameters'
    hidden_layer_states(datapath, testset_name, param_file_name)
