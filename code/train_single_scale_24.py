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

def train_cifar10(datapath, trainset_name, valset_name,
                  learning_rate=0.01, n_epochs=10,
                  nkerns=[32,32,64], batch_size=5000):
    """ This function is used to train cifar10 dataset for object recognition."""
    rng = numpy.random.RandomState(23455)                        # generate random number seed
    num_images = 50000
    num_test_images = 10000
    num_channels = 3                                             # for RGB 3-channel image inputs
    original_image_rows = 32
    original_image_cols = 32
    original_image_column_width = original_image_rows * original_image_cols * num_channels
    border_crop = 8 

    # convolutional layer 0 parameters #
    conv_layer0_rows = original_image_rows - border_crop              # image height 24
    conv_layer0_cols = original_image_cols - border_crop              # image width  24
    conv_layer0_pixels = conv_layer0_rows * conv_layer0_cols          # number of pixels in a layer: 576
    conv_layer0_column_width = conv_layer0_pixels * num_channels      # column_width = 1728
    conv_layer0_kernel_size = 5                                       # filter size of first layer kernels
    conv_layer0_pool_size = 2                                         # pool size of the first layer

    # convolutional layer 1 parameters #
    conv_layer1_rows = (conv_layer0_rows - conv_layer0_kernel_size + 1) / conv_layer0_pool_size       # conv_layer1_rows = 10
    conv_layer1_cols = (conv_layer0_cols - conv_layer0_kernel_size + 1) / conv_layer0_pool_size       # conv_layer1_cols = 10
    conv_layer1_kernel_size = 5 
    conv_layer1_pool_size = 1                                                                         # no pooling for the first layer

    # convolutional layer 2 parameters #
    conv_layer2_rows = (conv_layer1_rows - conv_layer1_kernel_size + 1) / conv_layer1_pool_size       # layer1_5_rows = 6
    conv_layer2_cols = (conv_layer1_cols - conv_layer1_kernel_size + 1) / conv_layer1_pool_size       # layer1_5_cols = 6
    conv_layer2_kernel_size = 3 
    conv_layer2_pool_size = 1 

    # output rows and columns of convolutional net #
    conv_output_rows = (conv_layer2_rows - conv_layer2_kernel_size + 1) / conv_layer2_pool_size       # layer2_rows = 4
    conv_output_cols = (conv_layer2_cols - conv_layer2_kernel_size + 1) / conv_layer2_pool_size       # layer2_cols = 4

    # fully connected layer parameters #
    fc_layer0_hidden_nodes = 64 
    fc_layer1_hidden_nodes = 10

    # optimization parameters
    momentum_coeff = 0.9
    weight_decay = 0.0001
    penalty_coeff = 0.01 

    num_batches = num_images / batch_size
    num_test_batches = num_test_images / batch_size

    # load in raw training data
    trainx_list, trainy_list, valx_list, valy_list = load_cifar10(datapath, trainset_name, valset_name, original_image_column_width)

    # preprocessing with mean subtraction
    trainx_list = mean_subtraction_preprocessing(trainx_list)
    valx_list   = mean_subtraction_preprocessing(valx_list)

    # preprocessing with unit scaling
#    trainx_list = unit_scaling(trainx_list)
#    valx_list   = unit_scaling(valx_list)

    # make shared dataset
    train_x, train_y = share_data(trainx_list, trainy_list)
    val_x, val_y = share_data(valx_list, valy_list)

    # get variable names for data and labels
    x = T.matrix('x')
    y = T.ivector('y')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... buliding the model'    

    full_size_input = x.reshape((batch_size, num_channels, original_image_rows, original_image_cols))
    conv_layer0_input = crop_images(full_size_input, (batch_size, num_channels, original_image_rows, original_image_cols))

    conv_layer0 = ConvPoolLayer(
        rng,
        input=conv_layer0_input,
        image_shape=(batch_size, num_channels, conv_layer0_rows, conv_layer0_cols),                    # image_shape = (500, 3, 32, 32)
        filter_shape=(nkerns[0], num_channels, conv_layer0_kernel_size, conv_layer0_kernel_size),      # filter_shape= (20, 3, 5, 5)
        poolsize=(conv_layer0_pool_size, conv_layer0_pool_size),
        sigma=0.0001,
        activation_mode=1
    )

    norm_layer0 = NormalizationLayer(
        data=conv_layer0.output,
        alpha=0.00005,
        mode=1
    )
  
    conv_layer1 = ConvPoolLayer(
        rng,
        input=norm_layer0.output,
        image_shape=(batch_size, nkerns[0], conv_layer1_rows, conv_layer1_cols),           # image_shape = (500, 20, 14, 14)
        filter_shape=(nkerns[1], nkerns[0], conv_layer1_kernel_size, conv_layer1_kernel_size),         # filter_shape= (50, 20, 5, 5)
        poolsize=(conv_layer1_pool_size, conv_layer1_pool_size),
        sigma=0.01,
        activation_mode=1
    ) 

    norm_layer1 = NormalizationLayer(
        data=conv_layer1.output,
        alpha=0.00005,
        mode=1
    )

    conv_layer2 = ConvPoolLayer(
        rng,
        input=norm_layer1.output,
        image_shape=(batch_size, nkerns[1], conv_layer2_rows, conv_layer2_cols),
        filter_shape=(nkerns[2], nkerns[1], conv_layer2_kernel_size, conv_layer2_kernel_size),
        poolsize=(conv_layer2_pool_size, conv_layer2_pool_size),
        sigma=0.01,
        activation_mode=1
    )

#    norm_layer2 = NormalizationLayer(
#        data=conv_layer2.output
#    )

    fc_layer0_input = conv_layer2.output.flatten(2)
    fc_layer0 = HiddenLayer(
        rng,
        input=fc_layer0_input,
	    n_in=nkerns[2] * conv_output_rows * conv_output_cols, 
        n_out=fc_layer0_hidden_nodes,
        activation=relu
    )    

    fc_layer1 = HiddenLayer(
        rng,
        input=fc_layer0.output,
    	n_in=fc_layer0_hidden_nodes, 
        n_out=fc_layer1_hidden_nodes,
        activation=relu
    )

    class_layer0 = LogisticRegression(input=fc_layer1.output, n_in=fc_layer1_hidden_nodes, n_out=10)

    # compare the difference between regularization of hidden layer weights and classifier weights.
    total_cost = class_layer0.negative_log_likelihood(y) + penalty_coeff * class_layer0.W.norm(2)

#    params = class_layer0.params + fc_layer1.params + fc_layer0.params + conv_layer2.params + conv_layer1.params + conv_layer0.params
#    grads = T.grad(total_cost, params)
#    updates = [
#        (param_i, param_i - learning_rate * grad_i)
#        for param_i, grad_i in zip(params, grads)
#    ]


    grad_classl0     = T.grad(total_cost, class_layer0.params)
    grad_fcl1        = T.grad(total_cost, fc_layer1.params)
    grad_fcl0        = T.grad(total_cost, fc_layer0.params)
    grad_convl2      = T.grad(total_cost, conv_layer2.params)
    grad_convl1      = T.grad(total_cost, conv_layer1.params)
    grad_convl0      = T.grad(total_cost, conv_layer0.params)

    updates = [
        (conv_layer0.momentum_W, momentum_coeff * conv_layer0.momentum_W - weight_decay * learning_rate * conv_layer0.params[0] - learning_rate * grad_convl0[0]),
        (conv_layer0.momentum_b, momentum_coeff * conv_layer0.momentum_b - weight_decay * learning_rate * conv_layer0.params[1] - learning_rate * grad_convl0[1]),
        (conv_layer0.params[0], conv_layer0.params[0] + momentum_coeff * conv_layer0.momentum_W - weight_decay * learning_rate * conv_layer0.params[0] - learning_rate * grad_convl0[0]),
        (conv_layer0.params[1], conv_layer0.params[1] + momentum_coeff * conv_layer0.momentum_b - weight_decay * learning_rate * conv_layer0.params[1] - learning_rate * grad_convl0[1]),

        (conv_layer1.momentum_W, momentum_coeff * conv_layer1.momentum_W - weight_decay * learning_rate * conv_layer1.params[0] - learning_rate * grad_convl1[0]),
        (conv_layer1.momentum_b, momentum_coeff * conv_layer1.momentum_b - weight_decay * learning_rate * conv_layer1.params[1] - learning_rate * grad_convl1[1]),
        (conv_layer1.params[0], conv_layer1.params[0] + momentum_coeff * conv_layer1.momentum_W - weight_decay * learning_rate * conv_layer1.params[0] - learning_rate * grad_convl1[0]),
        (conv_layer1.params[1], conv_layer1.params[1] + momentum_coeff * conv_layer1.momentum_b - weight_decay * learning_rate * conv_layer1.params[1] - learning_rate * grad_convl1[1]),

        (conv_layer2.momentum_W, momentum_coeff * conv_layer2.momentum_W - weight_decay * learning_rate * conv_layer2.params[0] - learning_rate * grad_convl2[0]),
        (conv_layer2.momentum_b, momentum_coeff * conv_layer2.momentum_b - weight_decay * learning_rate * conv_layer2.params[1] - learning_rate * grad_convl2[1]),
        (conv_layer2.params[0], conv_layer2.params[0] + momentum_coeff * conv_layer2.momentum_W - weight_decay * learning_rate * conv_layer2.params[0] - learning_rate * grad_convl2[0]),
        (conv_layer2.params[1], conv_layer2.params[1] + momentum_coeff * conv_layer2.momentum_b - weight_decay * learning_rate * conv_layer2.params[1] - learning_rate * grad_convl2[1]),

        (fc_layer0.momentum_W, momentum_coeff * fc_layer0.momentum_W - weight_decay * learning_rate * fc_layer0.params[0] - learning_rate * grad_fcl0[0]),
        (fc_layer0.momentum_b, momentum_coeff * fc_layer0.momentum_b - weight_decay * learning_rate * fc_layer0.params[1] - learning_rate * grad_fcl0[1]),
        (fc_layer0.params[0], fc_layer0.params[0] + momentum_coeff * fc_layer0.momentum_W - weight_decay * learning_rate * fc_layer0.params[0] - learning_rate * grad_fcl0[0]),
        (fc_layer0.params[1], fc_layer0.params[1] + momentum_coeff * fc_layer0.momentum_b - weight_decay * learning_rate * fc_layer0.params[1] - learning_rate * grad_fcl0[1]),

        (fc_layer1.momentum_W, momentum_coeff * fc_layer1.momentum_W - weight_decay * learning_rate * fc_layer1.params[0] - learning_rate * grad_fcl1[0]),
        (fc_layer1.momentum_b, momentum_coeff * fc_layer1.momentum_b - weight_decay * learning_rate * fc_layer1.params[1] - learning_rate * grad_fcl1[1]),
        (fc_layer1.params[0], fc_layer1.params[0] + momentum_coeff * fc_layer1.momentum_W - weight_decay * learning_rate * fc_layer1.params[0] - learning_rate * grad_fcl1[0]),
        (fc_layer1.params[1], fc_layer1.params[1] + momentum_coeff * fc_layer1.momentum_b - weight_decay * learning_rate * fc_layer1.params[1] - learning_rate * grad_fcl1[1]),

        (class_layer0.momentum_W, momentum_coeff * class_layer0.momentum_W - weight_decay * learning_rate * class_layer0.params[0] - learning_rate * grad_classl0[0]), 
        (class_layer0.momentum_b, momentum_coeff * class_layer0.momentum_b - weight_decay * learning_rate * class_layer0.params[1] - learning_rate * grad_classl0[1]), 
        (class_layer0.params[0], class_layer0.params[0] + momentum_coeff * class_layer0.momentum_W - weight_decay * learning_rate * class_layer0.params[0] - learning_rate * grad_classl0[0]),
        (class_layer0.params[1], class_layer0.params[1] + momentum_coeff * class_layer0.momentum_b - weight_decay * learning_rate * class_layer0.params[1] - learning_rate * grad_classl0[1]),
    ]

    training_index = T.iscalar()
    validate_index = T.iscalar()

    train_model = theano.function(
        [training_index],
        [total_cost, class_layer0.errors(y)],
        updates=updates,
        givens={
            x : train_x[training_index * batch_size : (training_index+1) * batch_size],
            y : train_y[training_index * batch_size : (training_index+1) * batch_size]
        }
    )

    test_model = theano.function(
        [validate_index],
        class_layer0.errors(y),
        givens={
            x: val_x[validate_index * batch_size : (validate_index+1) * batch_size],
            y: val_y[validate_index * batch_size : (validate_index+1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    patience = 10000
    epoch = 0

    while(epoch < n_epochs):
        batch_index = randint(0, num_batches-1)                                # randomly generate the batch number to be trained.
        print "selected training batch:", batch_index
        cost_ij, error = train_model(batch_index)
        epoch = epoch + 1
        print "number of iterations:   ", epoch 
        print "current cost:           ", cost_ij
        print "training error:         ", error
 
        if (epoch % 5 == 0):
            error_test = 0
            for test_batch_index in range(num_test_batches):
                error_test = error_test + test_model(test_batch_index)
            error_test = error_test / float(num_test_batches)    
            print "      "
            print "validate error of test_batch:", error_test
            print "      "
     
        if (epoch == n_epochs):
            saved_params = [training_cost, validation_error]
            saved_file = open('cost_error', 'wb')
            cPickle.dump(saved_params, saved_file)
            saved_file.close()
 
if __name__ == '__main__':
    dsetname = ['data_batch_1','data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'];  #dataset file names.
    vsetname = 'test_batch'
    train_cifar10('../data/cifar10/', dsetname, vsetname)
