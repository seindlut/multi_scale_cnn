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


# important learning rate 0.05
# important learning rate [20, 50]
def train_cifar10(datapath, dataset_name,
                  learning_rate=0.05, n_epochs=6000,
                  nkerns=[32,32,64], batch_size=500):
    """ This function is used to train cifar10 dataset for object recognition."""
    rng = numpy.random.RandomState(23455)                        # generate random number seed
    num_images = 50000
    num_test_images = 10000
    num_channels = 3                                             # for RGB 3-channel image inputs

    # convolutional layer 0 parameters #
    conv_layer0_rows = 32                                             # image height 32
    conv_layer0_cols = 32                                             # image width  32
    conv_layer0_pixels = conv_layer0_rows * conv_layer0_cols          # number of pixels in a layer: 1024
    conv_layer0_column_width = conv_layer0_pixels * num_channels      # column_width = 3072
    conv_layer0_kernel_size = 5                                       # filter size of first layer kernels
    conv_layer0_pool_size = 2                                         # pool size of the first layer

    # convolutional layer 1 parameters #
    conv_layer1_rows = (conv_layer0_rows - conv_layer0_kernel_size + 1) / conv_layer0_pool_size           # conv_layer1_rows = 14
    conv_layer1_cols = (conv_layer0_cols - conv_layer0_kernel_size + 1) / conv_layer0_pool_size           # conv_layer1_cols = 14
    conv_layer1_kernel_size = 5
    conv_layer1_pool_size = 2                                                                             # no pooling for the first layer

    # convolutional layer 2 parameters #
    conv_layer2_rows = (conv_layer1_rows - conv_layer1_kernel_size + 1) / conv_layer1_pool_size         # layer1_5_rows = 10
    conv_layer2_cols = (conv_layer1_cols - conv_layer1_kernel_size + 1) / conv_layer1_pool_size         # layer1_5_cols = 10
    conv_layer2_kernel_size = 3 
    conv_layer2_pool_size = 1 

    # output rows and columns of convolutional net #
    conv_output_rows = (conv_layer2_rows - conv_layer2_kernel_size + 1) / conv_layer2_pool_size       # layer2_rows = 6 
    conv_output_cols = (conv_layer2_cols - conv_layer2_kernel_size + 1) / conv_layer2_pool_size       # layer2_cols = 6

    # fully connected layer parameters #
#    fc_layer0_hidden_nodes = 64 
    fc_layer0_hidden_nodes = 32 
#    fc_layer1_hidden_nodes = 10

    # optimization parameters
    momentum_coeff = 0.9
    weight_decay = 0.0001
    penalty_coeff = 0.05 

    num_batches = num_images / batch_size
    num_test_batches = num_test_images / batch_size

    # read in data
    data_list  = numpy.empty(shape=[0, conv_layer0_column_width])         # for each set of training data,
                                                                          # column width is fixed.
    label_list = numpy.empty(shape=[0,])                                  # for each set of training labels,
                                                                          # row height is fixed.
    for i in range(len(dataset_name)):
        temp_data = unpickle(datapath+dataset_name[i])
        temp_x    = temp_data['data']
        temp_x    = mean_subtraction_preprocessing(temp_x)
        temp_y    = numpy.array(temp_data['labels'])                      # y labels are python lists, convert
                                                                          # to numpy.ndarray
        normalized_x = unit_scaling(temp_x)                                  # normalize data, rescale to 0 - 1
        
        data_list = numpy.append(data_list, normalized_x, axis=0)
        label_list= numpy.append(label_list, temp_y, axis=0)              # loop over the whole training set

    del temp_data, temp_x, temp_y, normalized_x
    shared_x, shared_y = share_data(data_list, label_list)

    validate_set = unpickle('../data/cifar10/test_batch')
    validate_x = validate_set['data']
    validate_x = mean_subtraction_preprocessing(validate_x)
    validate_y = validate_set['labels']
    normalized_valx = unit_scaling(validate_x)                               # normalize the validation set.
    evalset_x, evalset_y = share_data(normalized_valx, validate_y)
    del validate_set, validate_x, validate_y, normalized_valx

    # get variable names for data and labels
    x = T.matrix('x')
    y = T.ivector('y')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... buliding the model'    
    
    conv_layer0_input = x.reshape((batch_size, num_channels, conv_layer0_rows, conv_layer0_cols))

    conv_layer0 = ConvPoolLayer(
        rng,
        input=conv_layer0_input,
        image_shape=(batch_size, num_channels, conv_layer0_rows, conv_layer0_cols),        # image_shape = (500, 3, 32, 32)
        filter_shape=(nkerns[0], num_channels, conv_layer0_kernel_size, conv_layer0_kernel_size),      # filter_shape= (20, 3, 5, 5)
        poolsize=(conv_layer0_pool_size, conv_layer0_pool_size),
        activation_mode=1
    )

    norm_layer0 = NormalizationLayer(
        data=conv_layer0.output
    )
  
    conv_layer1 = ConvPoolLayer(
        rng,
#        input=conv_layer0.output,
        input=norm_layer0.output,
        image_shape=(batch_size, nkerns[0], conv_layer1_rows, conv_layer1_cols),           # image_shape = (500, 20, 14, 14)
        filter_shape=(nkerns[1], nkerns[0], conv_layer1_kernel_size, conv_layer1_kernel_size),         # filter_shape= (50, 20, 5, 5)
        poolsize=(conv_layer1_pool_size, conv_layer1_pool_size),
        activation_mode=1
    ) 

    norm_layer1 = NormalizationLayer(
        data=conv_layer1.output
    )

    conv_layer2 = ConvPoolLayer(
        rng,
#        input=conv_layer1.output,
        input=norm_layer1.output,
        image_shape=(batch_size, nkerns[1], conv_layer2_rows, conv_layer2_cols),
        filter_shape=(nkerns[2], nkerns[1], conv_layer2_kernel_size, conv_layer2_kernel_size),
        poolsize=(conv_layer2_pool_size, conv_layer2_pool_size),
        activation_mode=1
    )

    norm_layer2 = NormalizationLayer(
        data=conv_layer2.output
    )

#    fc_layer0_input = conv_layer2.output.flatten(2)    
    fc_layer0_input = norm_layer2.output.flatten(2)
    fc_layer0 = HiddenLayer(
        rng,
        input=fc_layer0_input,
	n_in=nkerns[2] * conv_output_rows * conv_output_cols, 
        n_out=fc_layer0_hidden_nodes,
        activation=relu
    )    

#    fc_layer1 = HiddenLayer(
#        rng,
#        input=fc_layer0.output,
#	n_in=fc_layer0_hidden_nodes, 
#        n_out=fc_layer1_hidden_nodes,
#        activation=relu
#    )
    # two logistic regression does not share any parameters, so there is
    # no predefined parameters.
#    class_layer0 = LogisticRegression(input=fc_layer1.output, n_in=fc_layer1_hidden_nodes, n_out=10)
    class_layer0 = LogisticRegression(input=fc_layer0.output, n_in=fc_layer0_hidden_nodes, n_out=10)


    # compare the difference between regularization of hidden layer weights and classifier weights.
    check = class_layer0.W.norm(2)
    total_cost = class_layer0.negative_log_likelihood(y) + penalty_coeff * class_layer0.W.norm(2)
#    total_cost = class_layer0.negative_log_likelihood(y) + penalty_coeff * (fc_layer0.W.norm(2)+fc_layer1.W.norm(2))

    grad_classl0     = T.grad(total_cost, class_layer0.params)
#    grad_fcl1        = T.grad(total_cost, fc_layer1.params)
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

#        (fc_layer1.momentum_W, momentum_coeff * fc_layer1.momentum_W - weight_decay * learning_rate * fc_layer1.params[0] - learning_rate * grad_fcl1[0]),
#        (fc_layer1.momentum_b, momentum_coeff * fc_layer1.momentum_b - weight_decay * learning_rate * fc_layer1.params[1] - learning_rate * grad_fcl1[1]),
#        (fc_layer1.params[0], fc_layer1.params[0] + momentum_coeff * fc_layer1.momentum_W - weight_decay * learning_rate * fc_layer1.params[0] - learning_rate * grad_fcl1[0]),
#        (fc_layer1.params[1], fc_layer1.params[1] + momentum_coeff * fc_layer1.momentum_b - weight_decay * learning_rate * fc_layer1.params[1] - learning_rate * grad_fcl1[1]),

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
            x : shared_x[training_index * batch_size : (training_index+1) * batch_size],
            y : shared_y[training_index * batch_size : (training_index+1) * batch_size]
        }
    )

    test_model = theano.function(
        [validate_index],
        class_layer0.errors(y),
        givens={
            x: evalset_x[validate_index * batch_size : (validate_index+1) * batch_size],
            y: evalset_y[validate_index * batch_size : (validate_index+1) * batch_size]
        }
    )

    check_norm = theano.function(
        [],
        check
    )
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    patience = 10000
    epoch = 0
    training_cost = []
    validation_error = []

    while(epoch < n_epochs):
        while(epoch < 2000):
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
                validation_error.append(error_test.tolist())
    
            training_cost.append(cost_ij.tolist())

        learning_rate = 0.01 
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
            validation_error.append(error_test.tolist())
     
            training_cost.append(cost_ij.tolist())

        if (epoch == n_epochs):
            saved_params = [training_cost, validation_error]
            saved_file = open('cost_error', 'wb')
            cPickle.dump(saved_params, saved_file)
            saved_file.close()
 
if __name__ == '__main__':
    dsetname = ['data_batch_1','data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'];  #dataset file names.
    train_cifar10('../data/cifar10/', dsetname)
