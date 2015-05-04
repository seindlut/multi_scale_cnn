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
import pdb


from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from multi_scale_cnn import MyNetConvPoolLayer

def load_params(path, file_name, num_layers):
    parameters = unpickle(path+file_name)

    # parameters loaded may be cudaarrays, first convert them into numpy arrays    
    layer_params = []
    for i in range (num_layers):
        layer_params.append(numpy.asarray(parameters[i].get_value()))
    pdb.set_trace()
    return layer_params   # return type is a python list of numpy arrays

def evaluate_cifar10(dataset_path, dataset_name, param_path, param_name, num_layers):
    """ This function is used to evaluate cifar10 dataset """
    rng = numpy.random.RandomState(23455) # generate random number seed
    # initialization of dataset
    num_channels = 3                      				  # for RGB 3-channel image inputs
    layer0_rows = 32                      				  # image height 
    layer0_cols = 32                      				  # image width
    layer0_sub_rows = layer0_rows / 2     				  # layer0_sub_rows = 16
    layer0_sub_cols = layer0_cols / 2      				  # layer0_sub_cols = 16
    kernel0_size = 5                      				  # filter size of first layer kernels
    pool0_size = 2                        				  # pool size of the first layer
    layer1_rows = (layer0_rows - kernel0_size + 1) / pool0_size           # layer1_rows = 14
    layer1_cols = (layer0_cols - kernel0_size + 1) / pool0_size           # layer1_cols = 14
    layer1_sub_rows = (layer0_sub_rows - kernel0_size + 1) / pool0_size   # layer1_sub_rows = 6
    layer1_sub_cols = (layer0_sub_cols - kernel0_size + 1) / pool0_size   # layer1_sub_cols = 6
    kernel1_size = 5
    pool1_size = 1                                                        # no pooling for the first layer
    layer2_rows = (layer1_rows - kernel1_size + 1) / pool1_size           # layer2_rows = 5
    layer2_cols = (layer1_cols - kernel1_size + 1) / pool1_size           # layer2_cols = 5
    hidden_nodes = 500
    batch_size = 10000
    nkerns = [20, 50]

 
    dataset = unpickle(dataset_path+dataset_name)
    dataset_x = dataset['data']                        			  # image set 10000 images
    dataset_y = dataset['labels']                     			  # label set
    assert dataset_x.shape[0] == len(dataset_y)
    evalset_x, evalset_y = share_data(dataset_x, dataset_y)
    num_batches = len(dataset_y) / batch_size
    del dataset_x
    del dataset_y
    
    # get variable names for data and labels
    x = T.matrix('x')
    y = T.ivector('y')
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... buliding the model'    
    net_params = load_params(param_path, param_name, num_layers)
    layer0_input = x.reshape((batch_size, num_channels, layer0_rows, layer0_cols))
    layer0_input_sub = downsample.max_pool_2d(input=layer0_input, ds=(2,2), ignore_border=True)

    layer0 = MyNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, num_channels, layer0_rows, layer0_cols),        # image_shape = (500, 3, 32, 32)
        filter_shape=(nkerns[0], num_channels, kernel0_size, kernel0_size),      # filter_shape= (20, 3, 5, 5)
        poolsize=(pool0_size, pool0_size),
        params_W=net_params[10],
        params_b=net_params[11]
    )                                      # construct the first layer
    layer0_sub = MyNetConvPoolLayer(
        rng,
        input=layer0_input_sub,
        image_shape=(batch_size, num_channels, layer0_sub_rows, layer0_sub_cols),# image_shape = (500, 3, 16, 16)
        filter_shape=(nkerns[0], num_channels, kernel0_size, kernel0_size),      # filter_shape= (20, 3, 5, 5)
        poolsize=(pool0_size, pool0_size),
        params_W=net_params[8],
        params_b=net_params[9]
    )

    layer1 = MyNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_rows, layer1_cols),           # image_shape = (500, 20, 14, 14)
        filter_shape=(nkerns[1], nkerns[0], kernel1_size, kernel1_size),         # filter_shape= (50, 20, 5, 5)
        poolsize=(pool1_size, pool1_size),
        params_W=net_params[6],
        params_b=net_params[7]
    )                                                                            # output size = (500, 50, 10, 10)
    layer1_sub = MyNetConvPoolLayer(
        rng,
        input=layer0_sub.output,
        image_shape=(batch_size, nkerns[0], layer1_sub_rows, layer1_sub_cols), # image_shape = (500, 20, 6, 6)
        filter_shape=(nkerns[1], nkerns[0], kernel1_size, kernel1_size),         # filter_shape= (50, 20, 5, 5)
        poolsize=(pool1_size, pool1_size),
        params_W=net_params[4],
        params_b=net_params[5]
    )                                                                            # output size = (500, 50, 2, 2)

    layer2_input = T.concatenate([layer1.output.flatten(2), layer1_sub.output.flatten(2)], axis=1)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1]*((layer1_rows+1-kernel1_size)*(layer1_cols+1-kernel1_size)+(layer1_sub_rows+1-kernel1_size)*(layer1_sub_cols+1-kernel1_size)),
        n_out=hidden_nodes,
        W=net_params[2],
        b=net_params[3],
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=hidden_nodes, n_out=10, W=net_params[0], b=net_params[1])

    # Then we only consider the predictions of layer3 and layer3_sub
    # Pridictions may not be the same, in this case we choose the one with
    # more certainty!
    index = theano.tensor.lscalar()

    test_model = theano.function(
        [],                                            
        layer3.errors(y),                     
        givens={
            x: evalset_x,
            y: evalset_y
        }
    )

    accuracy_rate = test_model()
    print accuracy_rate

if __name__ == "__main__":
    param_path = '/home/ramp/lishaohua/obj_detection/DeepLearningTutorials/code/'
    param_fname = 'p3'
    dataset_path = '/home/ramp/lishaohua/obj_detection/DeepLearningTutorials/data/cifar10/'
    dataset_name = 'data_batch_5'
    num_layers = 12 
    evaluate_cifar10(dataset_path, dataset_name, param_path, param_fname, num_layers)
