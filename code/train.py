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
from multi_scale_cnn import MyNetConvPoolLayer
import pdb


from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


# important learning rate 0.05
# important learning rate [20, 50]
def train_cifar10(datapath, dataset_name,
                  learning_rate=0.2, n_epochs=10000,
                  nkerns=[20, 50], batch_size=10000):
    """ This function is used to train cifar10 dataset for object recognition."""
    rng = numpy.random.RandomState(23455)                        # generate random number seed
    num_channels = 3                                             # for RGB 3-channel image inputs
    layer0_rows = 32                                             # image height 
    layer0_cols = 32                                             # image width
    layer_pixels = layer0_rows * layer0_cols                     # number of pixels in a layer: 1024
    column_width = layer_pixels * num_channels                   # column_width = 3072
    layer0_sub_rows = layer0_rows / 2                            # layer0_sub_rows = 16
    layer0_sub_cols = layer0_cols / 2                            # layer0_sub_cols = 16
    kernel0_size = 5                                             # filter size of first layer kernels
    pool0_size = 2                                               # pool size of the first layer
    layer1_rows = (layer0_rows - kernel0_size + 1) / pool0_size           # layer1_rows = 14
    layer1_cols = (layer0_cols - kernel0_size + 1) / pool0_size           # layer1_cols = 14
    layer1_sub_rows = (layer0_sub_rows - kernel0_size + 1) / pool0_size   # layer1_sub_rows = 6
    layer1_sub_cols = (layer0_sub_cols - kernel0_size + 1) / pool0_size   # layer1_sub_cols = 6
    kernel1_size = 5
    pool1_size = 1                                                        # no pooling for the first layer
    layer2_rows = (layer1_rows - kernel1_size + 1) / pool1_size           # layer2_rows = 5
    layer2_cols = (layer1_cols - kernel1_size + 1) / pool1_size           # layer2_cols = 5
    hidden_nodes = 50
    hidden_extra_nodes = 500
    penalty_coeff = 0.01

    # read in data
    data_list  = numpy.empty(shape=[0, column_width])                     # for each set of training data,
                                                                          # column width is fixed.
    label_list = numpy.empty(shape=[0,])                                # for each set of training labels,
                                                                          # row height is fixed.
    for i in range(len(dataset_name)):
        temp_data = unpickle(datapath+dataset_name[i])
        temp_x    = temp_data['data']
        temp_y    = numpy.array(temp_data['labels'])                      # y labels are python lists, convert
                                                                          # to numpy.ndarray
        normalized_x = normalize(temp_x)                                  # normalize data, rescale to 0 - 1
        
        data_list = numpy.append(data_list, normalized_x, axis=0)
        label_list= numpy.append(label_list, temp_y, axis=0)              # loop over the whole training set

    del temp_data, temp_x, temp_y, normalized_x
    shared_x, shared_y = share_data(data_list, label_list)

    validate_set = unpickle('../data/cifar10/test_batch')
    validate_x = validate_set['data']
    validate_y = validate_set['labels']
    normalized_valx = normalize(validate_x)                              # normalize the validation set.
    evalset_x, evalset_y = share_data(normalized_valx, validate_y)
    del validate_set, validate_x, validate_y, normalized_valx

    # get variable names for data and labels
    x = T.matrix('x')
    y = T.ivector('y')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... buliding the model'    
    # initialize layer0 parameters 
    layer0_fan_in = num_channels * layer0_rows * layer0_cols  # same as numpy.prod(filter_shape[1:])
    layer0_fan_out= nkerns[0] * kernel0_size * kernel0_size / (pool0_size * pool0_size)
    W_bound0 = numpy.sqrt(6. / (layer0_fan_in + layer0_fan_out))
    layer0_W = rng.uniform(low=-W_bound0, high=W_bound0, size=(nkerns[0], num_channels, kernel0_size, kernel0_size)) 
    
    # initialize layer1 parameters
    layer1_fan_in = num_channels * layer1_rows * layer1_cols  # same as numpy.prod(filter_shape[1:])
    layer1_fan_out= nkerns[1] * kernel1_size * kernel1_size / (pool1_size * pool1_size)
    W_bound1 = numpy.sqrt(6. / (layer1_fan_in + layer1_fan_out))
    layer1_W = rng.uniform(low=-W_bound1, high=W_bound1, size=(nkerns[1], nkerns[0], kernel1_size, kernel1_size)) 
    
    layer0_input = x.reshape((batch_size, num_channels, layer0_rows, layer0_cols))
    layer0_input_sub = downsample.max_pool_2d(input=layer0_input, ds=(2,2), ignore_border=True)

    layer0 = MyNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, num_channels, layer0_rows, layer0_cols),        # image_shape = (500, 3, 32, 32)
        filter_shape=(nkerns[0], num_channels, kernel0_size, kernel0_size),      # filter_shape= (20, 3, 5, 5)
        poolsize=(pool0_size, pool0_size),
        params_W=layer0_W,
    )                                      # construct the first layer
    layer0_sub = MyNetConvPoolLayer(
        rng,
        input=layer0_input_sub,
        image_shape=(batch_size, num_channels, layer0_sub_rows, layer0_sub_cols),# image_shape = (500, 3, 16, 16)
        filter_shape=(nkerns[0], num_channels, kernel0_size, kernel0_size),      # filter_shape= (20, 3, 5, 5)
        poolsize=(pool0_size, pool0_size),
        params_W=layer0_W
    )

    layer1 = MyNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], layer1_rows, layer1_cols),           # image_shape = (500, 20, 14, 14)
        filter_shape=(nkerns[1], nkerns[0], kernel1_size, kernel1_size),         # filter_shape= (50, 20, 5, 5)
        poolsize=(pool1_size, pool1_size),
        params_W=layer1_W
    )                                                                            # output size = (500, 50, 10, 10)
    layer1_sub = MyNetConvPoolLayer(
        rng,
        input=layer0_sub.output,
        image_shape=(batch_size, nkerns[0], layer1_sub_rows, layer1_sub_cols),   # image_shape = (500, 20, 6, 6)
        filter_shape=(nkerns[1], nkerns[0], kernel1_size, kernel1_size),         # filter_shape= (50, 20, 5, 5)
        poolsize=(pool1_size, pool1_size),
        params_W=layer1_W
    )                                                                            # output size = (500, 50, 2, 2)

    layer2_input = T.concatenate(
        [layer1.output.flatten(2), layer1_sub.output.flatten(2)], axis=1
    )
   
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1]*((layer1_rows+1-kernel1_size)*(layer1_cols+1-kernel1_size)+(layer1_sub_rows+1-kernel1_size)*(layer1_sub_cols+1-kernel1_size)), 
        n_out=hidden_nodes,
        activation=T.tanh
    )

    # two logistic regression does not share any parameters, so there is
    # no predefined parameters.
    layer3 = LogisticRegression(input=layer2.output, n_in=hidden_nodes, n_out=10)

    total_cost = layer3.negative_log_likelihood(y) + penalty_coeff * layer2.W.norm(2)

    params = layer3.params + layer2.params + layer1_sub.params + layer1.params + layer0_sub.params + layer0.params
    grad_l3     = T.grad(total_cost, layer3.params)
    grad_l2     = T.grad(total_cost, layer2.params)
    grad_l1_sub = T.grad(total_cost, layer1_sub.params)
    grad_l1     = T.grad(total_cost, layer1.params)
    grad_l0_sub	= T.grad(total_cost, layer0_sub.params)
    grad_l0     = T.grad(total_cost, layer0.params)

    updates = [
        (layer3.params[0]    , layer3.params[0]     - learning_rate * grad_l3[0]),
        (layer3.params[1]    , layer3.params[1]     - learning_rate * grad_l3[1]),
        (layer2.params[0]    , layer2.params[0]     - learning_rate * grad_l2[0]),
        (layer2.params[1]    , layer2.params[1]     - learning_rate * grad_l2[1]),
        (layer1_sub.params[0], layer1_sub.params[0] - learning_rate * (grad_l1_sub[0] + grad_l1[0])),
        (layer1_sub.params[1], layer1_sub.params[1] - learning_rate * (grad_l1_sub[1] + grad_l1[1])),
        (layer1.params[0]    , layer1.params[0]     - learning_rate * (grad_l1_sub[0] + grad_l1[0])),
        (layer1.params[1]    , layer1.params[1]     - learning_rate * (grad_l1_sub[1] + grad_l1[1])),
        (layer0_sub.params[0], layer0_sub.params[0] - learning_rate * (grad_l0_sub[0] + grad_l0[0])),
        (layer0_sub.params[1], layer0_sub.params[1] - learning_rate * (grad_l0_sub[1] + grad_l0[1])),
        (layer0.params[0]    , layer0.params[0]     - learning_rate * (grad_l0_sub[0] + grad_l0[0])),
        (layer0.params[1]    , layer0.params[1]     - learning_rate * (grad_l0_sub[1] + grad_l0[1]))
    ]

    training_index = T.iscalar()
    validate_index = T.iscalar()

    train_model = theano.function(
        [training_index],
        [total_cost, layer3.errors(y), layer0.output],
        updates=updates,
        givens={
            x : shared_x[training_index * batch_size : (training_index+1) * batch_size],
            y : shared_y[training_index * batch_size : (training_index+1) * batch_size]
        }
    )

    test_model = theano.function(
        [],
        layer3.errors(y),
        givens={
            x: evalset_x,
            y: evalset_y
        }
    )

    train_validation = theano.function(
        [],
        [total_cost, layer3.errors(y)],
        updates=updates,
        givens={
            x: evalset_x,
            y: evalset_y
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    patience = 10000
    epoch = 0
    done_looping = False

    # save parameters every 1000 iterations.
    param_files = ['p0', 'p1', 'p2', 'p3', 'p4',
                   'p5', 'p6', 'p7', 'p8', 'p9']
    while(epoch < n_epochs) and (not done_looping):
        batch_index = randint(0, 4)                                # randomly generate the batch number to be trained.
        cost_ij, error, output_check = train_model(batch_index)
        epoch = epoch + 1
        print "number of iterations:   ", epoch 
        print "selected training batch:", batch_index
        print "current cost:           ", cost_ij
        print "validate error:         ", error

        if (epoch % 10 == 0):
            error_test = test_model()
            print "      "
            print "validate error of test_batch:", error_test
            print "      "

        if (epoch % 1000 == 0):
            numpy.set_printoptions(threshold=numpy.nan)
#            print output_check 
            f = file(param_files[epoch/1000],'wb')
            cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

if __name__ == '__main__':
    dsetname = ['data_batch_1','data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'];  #dataset file names.
    train_cifar10('../data/cifar10/', dsetname)
