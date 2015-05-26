import setup
import numpy
import theano
import theano.tensor as T
import cPickle
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import hard_sigmoid
from random import randint

from src.IO import * 
from src.utils import *
from src.layer_conv_pool import ConvPoolLayer
from src.layer_normalization import NormalizationLayer
from src.layer_multi_layer_perceptron import HiddenLayer, DropoutHiddenLayer
from src.layer_logistic_regression import LogisticRegression

def train_stl10(file_name, batch_size=1000,
                n_epochs=10000, nkerns=[32,32,64],
                learning_rate=0.0005):
    """ Function for training stl10 dataset """
    # define layer parameters
    rng = numpy.random.RandomState(23455)
    num_images = 5000
    num_test_images = 8000 
    num_batches = num_images / batch_size
    num_test_batches = num_test_images / batch_size
    num_channels = 3 
    original_image_rows = 96
    original_image_cols = 96
    original_image_column_width = original_image_rows * original_image_cols * num_channels

    # convolutional layer 0 parameters #
    conv_layer0_rows = original_image_rows                   
    conv_layer0_cols = original_image_cols                   
    conv_layer0_pixels = conv_layer0_rows * conv_layer0_cols 
    conv_layer0_column_width = conv_layer0_pixels * num_channels
    conv_layer0_kernel_size = 7                              
    conv_layer0_pool_size = 3 

    # convolutional layer 1 parameters #
    conv_layer1_rows = (conv_layer0_rows - conv_layer0_kernel_size + 1) / conv_layer0_pool_size   
    conv_layer1_cols = (conv_layer0_cols - conv_layer0_kernel_size + 1) / conv_layer0_pool_size   
    conv_layer1_kernel_size = 7 
    conv_layer1_pool_size = 2
                                                                         
    # convolutional layer 2 parameters #
    conv_layer2_rows = (conv_layer1_rows - conv_layer1_kernel_size + 1) / conv_layer1_pool_size   
    conv_layer2_cols = (conv_layer1_cols - conv_layer1_kernel_size + 1) / conv_layer1_pool_size   
    conv_layer2_kernel_size = 5 
    conv_layer2_pool_size = 2 

    # output rows and columns of convolutional net #
    conv_output_rows = (conv_layer2_rows - conv_layer2_kernel_size + 1) / conv_layer2_pool_size   
    conv_output_cols = (conv_layer2_cols - conv_layer2_kernel_size + 1) / conv_layer2_pool_size   
    
    # fully connected layer parameters #
    fc_layer0_hidden_nodes = 64

    # optimization parameters
    momentum_coeff = 0.9
    weight_decay = 0.0001
    penalty_coeff = 0.01 

    # load and preprocess
    train_x, train_y = load_stl10_train(file_name)
    train_x = mean_subtraction_preprocessing(train_x)
    train_x = unit_scaling(train_x)
    train_x, train_y = share_data(train_x, train_y) 

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
        sigma=0.1,
        bias_val=0,
        activation_mode=1
    )

    conv_layer1 = ConvPoolLayer(
        rng,
        input=conv_layer0.output,
        image_shape=(batch_size, nkerns[0], conv_layer1_rows, conv_layer1_cols),    
        filter_shape=(nkerns[1], nkerns[0], conv_layer1_kernel_size, conv_layer1_kernel_size), 
        poolsize=(conv_layer1_pool_size, conv_layer1_pool_size),
        sigma=0.1,
        bias_val=0,
        activation_mode=1
    ) 

    conv_layer2 = ConvPoolLayer(
        rng,
        input=conv_layer1.output,
        image_shape=(batch_size, nkerns[1], conv_layer2_rows, conv_layer2_cols),
        filter_shape=(nkerns[2], nkerns[1], conv_layer2_kernel_size, conv_layer2_kernel_size),
        poolsize=(conv_layer2_pool_size, conv_layer2_pool_size),
        sigma=0.1,
        bias_val=0,
        activation_mode=1
    )


    fc_layer0_input = conv_layer2.output.flatten(2)
    fc_layer0 = HiddenLayer(
        rng,
        input=fc_layer0_input,
	    n_in=nkerns[2] * conv_output_rows * conv_output_cols, 
        n_out=fc_layer0_hidden_nodes,
        bias_val=0,
        activation=relu
    )    

    class_layer0 = LogisticRegression(input=fc_layer0.output, n_in=fc_layer0_hidden_nodes, n_out=10)

    # compare the difference between regularization of hidden layer weights and classifier weights.
    total_cost = class_layer0.negative_log_likelihood(y) + penalty_coeff * class_layer0.W.norm(2)

    params = conv_layer0.params + conv_layer1.params + conv_layer2.params + fc_layer0.params + class_layer0.params

    grad_classl0     = T.grad(total_cost, class_layer0.params)
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

    epoch=0
    while(epoch < n_epochs):
        batch_index = randint(0, num_batches-1)
        print "selected training batch:", batch_index
        cost_ij, error = train_model(batch_index)
        epoch = epoch + 1
        print "number of iterations:   ", epoch 
        print "current cost:           ", cost_ij
        print "training error:         ", error

    file_to_save = open('parameters','wb')
    cPickle.dump(params, file_to_save)
    file_to_save.close()
    

if __name__ == '__main__':
    path = '../data/stl10/'
    fname = 'train'
    train_stl10(path+fname)
