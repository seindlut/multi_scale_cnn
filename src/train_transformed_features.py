import numpy
import theano
import theano.tensor as T

from layer_logistic_regression import LogisticRegression
from IO import unpickle
from IO import pickle
from IO import load_cifar10_train_labels
from IO import share_data

def train_transformed_features(num_iterations=2500, lr=0.01):
    """ Function for training transformed features """
    # initialize test parameters
    num_examples = 50000
    num_dimensions = 10   
    num_classes = 10

    # load transformed features
    train_x = unpickle('transformed_features')
    path = '../data/cifar10/'
    fnames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    train_y = load_cifar10_train_labels(path, fnames)

    train_x, train_y = share_data(train_x, train_y)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    x = T.matrix('x')
    y = T.ivector('y')    
    classifier_input = x
    class_layer = LogisticRegression(
        input=classifier_input,
        n_in=num_dimensions,
        n_out=num_classes,
    )

    cost = class_layer.negative_log_likelihood(y)
    errors = class_layer.errors(y)
    
    grads = T.grad(cost, class_layer.params)
    updates = [(class_layer.W, class_layer.W - lr * grads[0]),
               (class_layer.b, class_layer.b - lr * grads[1])]

    test_model = theano.function(
        [],
        [cost, errors],
        updates=updates,
        givens={
            x: train_x,
            y: train_y
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    iteration = 0
    while(iteration < num_iterations):
        iteration += 1
        p_cost, p_error = test_model()
        print 'cost:  ', p_cost
        print 'error: ', p_error

        if(iteration == num_iterations):
            pickle(class_layer.params, 'parameters_trans')
if __name__ == '__main__':
    train_transformed_features()        
