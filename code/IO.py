import cPickle
import sys
import os
import pdb
import numpy
import theano
import theano.tensor as T

def share_data(data_x, data_y, borrow=True):
    # Function used to convert array and list into shared variables
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def unpickle(file):
    # Function to extract pickle files
    # extract pickle files
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
