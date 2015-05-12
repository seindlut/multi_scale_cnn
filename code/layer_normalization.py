import theano
import theano.tensor as T
import numpy

from utils import local_responce_normalization_

class NormalizationLayer(object):
    """ Class for normalization, use local responce normalization.
    """
    def __init__(self, data, k=2, n=5, alpha=0.0001, beta=0.75):
        """ Constructor.
            data: 4D tensor as input from previous layer.
            eps:  small constant in case denominator is 0.
        """
        self.output = local_responce_normalization_(data, k, n, alpha, beta) 
