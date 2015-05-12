import theano
import theano.tensor as T
import numpy

from utils import local_responce_normalization

class NormalizationLayer(object):
    """ Class for normalization, use local responce normalization.
    """
    def __init__(self, data, eps=0.001):
        """ Constructor.
            data: 4D tensor as input from previous layer.
            eps:  small constant in case denominator is 0.
        """
        self.output = local_responce_normalization(data, eps) 
