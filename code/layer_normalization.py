import theano
import theano.tensor as T
import numpy

from utils import local_responce_normalization_across

class NormalizationLayer(object):
    """ Class for normalization, use local responce normalization.
    """
    def __init__(self, data, n=5, alpha=0.0001, beta=0.75, mode=0):
        """ Constructor.
            data  : 4D tensor as input from previous layer.
            n     : size of receptive field
            alpha : addition coefficient
            beta  : exponential term
            mode  : 0 for across channel, non-zero for within channel
        """
        if (mode == 0):
            self.output = local_responce_normalization_across(data, n, alpha, beta) 
        else:
            self.output = local_responce_normalization_within(data, n, alpha, beta)
