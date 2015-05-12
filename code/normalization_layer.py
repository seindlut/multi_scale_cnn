import theano
import theano.tensor as T
import numpy

def mean_subtraction(data, data_shape):
    """ Function used for normalization by subtraction of
        mean value in the same spatial position.
        We use a 3D mean filter for implementation.
        data: input 4D theano.tensor
    """
    filter_shape = (data_shape[1], data_shape[1], 1, 1) 
    mean_filter = theano.shared(
        numpy.asarray(
            1./ data_shape[1] * numpy.ones(filter_shape),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    mean_tensor =  theano.tensor.nnet.conv.conv2d(
        input=data,
        filters=mean_filter,
        filter_shape=filter_shape,
        image_shape=data_shape
    )
    return (data - mean_tensor)

def local_responce_normalization(data, eps=0.001):
    """ Function used for local responce normalization. 
        data: input 4D theano.tensor
        eps: small constant in case the normalizer gets 0
    """
    normalizer = T.sqrt(eps + (data**2).sum(axis=1))
    return data / normalizer.dimshuffle(0,'x',1,2)

class NormalizationLayer(object):
    """ Class for normalization, use local responce normalization.
    """
    def __init__(self, data, eps=0.001):
        """ Constructor.
            data: 4D tensor as input from previous layer.
            eps:  small constant in case denominator is 0.
        """
        self.output = local_responce_normalization(data, eps) 
