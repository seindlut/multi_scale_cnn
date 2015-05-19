import theano
import theano.tensor as T
import numpy

def init_Wb(Wshape, Wmu, Wsigma, bval):
    """ Function for initalization of weight
        and bias.
        Wshape : shape of weight
        Wmu    : mu of weight
        Wsigma : sigma of weight
        bval   : bias value
    """
    rng = numpy.random.RandomState(12345)
    W = numpy.asarray(rng.normal(Wmu, Wsigma, Wshape))
    b = bval * numpy.ones((Wshape[0],))
    return [W, b]
