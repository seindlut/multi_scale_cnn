import theano.tensor as T

def relu(x):
    """ Function for rectified linear unit.
        Returns the maximum value of input and 0.
    """
    return T.switch(x<0, 0, x)
