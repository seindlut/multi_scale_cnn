import numpy

def normalize(data, scale=255.0):
    """ Function used to normalize data, i.e.
        to divide the data by a certain factor. 
        data: numpy.ndarray.
        scale: scaling factor to be divided. """

    return data / scale  
