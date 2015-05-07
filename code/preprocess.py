import numpy

def normalize(data, scale=255.0):
    """ Function used to normalize data, i.e.
        to divide the data by a certain factor. 
        data: numpy.ndarray.
        scale: scaling factor to be divided. """

    return data / scale  

def mean_subtraction(data):
    """ Function used to subtract the mean image
        of the dataset.
        data: numpy array of size n*m, representing
              n m-dim row vectors of image.
    """
    return (data - data.mean(axis=0))
