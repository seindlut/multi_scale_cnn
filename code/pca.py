import numpy

def mean_vec(mat):
    """ Function used to calculate the mean vector of
        an in put matrix.
        mat: input matrix with size n*m, representing
             m n-dim vectors.
        returns a mean vector of size n*1.
    """
    return numpy.mean(mat, axis=1)

def covariance(mat):
    """ Function used to calculate the covariance of
        an input matrix.
        mat: input matrix with size n*m, representing
             m n-dim vectors.
        returns the covariance matrix of mat.
    """
    mean_vector = mean_vec(mat)
    mat = mat - numpy.array([mean_vector]).T
    num_data = mat.shape[1]
    cov = 1. / (num_data - 1) * numpy.dot(mat, mat.T)
    return cov
