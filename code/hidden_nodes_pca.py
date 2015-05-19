import numpy
import pdb
from IO import unpickle
from IO import pickle

def mean_vec(mat):
    """ Function used to calculate the mean vector of
        an in put matrix.
        mat: input matrix with size n*m, representing
             m n-dim vectors.
        returns a mean vector of size n*1.
    """
    return numpy.mean(mat, axis=0)

def covariance(mat):
    """ Function used to calculate the covariance of
        an input matrix.
        mat: input matrix with size n*m, representing
             m n-dim vectors.
        returns the covariance matrix of mat.
    """
    mean_vector = mean_vec(mat)
    mat = mat - numpy.array([mean_vector])
    num_data = mat.shape[1]
    cov = 1. / (num_data - 1) * numpy.dot(mat.T, mat)
    return cov

def pca(mat, reduced_dim):
    """ Function used for principle component analysis.
        mat: input matrix with size n*m, representing
             m n-dim vectors.
        reduced_dim: dimension to be reserved after pca.
    """
    assert reduced_dim <= mat.shape[1]   # principle components should not 
                                         # be greater than the original 
    eig_val, eig_vec = numpy.linalg.eig(covariance(mat))
                                         # eigen values should be rearranged
    order = eig_val.argsort()[::-1]      # get the sorting order
    eig_val = eig_val[order]
    eig_vec = eig_vec[:, order]          # sort eigenvalues and eigenvectors

    transform = eig_vec[:, 0:reduced_dim].T
    
    return transform

if __name__ == '__main__':
    node_states = unpickle('hls')
    cov_mat = covariance(node_states)
    trans_mat = pca(cov_mat, 10)
    transformed_y = numpy.dot(node_states, trans_mat.T)
    pickle(trans_mat, 'transformation_matrix')
