import theano
import theano.tensor as T
import numpy

from utils import crop_images
from utils import mean_subtraction_preprocessing
class PreprocessingLayer(object):
    """ Class for preprocessing of data. """
    def __init__(self, input_data):
        self.data = input_data
    def subtract_mean(self):
    def crop_smaller(self):
