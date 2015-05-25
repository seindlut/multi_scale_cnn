import setup
import pdb

from src.IO import load_stl10_train


def train_stl10(file_name):
    """ Function for training stl10 dataset """
    train_x, train_y = load_stl10_train(file_name)
    pdb.set_trace()

if __name__ == '__main__':
    path = '../data/stl10/'
    fname = 'train'
    train_stl10(path+fname)
