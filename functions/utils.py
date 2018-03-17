import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from skimage.io import ImageCollection, imread
from skimage.transform import resize
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

from analytical import *
from computer_vision import preprocess_image

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(tf.convert_to_tensor(y_true),
                                            tf.convert_to_tensor(y_pred),
                                            2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0).eval(session=K.get_session())


def precision_recall_f1(labels, predictions):
    """
    calculates precision, recall, and f1 metrics

    parameters
    __________
    labels : np.array (2d)
        the true labels
    predictions : np.array (2d)
        the predictions

    returns
    __________
    (precision, recall, f1) : tuple (double)
    """
    return (precision_score(labels, predictions, average="weighted"),
            recall_score(labels, predictions, average="weighted"),
            f1_score(labels, predictions, average="weighted"))


def load_data(TRAIN_PATH="../data/stage1_train/",
              TEST_PATH="../data/stage1_test/"):
    """
    loads the training features, the training labels, and the test features

    parameters
    __________
    none

    return
    __________
    X_train : np.array
        the features of the training set
    Y_train : np.array
        the labels of the training set
    X_test : np.array
        the features of the test set
    """
    # Set some parameters
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    train_reader = DataReader(TRAIN_PATH, imsize = (IMG_HEIGHT, IMG_WIDTH),
                              num_channels = IMG_CHANNELS, scale = True)
    test_reader = DataReader(TEST_PATH, train = False,
                             imsize = (IMG_HEIGHT, IMG_WIDTH),
                             num_channels = IMG_CHANNELS, scale = True)

    print("Getting and resizing train images and masks ... ")
    X_train = train_reader.as_matrix()
    Y_train = np.stack(train_reader.masks)

    print("Getting and resizing test images ... ")
    X_test = test_reader.as_matrix()

    print("Done!")
    return X_train, Y_train, X_test


def flatten_data(data, labels = None, skip = 10):
    """
    Flattens our image matrices and masks into training pairs.
    """

    num_features = data.shape[-1]

    data = np.nan_to_num(data[::skip]).reshape((-1, num_features))

    if labels is not None:
        labels = labels[::skip].reshape((-1, 1))
        return data, labels

    return data


def flatten_training_data(X_train, Y_train, skip=10):
    """
    inputs an N x X x Y x D feature training set and corresponding
    N x X x Y x 1 label training set and flattens them into a
    NXY x D feature set (a.k.a. a pixel-wise feature set)
    and a NXY x 1 label set

    parameters
    __________
    X_train : np.array of dimensions N x X x Y x D
        the training features
    Y_train : np.array of dimensions N x X x Y x 1
        the training labels
    skip : int
        the number of elements you want to skip in between selection

    return
    __________
    flattened_features : np.array of dimensions NXY x D
        the flattened features of the training set
    labels : np.array of dimensions NXY x 1
        the flattened labels of the training set

    """

    return flatten_data(X_train, Y_train, skip=skip)


def flatten_test_data(X_train):
    """
    inputs an N x X x Y x D feature training set flattens it into a
    NXY x D feature set (a.k.a. a pixel-wise feature set)

    parameters
    __________
    X_train : np.array of dimensions N x X x Y x D
        the training features

    return
    __________
    flattened_features : np.array of dimensions NXY x D
        the flattened features of the training set

    """

    return flatten_data(X_train, Y_train, skip=skip)


class DataReader(object):

    def __init__(self, directory, train = True, imsize = (128, 128),
                 num_channels = 3, scale = True):
        """
        Class to read in our training and testing data, resize it, and store
        some metadata, including the image id and original size.  If we need to
        change the preprocessing for the images, we can do so in the _process
        method.
        """

        data_pattern = os.path.join(directory, "**/images/*.png")

        self.directory = directory
        self.train = train
        self.imsize = imsize
        self.num_channels = num_channels
        self.scale = scale
        self.IMG_MAX = 255.0

        self.image_metadata = []
        self.masks = []
        self.metadata_columns = ["image_id", "orig_shape"]

        imloader = lambda f: self._imloader(f)
        self.data_ic = ImageCollection(data_pattern, load_func = imloader)

    def _imloader(self, filename):
        """
        Function to read, resize, and process an image.
        """

        image_id, _, _ = filename.lstrip(self.directory + "/").split("/")

        img = imread(filename)[:, :, :self.num_channels]
        orig_shape = img.shape[:2]
        img = self._process(img)

        self.image_metadata.append((image_id, orig_shape))

        #   Load training labels if we're loading a training dataset
        if self.train:
            mask = self._mask_loader(image_id)
            self.masks.append(mask)

        return img

    def _mask_loader(self, image_id):
        """
        Function to load masks of specific image.
        """

        mask_pattern = os.path.join(self.directory, image_id, "masks/*.png")
        ic = ImageCollection(mask_pattern)

        mask = np.zeros(self.imsize, dtype='uint8')
        for indiv_mask in ic:
            mask = np.maximum(mask, self._process(indiv_mask))

        return mask

    def _process(self, img):
        """
        Processes an image per our specifications.
        """

        return preprocess_image(img, self.imsize, self.scale)

    def get_metadata(self):
        """
        Returns a pandas dataframe of the current stored metadata.
        """

        if (len(self.image_metadata) == 0):
            raise Warning("Returning empty metadata.")

        return pd.DataFrame(
            self.image_metadata,
            columns = self.metadata_columns)

    def as_matrix(self, start = 0, end = None, skip = 1):
        """
        Returns a dense version of our training data as a matrix of shape
        (N, X, Y, D).  Clears out previously saved masks and metadata.

        The parameters start, end, and skip operate like in range.
        """

        self.image_metadata = []
        self.masks = []

        end = len(self.data_ic) if end is None else end

        return self.data_ic[start:end:skip].concatenate()

    def get(self, *args):
        """
        Returns (get_metadata(), as_matrix(*args)).
        """

        matrix = self.as_matrix(*args)
        return (self.get_metadata(), matrix)
