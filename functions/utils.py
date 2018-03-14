import os
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.misc import imread
from skimage.transform import resize
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

from analytical import *


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
    y_true_in : np.array (2d)
        the true labels
    y_pred_in : np.array (2d)
        the predictions

    returns
    __________
    (precision, recall, f1) : tuple (double)
    """
    return precision_score(labels, predictions, average="weighted"), recall_score(labels, predictions, average="weighted"), f1_score(labels, predictions, average="weighted")


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

    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH))
    print("Getting and resizing train images and masks ... ")
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + "/images/" + id_ + ".png")[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode="constant", preserve_range=True)
        X_train[n] = img / 255.0
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + "/masks/"))[2]:
            mask_ = imread(path + "/masks/" + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode="constant",
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = np.squeeze(mask) / 255.0

    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    sizes_test = []
    print("Getting and resizing test images ... ")
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + "/images/" + id_ + ".png")[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode="constant", preserve_range=True)
        X_test[n] = img / 255.0

    print("Done!")

    return X_train, Y_train, X_test


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
    features_seperate = np.nan_to_num(
        [x.reshape((128*128, 9)) for x in X_train])
    flattened_features = []
    labels = []
    for i in range(0, len(X_train), skip):
        flattened_features.extend(features_seperate[i])
        labels.extend(Y_train[i].reshape((128*128)))
    return np.array(flattened_features), np.array(labels)


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
    flattened_features_by_image = np.nan_to_num(
        [x.reshape((128*128, 8)) for x in X_train])
    flattened_features = []
    for i in range(len(flattened_features_by_image)):
        flattened_features.extend(flattened_features_by_image[i])
    return flattened_features_by_image
