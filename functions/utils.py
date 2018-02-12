import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

'''
foreground_background_plot
plots overlapping histograms of the foreground and background of the images

parameters
__________
images : np.array
    the raw data
masks : np.array
    the masks

returns
__________
none
'''
def foreground_background_plot(images, masks):
    if (len(images) != len(masks)):
        raise ValueError("You must have the same number of images and masks.")
    plt.hist([images[i] * masks[i] for i in range(len(images))], bins = 25, label = 'Foreground')
    plt.hist([images[i] * (1 - masks[i].astype(int)) for i in range(len(images))], bins = 25, label = 'Background')
    plt.legend(loc='upper right')
    plt.show()

'''
load_data
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
'''
def load_data():
    # Set some parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3
    TRAIN_PATH = '../data/stage1_train/'
    TEST_PATH = '../data/stage1_test/'

    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = imresize(img, (IMG_HEIGHT, IMG_WIDTH))
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(imresize(mask_, (IMG_HEIGHT, IMG_WIDTH)), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = imresize(img, (IMG_HEIGHT, IMG_WIDTH))
        X_test[n] = img

    return X_train, Y_train, X_test
