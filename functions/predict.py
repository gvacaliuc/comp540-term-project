import pickle

import numpy as np
from sklearn.externals import joblib


def lda_predict(image):
    """
    uses lda to segment an image

    parameters
    __________
    image : np.array (128 x 128 x 9)
        the input image

    returns
    __________
    np.array (128 x 128)
        the predictions
    """
    lda = joblib.load("../saved_models/sparse/lda.model")
    new_image = np.zeros((128, 128))
    for i in range(len(image)):
        for j in range(len(image[0])):
            new_image[i][j] = lda.predict_proba(
                image[i][j].reshape(1, -1))[0][1]
    return new_image


def lr_predict(image):
    """
    uses ElasticNet to segment an image

    parameters
    __________
    image : np.array (128 x 128 x 9)
        the input image

    returns
    __________
    np.array (128 x 128)
        the predictions
    """
    lr = joblib.load("../saved_models/white_background/lr.model")
    new_image = np.zeros((128, 128))
    for i in range(len(image)):
        for j in range(len(image[0])):
            new_image[i][j] = lr.predict(image[i][j].reshape(1, -1))
    return new_image

def svr_predict(image):
    """
    uses SVR to segment an image

    parameters
    __________
    image : np.array (128 x 128 x 9)
        the input image

    returns
    __________
    np.array (128 x 128)
        the predictions
    """
    svr = joblib.load("../saved_models/purple_foreground/SVR.model")
    new_image = np.zeros((128, 128))
    for i in range(len(image)):
        for j in range(len(image[0])):
            new_image[i][j] = svr.predict(image[i][j].reshape(1, -1))
    return new_image
