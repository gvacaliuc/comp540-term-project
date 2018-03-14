import cv2
import numpy as np
from skimage.filters import threshold_otsu


def otsu(image):
    '''
    performs otsu's binarization on an image

    parameters
    __________
    X : np.array
        the image you are trying to predict

    return
    __________
    predictions : np.array
        the thresholded image
    '''
    thresh_val = threshold_otsu(image)
    mask = np.array(image > .5)
    return np.multiply(image, mask)
