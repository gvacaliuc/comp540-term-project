from skimage.filters import threshold_otsu
import numpy as np
import cv2

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
