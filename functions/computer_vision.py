from skimage.filters import threshold_otsu
from scipy.ndimage.measurements import label
import numpy as np


def otsu(image):
    """
    performs otsu's binarization on an image

    parameters
    __________
    X : np.array
        the image you are trying to predict

    return
    __________
    predictions : np.array
        the thresholded image
    """
    thresh_val = threshold_otsu(image)
    mask = np.array(image > thresh_val)
    return np.multiply(image, mask)


def non_max_component_suppression(X, percent=99, min_area=0):
    '''
    performs non_max_suppression within connected components

    parameters
    __________
    X : np.array
        the image you are trying to suppress
    percentile : int
        the percent of images you are accepting
    min_area : int
        the minimum area of a component you are accepting

    return
    __________
    predictions : np.array
        the thresholded image
    '''
    otsu_predictions = otsu(X)
    labeled, num_labels = label(otsu_predictions)
    flattened_labels = labeled.reshape((128*128, 1))
    flattened_otsu_predictions = otsu_predictions.reshape((128*128, 1))
    for label_num in range(1, num_labels + 1):
        indices = np.where(flattened_labels == label_num)[0]
        component = flattened_otsu_predictions[indices]
        val = np.percentile(component, 100 - percent)
        if (len(indices) > min_area):
            flattened_otsu_predictions[indices] = np.multiply(
                component, component >= val)
        else:
            flattened_otsu_predictions[indices] = 0
    return flattened_otsu_predictions.reshape((128, 128))
