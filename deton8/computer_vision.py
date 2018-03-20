import numpy as np
from scipy.ndimage.measurements import label
from skimage.filters import threshold_otsu
from skimage.transform import resize
import matplotlib.pyplot as plt
IMG_MAX = 255.0

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


def NMCS(X, percent=.95, min_area=50):
    """
    performs non_max_suppression within connected components

    parameters
    __________
    X : np.array
        the image you are trying to suppress
    percentile : int
        the fraction of images you are accepting
    min_area : int
        the minimum area of a component you are accepting

    return
    __________
    predictions : np.array
        the thresholded image
    """
    otsu_predictions = otsu(X)
    labeled, num_labels = label(otsu_predictions)
    flattened_labels = labeled.reshape((256*256, 1))
    flattened_otsu_predictions = otsu_predictions.reshape((256*256, 1))
    for label_num in range(1, num_labels + 1):
        indices = np.where(flattened_labels == label_num)[0]
        component = flattened_otsu_predictions[indices]
        val = np.percentile(component, 100*(1 - percent))
        if (len(indices) > min_area):
            flattened_otsu_predictions[indices] = np.multiply(
                component, component >= val)
        else:
            flattened_otsu_predictions[indices] = 0
    return flattened_otsu_predictions.reshape((256, 256))


def preprocess_image(img, imsize = (256, 256), scale = True):
    """
    Processes an image per our specifications.
    """

    WHITE_THRESHOLD = 0.55

    img = resize(img, imsize, mode="constant", preserve_range=True)
    img /= IMG_MAX if scale else 1

    img = (1 - img) if img.mean() > WHITE_THRESHOLD else img

    return img
