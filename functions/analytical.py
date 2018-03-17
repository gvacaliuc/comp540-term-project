import warnings

import cv2
import numpy as np
from skimage import exposure


class BasisTransformer(object):

    def __init__(self, *args, **kwargs):
        """
        Class to transform our original RGB features into our hand designed
        features.
        """

        pass

    def transform(self, data):
        """
        Transforms the "data".

        :param data: array of shape N x X x Y x C
        :type data: ndarray
        """

        return np.stack([basis_map(im) for im in data])


def basis_map(image):
    """
    maps each pixel of an image to a set of features

    parameters
    __________
    image : np.array (height x width x channels)
        the input image

    returns
    __________
    np.array (height x width x 9)
        the image features
    """

    num_features = 9
    IMG_MAX = 255.0
    new_image = np.zeros((*image.shape[:2], num_features))

    mean = np.mean(image)
    std = np.std(image)


    # Diameter of each pixel neighborhood that is used during filtering.
    bilateral_d = 2
    # Filter sigma in the color space. A larger value of the parameter means
    # that farther colors within the pixel neighborhood (see sigmaSpace) will
    # be mixed together, resulting in larger areas of semi-equal color. 
    bilateral_sigma_color = 75
    # Filter sigma in the coordinate space. A larger value of the parameter
    # means that farther pixels will influence each other as long as their
    # colors are close enough (see sigmaColor ). When d>0, it specifies the
    # neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
    # to sigmaSpace.
    bilateral_sigma_space = 75

    bilateral = cv2.bilateralFilter(
            np.mean(image * IMG_MAX, axis=2).astype("uint8"),
            bilateral_d,
            bilateral_sigma_color,
            bilateral_sigma_space) / IMG_MAX
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = np.mean(
            exposure.rescale_intensity(image, in_range=(p2, p98)),
            axis=2)

    # Limit of the amplification of the adaptive histogram.
    equalize_hist_clip_limit = 0.03

    # TODO: This raisies a warning about precision loss, but I don't know why.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        equalize_hist = np.mean(
            exposure.equalize_adapthist(
                image, clip_limit=equalize_hist_clip_limit),
            axis=2)

    # Dialation region to consider.
    dialation_kernel = np.ones((5, 5))
    # Number of dialation iterations to run.
    dialation_iters = 2
    dilate = np.mean(cv2.dilate(image, 
                                dialation_kernel, 
                                iterations=dialation_iters), 
                     axis=2)

    # First 3 dimensions are original color space.
    new_image[:, :, :3] = image
    # 4th dimension is the deviation of a mean of a pixel from the image mean.
    new_image[:, :, 3] = (np.mean(image, axis = 2) - mean) / std
    # 5th dimension is the result of a bilateral filtering
    new_image[:, :, 4] = bilateral
    # 6th dimension is the rescaled image
    new_image[:, :, 5] = img_rescale
    # 7th dimension is the result of a adaptive histogram
    new_image[:, :, 6] = equalize_hist
    # 8th dimension is a dialation
    new_image[:, :, 7] = dilate

    for i in range(len(image)):
        for j in range(len(image)):
            lower_i = max(i - 1, 0)
            upper_i = min(i + 1, image.shape[0] - 1)
            lower_j = max(j - 1, 0)
            upper_j = min(j + 1, image.shape[1] - 1)
            neighborhood = new_image[lower_i:upper_i, lower_j:upper_j, :8]
            neighbor_dist = np.linalg.norm(neighborhood - new_image[i, j, :8],
                                           axis = 2)
            new_image[i, j, 8] = np.mean(neighbor_dist)

    return np.nan_to_num(new_image)
