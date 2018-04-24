import inspect

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import feature, measure, morphology, segmentation
from sklearn.base import BaseEstimator, TransformerMixin

from .computer_vision import postprocess


def watershed_cc(pred, original, nms_min_distance=1, watershed_line=True,
                 return_mask=False):
    """
    Finds a set of components believed to be individual nuclei using the
    watershed segmentation algorithm. Works quite well, even if we have
    several nuclei grouped together.

    :param pred: our preprocessed predictions
    :param nms_min_distance: the minimum distance between two peaks in the
                             nms
    :param return_mask: whether or not to return the segmentation mask

    :return: a list of binary arrays masking individual components, as well
             as a full mask w/ unique integers indicating components if
             return_mask is True.
    """
    im = np.multiply(pred, original)
    dt = ndimage.distance_transform_edt(pred)
    peaks = feature.peak_local_max(
            dt,
            exclude_border = False,
            indices = False,
            min_distance = 5)
    local_maxes = list(zip(*np.where(peaks == True)))
    for coord1 in local_maxes.copy():
        for coord2 in local_maxes.copy():
            if np.linalg.norm(np.array(coord1) - np.array(coord2)) < 30 and np.linalg.norm(im[coord1] - im[coord2]) < .1 and coord1 in local_maxes and not coord1 == coord2:
                local_maxes.remove(coord2)
                peaks[coord2] = 0
    markers = measure.label(peaks)
    seg = segmentation.watershed(-dt, markers, mask=pred,
                                    watershed_line=True)
    return seg.astype(np.int32)


class NucleiSegmenter(BaseEstimator, TransformerMixin):
    """
    Class to perform our nuclei segmentation given a set of thresholded images.
    """

    def __init__(self, nms_min_distance=3, watershed_line=True):
        """
        Creates a NucleiSegmenter.

        :param nms_min_distance: Indicates the minimum distance that any two
        nuclei predictions must be from each other.
        """

        #   Sets all attributes.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.set_params(**values)

    def fit(self, predictions, preprocessed):
        """
        Segments each thresholded image in the list of images.

        :param predictions: numpy array of shape N x X x Y, holding our nuclei
        prediction for each pixel
        :param preprocessed: numpy array of shape N x X x Y, holding the
        preprocessed images.

        :return: numpy array of shape N x X x Y, however each pixel in each
        image will be set to a discrete value 0, ..., n.  0 indicates
        background.
        """

        components = np.zeros_like(predictions, dtype=np.int32)
        if not predictions.shape == preprocessed.shape:
            raise ValueError("Arrays must be of the same shape.")

        if len(np.unique(predictions)) != 2:
            raise ValueError("Images must be thresholded already.")

        for ind, (mask, orig) in enumerate(zip(predictions, preprocessed)):
            components[ind] = watershed_cc(mask, orig,
                    nms_min_distance=self.nms_min_distance,
                    watershed_line=self.watershed_line)

        self.components_ = components

        return self

    def fit_transform(self, predictions, preprocessed):
        """
        Returns the fitted segmentations.
        """

        return self.fit(predictions, preprocessed).components_
