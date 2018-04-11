import inspect

import numpy as np
from scipy import ndimage
from skimage import feature, measure, morphology, segmentation
from sklearn.base import BaseEstimator, TransformerMixin
from .computer_vision import postprocess
import matplotlib.pyplot as plt


def watershed_cc(pred, nms_min_distance=3, watershed_line=True,
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

    dt = ndimage.distance_transform_edt(pred)
    peaks = feature.peak_local_max(
            dt,
            exclude_border = False,
            indices = False,
            min_distance = nms_min_distance)
    local_maxes = list(zip(*np.where(peaks == 1)))
    for coord1 in local_maxes:
        local_maxes.remove(coord1)
        for coord2 in local_maxes:
            if np.linalg.norm(np.array(coord1) - np.array(coord2)) < 10 and coord2 in local_maxes:
                local_maxes.remove(coord2)
                peaks[coord2] = 0
    markers = measure.label(peaks)
    seg = segmentation.watershed(-dt, markers, mask=pred,
                                 watershed_line=watershed_line)
    return seg



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

    def fit(self, images):
        """
        Segments each thresholded image in the list of images.
        """

        self.components_ = []
        for img in images:
            if (len(np.unique(img)) != 2):
                raise ValueError("Images must be thresholded already.")
            self.components_.append(
                    np.array(watershed_cc(postprocess(img),
                                 nms_min_distance=self.nms_min_distance,
                                 watershed_line=self.watershed_line)))

        return self

    def fit_transform(self, images):
        """
        Returns the fitted segmentations.
        """

        return self.fit(images).components_
