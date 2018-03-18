import numpy as np
from scipy import ndimage
from skimage import feature, measure, morphology, segmentation


def component_preprocesser(pred):
    """
    Preprocesses our predictions to attempt to smooth and restore
    the nuclei shape as well as rid ourselves of undesirable salt.

    :param pred: a binary array of our predictions
    :type pred: np.ndarray

    :return: our preprocessed predictions
    """

    return morphology.binary_opening(pred)


def watershed_cc(pred, nms_min_distance=3, return_mask=False):
    """
    Finds a set of components believed to be individual nuclei using the
    watershed segmentation algorithm. Works quite well, even if we have
    several nuclei grouped together.

    :param pred: our preprocessed predictions
    :param nms_min_distance: the minimum distance between two peaks in the nms
    :param return_mask: whether or not to return the segmentation mask

    :return: a list of binary arrays masking individual components, as well as
             a full mask w/ unique integers indicating components if return_mask
             is True.
    """

    dt = ndimage.distance_transform_edt(pred)
    peaks = feature.peak_local_max(
            dt,
            exclude_border = False,
            indices = False,
            min_distance = nms_min_distance)
    markers = measure.label(peaks)
    seg = segmentation.watershed(-dt, markers, mask=pred, watershed_line=True)

    #   get a component for everything but background
    ccs = [seg == lbl for lbl in np.unique(seg)[1:]]

    if return_mask:
        return ccs, seg
    else:
        return ccs
