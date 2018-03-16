import numpy as np
from scipy import ndimage
from skimage import feature, measure, segmentation


def preprocess(pred):
    """
    Preprocesses our predictions to attempt to smooth and restore 
    the nuclei shape.
    """

    return pred

def watershed_cc(pred):
    """
    Finds a set of components believed to be individual nuclei using the 
    watershed segmentation algorithm.

    :param pred: our preprocessed predictions
    """

    dt = ndimage.distance_transform_edt(pred)
    peaks = feature.peak_local_max(dt, exclude_border = False, indices = False)
    markers = measure.label(peaks)
    seg = segmentation.watershed(-dt, markers, mask = pred)

    #   get a component for everything but background
    ccs = [seg == lbl for lbl in np.unique(seg)[1:]]

    return ccs

def measure_label_cc(pred):
    """
    Finds a set of components believed to be individual nuclei simply by
    labeling connected components.

    :param pred: our preprocessed predictions
    """

    seg = measure.label(pred)

    #   get a component for everything but background
    ccs = [seg == lbl for lbl in np.unique(seg)[1:]]

    return ccs
