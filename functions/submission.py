"""
submission.py

Contains methods to process nuclei and individual nucleus predictions by
performing the requisite RLE encoding.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def encode_rle_single_mask(nucleus_mask):
    """
    Run length encodes a single mask.

    :param nucleus_mask: a binary ndarray masking a single nucleus in the image
    :return rle: a list of tuples holding the run length encoding of a
                 mask using a flattened 1-indexed coding
    """

    #   must use fortran style arrays due to kaggle requirements
    flat_bw_mask = np.array(nucleus_mask > 0,
                            dtype = "int64").flatten(order = "F")

    nonzero_ind  = np.nonzero(flat_bw_mask)[0]
    diff = np.diff(nonzero_ind)
    runstarts = np.hstack([[0], np.nonzero(diff != 1)[0] + 1])
    rle = [(1 + nonzero_ind[start], nextstart - start)
           for (start, nextstart) in zip(runstarts[:-1], runstarts[1:])]
    last_start = runstarts[-1]
    rle += [(1 + nonzero_ind[last_start], len(nonzero_ind) - last_start)]

    return rle


def decode_rle(encoding, shape):
    """
    Decodes a run-length encoding into a mask, given a shape.  Used to test
    the encoding method.

    :param encoding: a list of tuples holding the run-length encoding
    :param shape: a tuple holding the height and with of our mask
    :return mask: the rebuilt mask
    """

    mask = np.zeros(shape).flatten()

    for tup in rle_encoding:
        true_ind = tup[0] - 1
        mask[true_ind:(true_ind + tup[1])] = 1

    mask = mask.reshape(shape, order = "F")

    return mask


def encode_image_masks(nuclei_masks):
    """
    Performs the run length encoding on each mask of an iterable.
    """

    return [encode_rle_single_mask(mask) for mask in nuclei_masks]


class RLEncoder(BaseEstimator):
    """
    Class to run length encode a set of image predictions.
    """

    def __init__(self):
        super()

    def fit(self, metadata, predictions):
        """
        Fits the encoder with the given metadata and predicted binary masks.

        :param metadata: a dataframe with two columns per image: image_id and
                         orig_shape
        :type metadata: pandas.DataFrame
        :param predictions: a list of lists of binary 2D numpy arrays which
                            have already been reshapen according to the
                            original shape in the metadata.  Each sublist
                            indicates a list of nuclei predictions.  Each array
                            in the sublist is an individual nucleus mask.
        :type predictions: iterable
        """

        if (len(metadata) != len(predictions)):
            raise ValueError("""metadata and prediction list must be the same
                    length.""")

        encodings = [encode_image_masks(mask_list) for mask_list in predictions]

        #   Turn encodings into strings
        rle_list_to_str_list = lambda lst: [" ".join(["{} {}".format(*tup)
                                                      for tup in code])
                                            for code in lst]
        encodings = [rle_list_to_str_list(rle_list) for rle_list in encodings]

        self.encoding_ = metadata.copy()
        self.encoding_["rle_mask_list"] = encodings
