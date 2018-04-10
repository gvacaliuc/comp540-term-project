"""
submission.py

Contains methods to process nuclei and individual nucleus predictions by
performing the requisite RLE encoding.
"""

import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.base import BaseEstimator


def rle_encode(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return list(zip(run_lengths[::2], run_lengths[1::2]))


def decode_rle(encoding, shape):
    """
    Decodes a run-length encoding into a mask, given a shape.  Used to test
    the encoding method.

    :param encoding: a list of tuples holding the run-length encoding
    :param shape: a tuple holding the height and with of our mask
    :return mask: the rebuilt mask
    """

    mask = np.zeros(shape).flatten()

    for tup in encoding:
        true_ind = tup[0] - 1
        mask[true_ind:(true_ind + tup[1])] = 1

    mask = mask.reshape(shape, order = "F")

    return mask


def encode_image_masks(nuclei_masks):
    """
    Performs the run length encoding on each mask of an iterable.
    """
    return [rle_encode(mask) for mask in nuclei_masks]


def resize_im_list(imlist, shape):
    """
    Resizes each image in a list of images.
    """
    return [resize(im, shape, mode="constant", preserve_range=True) > 0
            for im in imlist]


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

        encodings = [encode_image_masks(resize_im_list(mask_list, orig_shape))
                     for mask_list, orig_shape in zip(predictions,
                                                      metadata.orig_shape)]

        #   Turn encodings into strings
        rle_list_to_str_list = lambda lst: [" ".join(["{} {}".format(*tup)
                                                      for tup in code])
                                            for code in lst]
        encodings = [rle_list_to_str_list(rle_list) for rle_list in encodings]

        self.encoding_ = metadata.copy()
        self.encoding_["rle_mask_list"] = encodings

        return self

    def fit_transform(self, metadata, predictions):
        """
        Fits the estimator and returns the encoding_ as expected by Kaggle.
        """

        df = self.fit(metadata, predictions).encoding_

        items_as_cols = df.apply(lambda x: pd.Series(x['rle_mask_list']),
                                 axis=1)

        # Keep original df index as a column so it's retained after melt
        items_as_cols['orig_index'] = items_as_cols.index

        melted_items = pd.melt(items_as_cols, id_vars='orig_index',
                               var_name='sample_num', value_name='rle_mask')
        melted_items.set_index('orig_index', inplace=True)

        df = df.merge(melted_items, left_index=True, right_index=True).dropna()
        df = df[["image_id", "rle_mask"]]
        df = df.rename(columns={"image_id": "ImageId",
                                "rle_mask": "EncodedPixels"})

        return df
