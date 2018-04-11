"""
submission.py

Contains methods to process nuclei and individual nucleus predictions by
performing the requisite RLE encoding.
"""

import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.base import BaseEstimator

def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def encode_mask(mask):
    """
    Performs the run length encoding for a given mask.  The mask is expected
    to be a single entry of the input to RLEncoder.fit.

    :param mask: an image of shape X x Y.  Discrete Valued.

    :return: a list of strings corresponding to the run length encoding of this
             mask.
    """
    nuclei_masks = [(mask == ind).astype(mask.dtype) for ind in range(1, int(mask.max() + 1))]
    return [rle_to_string(rle_encode(mask)) for mask in nuclei_masks]


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
        :param predictions: a numpy array of shape: N x X x Y. Each image in
                            predictions corresponds to a mask.  Each image
                            should be discrete valued, such that image == i
                            produces a binary mask of a single nucleus.
        :type predictions: numpy.ndarray
        """

        if (len(metadata) != len(predictions)):
            raise ValueError("""metadata and prediction list must be the same
                    length.""")

        encodings = [encode_mask(
            resize(mask_list, orig_shape, mode="constant", preserve_range=True))
            for mask_list, orig_shape in zip(predictions, metadata.orig_shape)]

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
