"""
submission.py

Contains methods to process nuclei and individual nucleus predictions by
performing the requisite RLE encoding.
"""

import warnings

import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.base import BaseEstimator

from tqdm import tqdm

def rle_encode(mask, index_start=1):
    """
    Run Length Encodes a single, binary mask.

    :param mask: X x Y binary array
    :return: a numpy array holding our run length encoding.
    """

    pixels = mask.flatten(order = 'F')
    # We need to allow for cases where there is a '1' at either end of the
    # sequence.  We do this by padding with a zero at each end. 
    pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
    pixel_padded[1:-1] = pixels
    pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0]
    rle[1::2] = rle[1::2] - rle[::2]
    rle[::2] += index_start

    return rle


def encode_nuclei_mask(mask, return_string=True):
    """
    Performs the run length encoding for a given mask.  The mask is expected
    to be a single entry of the input to RLEncoder.fit.

    :param mask: an image of shape X x Y.  Discrete Valued.

    :return: a list of numpy arrays with the run-length encodings of
    every pixel in our mask.
    """
    return [(mask == ind).astype(mask.dtype) 
            for ind in range(1, int(mask.max() + 1))] 

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

        tups = zip(predictions, metadata.orig_shape, metadata.image_id)
        encodings = []
        for mask, orig_shape, image_id in tqdm(tups):
            resized_mask = resize(
                    mask, orig_shape, preserve_range=True, mode="constant")
            encoding_list = encode_nuclei_mask(resized_mask)
            check_encodings(encoding_list, orig_shape)
            encodings.extend([(image_id, enc) 
                              for enc in encoding_list])

        self.encoding_ = pd.DataFrame(
                encodings, columns=["ImageId", "EncodedPixels"])

        return self

    def fit_transform(self, metadata, predictions):
        """
        Fits the estimator and returns the encoding_ as expected by Kaggle.
        """

        return self.fit(metadata, predictions).encoding_


def _check_encoding(encoding, image_shape):
    """
    Checks a single encoding.
    """

    start_inds = encoding[::2]
    lengths = encoding[1::2]

    end_inds = start_inds + lengths

    if not (len(start_inds) == len(lengths)):
        raise ValueError("Odd length encoding.")

    #   That they're sorted
    if not (np.all(np.argsort(start_inds) == np.arange(len(start_inds)))):
        raise ValueError("Encoding isn't sorted.")

    #   That they're positive
    assert(np.all(start_inds > 0))
    assert(np.all(lengths > 0))
    if not (np.all(encoding > 0)):
        raise ValueError("Non-Positive values in encoding.")

    #   That they don't overlap, and have at least 1 pixel in between
    #   (otherwise they'd be joined)
    if not (np.all(end_inds[:-1] < start_inds[1:])):
        raise ValueError("Overlapping or contiguous encoding.")

    #   That it doesn't extend past the end
    if not ((end_inds[-1] - 1 ) <= np.prod(image_shape)):
        raise ValueError("Encoding out of image bounds.")


def check_encodings(encodings, image_shape):
    """
    Method to verify that our encoding satisfies kaggle's standards.
    Specifically, for each encoding in encodings, checks that:
        * pairs are sorted on start index
        * all numbers are positive
        * a given encoding doesn't overlap itself
        * all pairs specify pixels within the image boundary.

    It also checks that no encodings overlap each other.
    """

    mask_counts = np.zeros((len(encodings), np.prod(image_shape)))

    for ind, enc in enumerate(encodings):
        _check_encoding(enc, image_shape)
        for pair in zip(enc[::2], enc[1::2]):
            start, length = pair[0] - 1, pair[1]
            mask_counts[ind, start:start+length] += 1

    if np.max(mask_counts) > 1:
        raise ValueError("Some encodings overlap.")

    return mask_counts
