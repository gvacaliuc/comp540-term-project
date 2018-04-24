import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import ImageCollection, imread
from skimage.transform import resize
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

from .processing import postprocess, preprocess_image
import inspect


class NucleiDataset(object):
    def __init__(self,
                 directory,
                 train=True,
                 imsize=(256, 256),
                 num_channels=3,
                 scale=True,
                 invert_white_images=True):
        """
        Class to read in our training and testing data, resize it, and store
        some metadata, including the image id and original size.  If we need to
        change the preprocessing for the images, we can do so in the _process
        method.
        """

        #   Sets all attributes.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.IMG_MAX = 255.0

        data_pattern = os.path.join(directory, "**/images/*.png")

        self.metadata_ = []
        self.masks_ = []
        self.metadata_columns = ["image_id", "orig_shape"]

        self.data_ic_ = ImageCollection(data_pattern)

    def _load_image(self, filename):
        """
        Function to read, resize, and process an image.
        """

        path = filename.split("/")
        image_id = path[len(self.directory.split("/")) - 1]

        try:
            img = imread(filename)[:, :, :self.num_channels]
        except IndexError:
            tmp = imread(filename)
            img = np.stack([tmp] * 3).transpose(1, 2, 0)
        orig_shape = img.shape[:2]
        img = self._process(img)

        masks = np.zeros(self.imsize)

        #   Load training labels if we're loading a training dataset
        if self.train:
            masks = self._load_mask(image_id)

        return (img, masks, image_id, orig_shape)

    def _load_mask(self, image_id):
        """
        Function to load masks of specific image.
        """

        mask_pattern = os.path.join(self.directory, image_id, "masks/*.png")
        ic = ImageCollection(mask_pattern)

        mask = np.zeros(self.imsize, dtype='uint8')
        for lbl, indiv_mask in enumerate(ic):
            mask += ((
                1 + lbl) * self._process(indiv_mask, True).astype('uint8'))

        return mask

    def _process(self, img, mask=False):
        """
        Processes an image per our specifications.
        """

        if mask:
            return preprocess_image(img, self.imsize, False, False)

        return preprocess_image(img, self.imsize, self.scale,
                                self.invert_white_images)

    def load(self, max_size=None):
        """
        Loads in data up to the desired maximum amount.
        """

        max_size = max_size if max_size else np.inf
        num_images = min(max_size, len(self.data_ic_.files))

        metadata = []
        self.masks_ = np.zeros((num_images, *self.imsize))
        self.images_ = np.zeros((num_images, *self.imsize, self.num_channels))

        for im_num, filename in tqdm(
                enumerate(self.data_ic_.files[:num_images]), unit="image"):
            img, mask, image_id, orig_shape = self._load_image(filename)
            self.images_[im_num] = img
            self.masks_[im_num] = mask
            metadata.append((image_id, orig_shape))

        self.metadata_ = pd.DataFrame(metadata, columns=self.metadata_columns)

        return self


def flatten_data(X):
    """
    Flattens the data in X.

    :param X: ndarray of shape (N, X, Y, C)

    :return reshaped: ndarray of shape (N * X * Y, C)
    """

    return X.reshape((-1, X.shape[-1]))


def expand_data(X, orig_shape=(256, 256)):
    """
    Restructures the data in X as a set of images.

    :param X: ndarray of shape (N * X * Y, C)

    :return reshaped: ndarray of shape (N, X, Y, C)
    """

    return X.reshape((-1, *orig_shape, X.shape[-1]))


def get_model_results_global_thresh(model, data, labels):
    """
    Data and Labels should be the flattened data / labels.
    """

    results = []
    columns = ["percent_thresh", "mask_thresh", "f1_score"]

    if len(data.shape) >= 4:
        labels = labels.reshape((len(labels) * 256 * 256))
    for percent_thresh in np.arange(.9, 1, 0.015):
        for mask_thresh in np.arange(.3, .7, 0.15):
            y_pred = model.predict(data)
            if len(y_pred.shape) < 4:
                y_pred = np.array([postprocess(y_pred[256*256*i:256*256*(i+1)].reshape((256, 256)), \
                               percent=percent_thresh)
                                for i in range(int(len(y_pred) / (256*256)))])

            else:
                y_pred = np.array([
                    postprocess(y_pred[i], percent=percent_thresh)
                    for i in range(int(len(y_pred)))
                ])
            y_pred = y_pred.reshape((len(y_pred) * 256 * 256))
            results.append((percent_thresh, mask_thresh,
                            f1_score(labels > mask_thresh, y_pred > 0)))

    return pd.DataFrame(results, columns=columns)


def f1(predictions, labels):
    return np.mean(
        f1_score((np.array(labels).astype('uint8') > 0).flatten(),
                 (np.array(predictions).astype('uint8') > .5).flatten()))
