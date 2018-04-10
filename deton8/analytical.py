import warnings
import inspect

import cv2
import numpy as np
from skimage import exposure
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin


class BasisTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 bilateral_d=2,
                 bilateral_sigma_color=75,
                 bilateral_sigma_space=75,
                 equalize_hist_clip_limit=0.03,
                 dialation_kernel=np.ones((5, 5)),
                 dialation_iters=2):
        """
        :param bilateral_d: Diameter of each pixel neighborhood that is used
        during bilateral filtering.

        :param bilateral_sigma_color: Filter sigma in the color space. A larger
        value of the parameter means that farther colors within the pixel
        neighborhood (see sigmaSpace) will be mixed together, resulting in
        larger areas of semi-equal color.

        :param bilateral_sigma_space: Filter sigma in the coordinate space. A
        larger value of the parameter means that farther pixels will influence
        each other as long as their colors are close enough (see sigmaColor ).
        When d>0, it specifies the neighborhood size regardless of sigmaSpace.
        Otherwise, d is proportional to sigmaSpace.

        :param equalize_hist_clip_limit: Limit of the amplification of the
        adaptive histogram.

        :param dialation_kernel: Dialation region to consider.

        :param dialation_iters: Number of dialation iterations to run.
        """

        #   Sets all attributes.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.set_params(**values)

    def fit(self, images):
        """
        Takes an ndarray of images and increases the number of channels by
        performing our basis map to each image.

        :param images: an array of shape N x X x Y x C

        :return self: the fitted estimator
        """
        self.features_ = []
        for i in tqdm(range(len(images)), unit="pair"):
            self.features_.append(self._basis_map(images[i]))


        self.features_ = np.stack(self.features_)

        return self

    def fit_transform(self, images):
        """
        Takes an ndarray of images and increases the number of channels by
        performing our basis map to each image.

        :param images: an array of shape N x X x Y x C

        :return self.features_: the transformed data
        """

        return self.fit(images).features_

    def _basis_map(self, image):
        """
        Maps each pixel of an image to a set of features.

        :param image: the input image (height x width x channels)

        The rest of the parameters are documented in BasisTransformer.

        :return: the image features
        """

        num_features = 6
        IMG_MAX = 255.0
        new_image = np.zeros((*image.shape[:2], num_features))

        mean = np.mean(image)
        std = np.std(image)

        bilateral = cv2.bilateralFilter(
                np.mean(image * IMG_MAX, axis=2).astype("uint8"),
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space) / IMG_MAX
        p2, p98 = np.percentile(image, (2, 98))
        img_rescale = np.mean(
                exposure.rescale_intensity(image, in_range=(p2, p98)),
                axis=2)

        # TODO: This raisies a warning about precision loss, but idk why.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            equalize_hist = np.mean(
                exposure.equalize_adapthist(
                    image, clip_limit=self.equalize_hist_clip_limit),
                axis=2)

        dilate = np.mean(cv2.dilate(image,
                                    self.dialation_kernel,
                                    iterations=self.dialation_iters),
                         axis=2)

        # First 3 dimensions are original color space.
        new_image[:, :, 0] = image
        # 4th dimension is the pixels z-score.
        new_image[:, :, 1] = (np.mean(image, axis = 2) - mean) / std
        # 5th dimension is the result of a bilateral filtering
        new_image[:, :, 2] = bilateral
        # 6th dimension is the rescaled image
        new_image[:, :, 3] = img_rescale
        # 7th dimension is the result of a adaptive histogram
        new_image[:, :, 4] = equalize_hist
        # 8th dimension is a dialation
        new_image[:, :, 5] = dilate

        return np.nan_to_num(new_image)
