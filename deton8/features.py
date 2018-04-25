import inspect
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.morphology import disk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm


class BasisTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 bilateral_d=2,
                 bilateral_sigma_color=75,
                 bilateral_sigma_space=75,
                 rescale_intensity_range=(50, 99),
                 equalize_hist_clip_limit=0.03,
                 dialation_kernel=disk(radius=3),
                 dialation_iters=1):
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

    def fit(self, images, y=None):
        """
        Takes an ndarray of images and increases the number of channels by
        performing our basis map to each image.

        :param images: an array of shape N x X x Y x 1

        :return self: the fitted estimator
        """
        self.features_ = []
        for i in tqdm(range(len(images)), unit="pair"):
            self.features_.append(self._basis_map(images[i]))

        self.features_ = np.stack(self.features_)

        return self

    def transform(self, images, y=None):
        """
        Takes an ndarray of images and increases the number of channels by
        performing our basis map to each image.

        :param images: an array of shape N x X x Y x 1

        :return self.features_: the transformed data
        """

        return self.fit(images).features_

    def fit_transform(self, images, y=None):
        """
        Takes an ndarray of images and increases the number of channels by
        performing our basis map to each image.

        :param images: an array of shape N x X x Y x 1

        :return self.features_: the transformed data
        """

        return self.fit(images).features_

    def _basis_map(self, image):
        """
        Maps each pixel of an image to a set of features.  Input image should
        be grayscale.

        :param image: the input image (height x width x 1)

        The rest of the parameters are documented in BasisTransformer.

        :return: the image features
        """

        num_features = 6
        IMG_MAX = 255.0

        image = image[:, :, 0]
        uint8_image = (image * IMG_MAX).astype('uint8')

        new_image = np.zeros((*image.shape[:2], num_features))

        mean = np.mean(image)
        std = np.std(image)

        bilateral = cv2.bilateralFilter(uint8_image, self.bilateral_d,
                                        self.bilateral_sigma_color,
                                        self.bilateral_sigma_space) / IMG_MAX
        low, high = np.percentile(image, self.rescale_intensity_range)

        # NOTE: This raises a warning about precision loss and invalid div. values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            img_rescale = exposure.rescale_intensity(
                image, in_range=(low, high))
            equalize_hist = exposure.equalize_adapthist(
                image, clip_limit=self.equalize_hist_clip_limit)

        dilate = cv2.dilate(
            image, self.dialation_kernel, iterations=self.dialation_iters)

        # First dimension are original color space.
        new_image[:, :, 0] = image
        # 2nd dimension is the pixels z-score.
        new_image[:, :, 1] = (image - mean) / std
        # 3rd dimension is the result of a bilateral filtering
        new_image[:, :, 2] = bilateral
        # 4th dimension is the rescaled image
        new_image[:, :, 3] = img_rescale
        # 5th dimension is the result of a adaptive histogram
        new_image[:, :, 4] = equalize_hist
        # 6th dimension is a dialation
        new_image[:, :, 5] = dilate

        return np.nan_to_num(new_image)


def extend_features(x_feat, sgd, pa):
    """
    Given features to feed into an SGD or PA regressor, stacks the
    predictions and returns the extended features.

    :param x_feat: N x X x Y x C array
    :param sgd: A MiniBatchRegressor with an SGDRegressor at its core
    :param pa: A MiniBatchRegressor with an PassiveAggressiveRegressor at its
    core

    :return: N x X x Y x (C + 2) array
    """

    #   Predict and add 4th dimension for stacking
    x_sgd = sgd.predict(x_feat)[:, :, :, np.newaxis]
    x_pa = pa.predict(x_feat)[:, :, :, np.newaxis]

    x_extended = np.concatenate([x_feat, x_sgd, x_pa], axis=3)

    return x_extended


def transform_images(feature, labels=None):
    cm = ColorMatcher()
    style_image = np.load('../data/style_image.npz')["style_image"]
    x_test_preprocessed = cm.fit_transform(style_image, x_test)
    transformer = BasisTransformer()
    x_test_transformed = transformer.fit_transform(
        np.expand_dims(x_test_preprocessed, axis=3))
    x_test_flat = np.nan_to_num(x_test_transformed).reshape(
        (-1, x_test_transformed.shape[-1]))
    weights = np.load('../weights/linear_pipeline_regressor_weights.npz')
    sgd_regressor = MiniBatchRegressor(
        regressor=SGDRegressor(
            penalty='elasticnet', l1_ratio=0.11, max_iter=5, tol=None),
        batch_size=1000,
        num_iters=50000)
    #this is so we can call predict
    sgd_regressor.fit(np.zeros((1, 8)), np.zeros((1, 1)).ravel())
    sgd_regressor.regressor.coef_ = weights["sgd"]
    pa_regressor = MiniBatchRegressor(
        regressor=PassiveAggressiveRegressor(C=.2, max_iter=5, tol=None),
        batch_size=1000,
        num_iters=1000)
    #this is so we can call predict
    pa_regressor.fit(np.zeros((1, 8)), np.zeros((1, 1)).ravel())
    pa_regressor.regressor.coef_ = weights["pa"]
    x_test_extended = np.zeros((len(x_test), 256, 256,
                                x_test_transformed.shape[3] + 2))
    x_test_extended = np.zeros((*x_test_transformed.shape[:3],
                                x_test_transformed.shape[3] + 2))
    x_test_extended[:, :, :, :x_test_transformed.shape[3]] = x_test_transformed
    x_test_extended[:, :, :, x_test_transformed.shape[
        3]] = sgd_regressor.predict(x_test_transformed)
    x_test_extended[:, :, :, x_test_transformed.shape[3] +
                    1] = pa_regressor.predict(x_test_transformed)
    return x_test_extended


FeatureTransformer = FunctionTransformer(transform_images)
