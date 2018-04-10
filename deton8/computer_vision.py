import inspect

import numpy as np
from scipy.ndimage.measurements import label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.base import TransformerMixin, BaseEstimator
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes

IMG_MAX = 255.0

def otsu(image):
    """
    performs otsu's binarization on an image

    parameters
    __________
    X : np.array
        the image you are trying to predict

    return
    __________
    predictions : np.array
        the thresholded image
    """
    thresh_val = threshold_otsu(image)
    mask = np.array(image > thresh_val)
    return np.multiply(image, mask)


def postprocess(X, min_area=0):
    """
    performs non_max_suppression within connected components

    parameters
    __________
    X : np.array
        the image you are trying to suppress
    percentile : int
        the fraction of images you are accepting
    min_area : int
        the minimum area of a component you are accepting

    return
    __________
    predictions : np.array
        the thresholded image
    """
    otsu_predictions = otsu(X)
    labeled, num_labels = label(otsu_predictions)
    flattened_labels = labeled.reshape((256*256, 1))
    flattened_otsu_predictions = otsu_predictions.reshape((256*256, 1))
    for label_num in range(0, num_labels + 4):
        indices = np.where(flattened_labels == label_num)[0]
        component = flattened_otsu_predictions[indices]
        if (len(indices) > min_area):
            flattened_otsu_predictions[indices] = component
        else:
            flattened_otsu_predictions[indices] = 0
    return binary_fill_holes(flattened_otsu_predictions.reshape((256, 256)))


def preprocess_image(img, imsize = (256, 256), scale = True):
    """
    Processes an image per our specifications.
    """

    WHITE_THRESHOLD = 0.55

    img = resize(img, imsize, mode="constant", preserve_range=True)
    img /= IMG_MAX if scale else 1

    img = (1 - img) if img.mean() > WHITE_THRESHOLD else img

    return img


def match_color_with_source_dist(
        target_img, Cs, mu_s, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    matched_img = np.clip(matched_img, 0, 1)
    return matched_img


class ColorMatcher(BaseEstimator, TransformerMixin):
    """
    Given a set of color images to match, transforms a set of given images to
    match the color distribution.
    """

    def __init__(self, mode="pca", eps=1e-5, threshold=0.2):
        """
        Creates a new ColorMatcher.

        :param mode: color matching mode to be in. one of [pca, chol, sym].
        :param eps: a parameter to add to prevent division over 0
        """

        #   Sets all attributes.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.set_params(**values)

        valid_modes = ["pca", "cholesky", "symmetric"]
        if mode not in valid_modes:
            raise ValueError("Invalid ColorMatcher mode: {}.".format(mode))

    def fit(self, style_images):
        """
        Stores an array of style images to be used in transformations.

        :param style_images: an array of style images to use
        """

        #   Threshold style_images immediately to increase contrast:
        style_images = (style_images > self.threshold).astype(style_images.dtype)

        _, _, _, channels = style_images.shape
        #   Take mean over height / width
        per_image_channel_mean = np.mean(
                style_images, axis=(1, 2), keepdims=True)
        #   Store per channel mean
        self.mu_source_ = np.mean(per_image_channel_mean, axis=0)

        centered = style_images - per_image_channel_mean
        flattened = centered.reshape((-1, channels))

        self.cov_source_ = (flattened.T.dot(flattened)
                            / np.prod(style_images.shape[:3])
                            + self.eps * np.eye(channels))

        #   Handle Cholesky
        self.cholesky_ = np.linalg.cholesky(self.cov_source_)

        #   Handle PCA
        eigval, eigvec = np.linalg.eigh(self.cov_source_)
        self.Qs_ = eigvec.dot(np.sqrt(np.diag(eigval))).dot(eigvec.T)

        #   Handle Symmetric

        return self

    def transform(self, content_images):
        """
        Transforms the images in content_images according to the covariance
        matrix and mode specified previously.
        """

        return np.stack([self._compute_projection(im) for im in content_images])

    def fit_transform(self, style_images, content_images):
        """
        Fits the estimator with the style images, then transforms the content
        images.

        :param style_images: same as in fit
        :param content_images: same as in transform
        """

        return self.fit(style_images).transform(content_images)

    def _compute_projection(self, image):
        """
        Computes the projection of a single image.  Before matching color, we
        convert the rgb to greyscale.
        """

        greyscale = np.rollaxis(np.stack([rgb2gray(image)]*3), 0, 3)

        return match_color_with_source_dist(
                greyscale, self.cov_source_,
                self.mu_source_, self.mode, self.eps)
