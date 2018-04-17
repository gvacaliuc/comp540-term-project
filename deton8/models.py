"""
models.py

File to hold our models.
"""

import inspect

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import SGDRegressor

from .utils import flatten_data

class MiniBatchRegressor(BaseEstimator, RegressorMixin):
    """
    Model that takes a sklearn regressor implementing the partial_fit method
    and fits and predicts our image data.
    """

    def __init__(self,
            regressor = SGDRegressor(),
            batch_size = 400,
            num_iters = 400,
            verbose = False,
            tqdm = lambda x: x):
        """
        Instantiates MiniBatchImageRegressor with the given parameters.
        """

        #   Sets all attributes.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.set_params(**values)

    def fit(self, data, labels):
        """
        Given a set of input images and masks, fits the regressor for the
        number of iterations provided.

        :param data: ndarray of shape (N, X, Y, C)
        :param labels: ndarray of shape (N, X, Y)
        """

        x_flat = flatten_data(data)
        y_flat = labels.reshape((x_flat.shape[0],))

        self.regr_ = self.regressor

        arange = np.arange(x_flat.shape[0])

        for itr in self.tqdm(range(self.num_iters)):
            batch_idx = np.random.choice(arange, size=self.batch_size)
            if self.verbose:
                print("Iteration {}/{}".format(itr, self.num_iters))

            self.regressor.partial_fit(x_flat[batch_idx], y_flat[batch_idx])

        return self

    def predict(self, data):
        """
        Predicts the output of our model on the provided data.

        :param data: ndarray of shape (N, X, Y, C)
        """

        num_features = images.shape[-1]
        data = images.reshape((-1, num_features))
        return self.predict(data).reshape(images.shape[:-1])
