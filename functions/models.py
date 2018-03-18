"""
models.py

File to hold our models.
"""

import inspect

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import SGDRegressor


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
        Trains our regressor for the number of iterations provided 
        as well as 
        """

        self.regr_ = self.regressor
        
        arange = np.arange(data.shape[0])

        for itr in self.tqdm(range(self.num_iters)):
            batch_idx = np.random.choice(arange, size=self.batch_size)
            if self.verbose:
                print("Iteration {}/{}".format(itr, self.num_iters))

            self.regr_.partial_fit(data[batch_idx], labels[batch_idx])

    def predict(self, data):
        """
        Predicts the output of our model on the provided data.
        """

        return self.regr_.predict(data)

    def predict_images(self, images):
        """
        Predicts the output of our model for a set of provided images.  Note
        that the number of channels should be the number of features this
        model was trained with.

        :param images: ndarray of shape N x X x Y x C
        """

        num_features = images.shape[-1]
        data = images.reshape((-1, num_features))
        return self.predict(data).reshape(images.shape[:-1])
