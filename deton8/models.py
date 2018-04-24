"""
models.py

Holds our linear model mini batch wrappers and our UNet.
"""

import inspect
from tempfile import mkdtemp

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import *
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from tqdm import tqdm

from .features import BasisTransformer
from .processing import postprocess
from .utils import NucleiDataset, expand_data, flatten_data
from . import metrics


class MiniBatchRegressor(BaseEstimator, RegressorMixin):
    """
    Model that takes a sklearn regressor implementing the partial_fit method
    and fits and predicts our image data.
    """

    def __init__(self,
                 regressor=SGDRegressor(),
                 batch_size=400,
                 num_iters=400,
                 verbose=False):
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
        y_flat = labels.reshape((x_flat.shape[0], ))

        self.regr_ = self.regressor

        arange = np.arange(x_flat.shape[0])

        for itr in tqdm(range(self.num_iters), unit="iteration"):
            batch_idx = np.random.choice(arange, size=self.batch_size)
            if self.verbose:
                print("Iteration {}/{}".format(itr, self.num_iters))

            self.regressor.partial_fit(x_flat[batch_idx], y_flat[batch_idx])

        return self

    def predict(self, images):
        """
        Predicts the output of our model on the provided data.

        :param data: ndarray of shape (N, X, Y, C)
        """

        num_features = images.shape[-1]
        data = images.reshape((-1, num_features))
        output_shape = (images.shape[:3]
                        if len(images.shape) == 4 else images.shape[:2])
        return self.regr_.predict(data).reshape((*output_shape))


class UNet(object):
    """
    Class to encapsulate the model building and training of our binarizing
    UNet.
    """

    def __init__(self,
                 numchannels=2,
                 steps_per_epoch=25,
                 epochs=50,
                 callbacks=[]):
        """
        Creates a UNet.
        """

        self.numchannels = numchannels
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.callbacks = callbacks

        self.build_model()

    def build_model(self):
        """
        Builds the keras model.
        """

        input_layer = Input(shape=(256, 256, numchannels))
        c1 = Conv2D(
            filters=8, kernel_size=(3, 3), activation='relu',
            padding='same')(input_layer)
        l = MaxPool2D(strides=(2, 2))(c1)
        c2 = Conv2D(
            filters=16, kernel_size=(3, 3), activation='relu',
            padding='same')(l)
        l = MaxPool2D(strides=(2, 2))(c2)
        c3 = Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu',
            padding='same')(l)
        l = MaxPool2D(strides=(2, 2))(c3)
        c4 = Conv2D(
            filters=32, kernel_size=(1, 1), activation='relu',
            padding='same')(l)
        l = concatenate([UpSampling2D(size=(2, 2))(c4), c3], axis=-1)
        l = Conv2D(
            filters=32, kernel_size=(2, 2), activation='relu',
            padding='same')(l)
        l = concatenate([UpSampling2D(size=(2, 2))(l), c2], axis=-1)
        l = Conv2D(
            filters=24, kernel_size=(2, 2), activation='relu',
            padding='same')(l)
        l = concatenate([UpSampling2D(size=(2, 2))(l), c1], axis=-1)
        l = Conv2D(
            filters=16, kernel_size=(2, 2), activation='relu',
            padding='same')(l)
        l = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(l)
        l = Dropout(0.5)(l)
        output_layer = Conv2D(
            filters=1, kernel_size=(1, 1), activation='sigmoid')(l)
        self.model = Model(input_layer, output_layer)
        self.model.compile(
            optimizer=Adam(.01),
            loss=metrics.dice_coef_loss,
            metrics=[metrics.dice_coef, metrics.mean_iou, metrics.f1])

    def load_weights(filename):
        """
        Loads weights from a saved file.
        """

        self.model.load_weights(filename)
        print("Successfully loaded weights from {}...".format(filename))

    def save_weights(filename):
        """
        Saves weights to the specified file.
        """

        self.model.save_weights(filename)

    def load_architecture(filename):
        """
        Loads the keras architecture from a specified file.
        """

        with open(filename, "r") as model_file:
            self.model = model_from_yaml(model_file.read())

    def save_architecture(filename):
        """
        Saves the keras architecture to the specified file.
        """

        with open(filename, "w") as model_file:
            model_str = self.model.to_yaml()
            model_file.write(model_str)

    def get_generator(self, x_train, y_train, batch_size):
        data_generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2).flow(
                x_train, x_train, batch_size, seed=42)
        mask_generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2).flow(
                y_train, y_train, batch_size, seed=42)
        while True:
            x_batch, _ = data_generator.next()
            y_batch, _ = mask_generator.next()
            yield x_batch, y_batch

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit_generator(
            self.get_generator(x_train, np.expand_dims(y_train, axis=3), 8),
            steps_per_epoch=self.steps_per_epoch,
            validation_data=(x_val, np.expand_dims(y_val, axis=3)),
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True)

    def predict(self, x_test):
        return self.model.predict(x_test)


def save(pipeline, path):
    """
    Save a pipeline to a .pkl file.
    """

    joblib.dump(pipeline, path)


def load(path):
    """
    Load a pipeline from a .pkl file.
    """

    return joblib.load(path)
