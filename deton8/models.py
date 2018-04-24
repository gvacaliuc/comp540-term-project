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
from keras.models import Model
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

from .analytical import BasisTransformer
from .computer_vision import postprocess
from .models import MiniBatchRegressor
from .preprocess import preprocess
from .unet import UNet
from .utils import NucleiDataset, expand_data, flatten_data


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
    Class to encapsulate the model building and training of our 
    Binarizing UNet.
    """

    def __init__(self,
                 numchannels=2,
                 steps_per_epoch=25,
                 epochs=50,
                 callbacks=[],
                 saved_weights="../weights/unet_weights.h5"):
        """
        Creates a UNet.
        """

        self.numchannels = numchannels
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.callbacks = callbacks

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
            loss=self.dice_coef_loss,
            metrics=[self.dice_coef, self.mean_iou, self.f1])
        if saved_weights:
            self.model.load_weights(saved_weights)
            print("Loaded saved weights...")

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

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        tp = K.sum(y_true_f * y_pred_f)
        fp = K.sum((1 - y_true_f) * y_pred_f)
        p = K.sum(y_true_f)

        return tp / (p + fp)

    def mean_iou(self, y_true, y_pred):
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        return score

    def dice_coef_loss(self, y_true, y_pred):
        return -1 * self.dice_coef(y_true, y_pred)

    def f1(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        tp = K.sum(y_true_f * y_pred_f)
        fp = K.sum((1 - y_true_f) * y_pred_f)
        fn = K.sum(y_true_f * (1 - y_pred_f))

        prec = tp / (tp + fp)
        recall = tp / (tp + fn)

        return 2 * (prec * recall) / (prec + recall)

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


def pipeline(directory, train=True, max_size=None):
    dataset = NucleiDataset(directory, train=train).load(max_size=max_size)
    metadata = dataset.metadata_
    x_raw = dataset.images_
    x_preprocessed = preprocess(x_raw)
    unet = UNet()
    x_predictions = (unet.predict(x_preprocessed) > 0).astype("uint8")
    x_postprocessed = np.array(
        [postprocess(im, min_area=15) for im in x_predictions])
    return x_postprocessed, metadata


def LinearPipeline(memory=mkdtemp()):
    """
    Returns a sklearn.pipeline.Pipeline with our pipeline preconfigured.
    Hyperparameters may be set using set_params().
    """

    return Pipeline(
        [("flattener", FunctionTransformer(flatten_data, validate=False)),
         ("whitener", PCA(
             n_components=1, svd_solver='randomized', whiten=True)),
         ("minmaxscaler", MinMaxScaler()),
         ("expander", FunctionTransformer(expand_data, validate=False)),
         ("basis_transformer", BasisTransformer()),
         ("regressor", MiniBatchRegressor(batch_size=1000, num_iters=1000))],
        memory=memory)


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
