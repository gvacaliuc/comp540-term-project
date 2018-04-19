from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import tensorflow as tf

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
        c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
        l = MaxPool2D(strides=(2,2))(c1)
        c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
        l = MaxPool2D(strides=(2,2))(c2)
        c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
        l = MaxPool2D(strides=(2,2))(c3)
        c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
        l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
        l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
        l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
        l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
        l = Dropout(0.5)(l)
        output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer=Adam(.01),
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
                shear_range=0.2).flow(x_train, x_train, batch_size, seed=42)
        mask_generator = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                shear_range=0.2).flow(y_train, y_train, batch_size, seed=42)
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
        return -1*self.dice_coef(y_true, y_pred)

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
            self.get_generator(x_train, np.expand_dims(y_train, axis = 3), 8),
            steps_per_epoch = self.steps_per_epoch,
            validation_data = (x_val, np.expand_dims(y_val, axis = 3)),
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True)

    def predict(self, x_test):
        return self.model.predict(x_test)
