from .preprocess import preprocess
from .unet import UNet
from .computer_vision import postprocess
import numpy as np


from .utils import DataReader

def pipeline(directory):
    dataReader = DataReader(directory)
    metadata, x_raw = dataReader.get()
    x_preprocessed = preprocess(x_raw)
    unet = UNet()
    x_predictions = (unet.predict(x_preprocessed) > 0).astype("uint8")
    x_postprocessed = np.array([postprocess(im, min_area = 15) for im in x_predictions])
    return x_postprocessed, metadata


