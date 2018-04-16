from .preprocess import preprocess
from .unet import UNet
from .computer_vision import postprocess
import numpy as np


from .utils import NucleiDataset

def pipeline(directory, train=True, max_size=None):
    dataset = NucleiDataset(directory, train=train).load(max_size=max_size)
    metadata = dataset.metadata_
    x_raw = dataset.images_
    x_preprocessed = preprocess(x_raw)
    unet = UNet()
    x_predictions = (unet.predict(x_preprocessed) > 0).astype("uint8")
    x_postprocessed = np.array([postprocess(im, min_area = 15) for im in x_predictions])
    return x_postprocessed, metadata
