from .preprocess import preprocess
from .unet import UNet
from .computer_vision import postprocess
import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tempfile import mkdtemp
from .analytical import BasisTransformer
from .models import MiniBatchRegressor

from .utils import NucleiDataset, flatten_data, expand_data

def pipeline(directory, train=True, max_size=None):
    dataset = NucleiDataset(directory, train=train).load(max_size=max_size)
    metadata = dataset.metadata_
    x_raw = dataset.images_
    x_preprocessed = preprocess(x_raw)
    unet = UNet()
    x_predictions = (unet.predict(x_preprocessed) > 0).astype("uint8")
    x_postprocessed = np.array([postprocess(im, min_area = 15) for im in x_predictions])
    return x_postprocessed, metadata

def LinearPipeline(memory=mkdtemp()):
    """
    Returns a sklearn.pipeline.Pipeline with our pipeline preconfigured.
    Hyperparameters may be set using set_params().
    """

    return Pipeline(
        [("flattener", FunctionTransformer(flatten_data)),
         ("whitener", 
          PCA(n_components=1, svd_solver='randomized', whiten=True)),
         ("expander", FunctionTransformer(expand_data)),
         ("basis_transformer", BasisTransformer()),
         ("regressor", MiniBatchRegressor(batch_size=1000, num_iters=1000))],
        memory=memory)
