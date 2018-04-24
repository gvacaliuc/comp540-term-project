"""
deton8 -- Detector of Nuclei

Main script to train and test the model.
"""

import argparse
import sys
import yaml

from skimage.morphology import disk

from .utils import NucleiDataset
from .models import MiniBatchRegressor, UNet
from .features import extend_features
from .features import BasisTransformer
from .processing import Preprocesser

parser = argparse.ArgumentParser()
parser.add_argument(
    "command",
    choices=["train", "test", "generate_config"],
    type=str,
    help="""The command to run.
                train: trains all models given a configuration
                test: tests on a directory and builds a submission, saving it
                      to --csv if provided.  If masks are present then will
                      also report score.
                generate_config: generates a default training configuration."""
)
parser.add_argument(
    "--config",
    dest="config_file",
    type=str,
    default=None,
    help="YAML configuration file to load if train/test/submit, else ")
parser.add_argument(
    "--csv",
    dest="submission_csv",
    type=str,
    default=None,
    help="If 'submit' is called, the submission csv is saved here.")


def run_configuration(config):
    """
    Runs the pipeline according to the configuration.
    """


def train(config):
    """
    """

    num_train = 100
    num_val = 50

    train_data = NucleiDataset(train_dir).load(num_train)
    val_data = NucleiDataset(val_dir).load(num_val)

    x_train = train_data.images_
    y_train = train_data.masks_  # value in 0, 1, ..., n
    y_train_bin = (y_train > 0).astype(y_train.dtype)  # value in {0, 1}
    x_val = val_data.images_
    y_val = val_data.masks_
    y_val_bin = (y_val > 0).astype(y_val.dtype)

    preprocesser = Preprocesser()

    x_train_pre = preprocesser.fit_transform(x_train)
    x_val_pre = preprocesser.transform(x_val)

    bilateral_d = 2
    bilateral_sigma_color = 75
    bilateral_sigma_space = 75
    equalize_hist_clip_limit = 0.03
    dialation_kernel = disk(radius=3)
    dialation_iters = 1

    transformer = BasisTransformer(
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
        equalize_hist_clip_limit=equalize_hist_clip_limit,
        dialation_kernel=dialation_kernel,
        dialation_iters=dialation_iters)

    x_train_feat = transformer.fit_transform(x_train_pre)
    x_val_feat = transformer.fit_transform(x_val_pre)

    sgd_params = {
        "regressor":
        SGDRegressor(
            penalty='elasticnet', l1_ratio=0.11, max_iter=5, tol=None),
        "batch_size":
        1000,
        "num_iters":
        25000,
    }
    pa_params = {
        "regressor": PassiveAggressiveRegressor(C=.2, max_iter=5, tol=None),
        "batch_size": 1000,
        "num_iters": 25000,
    }

    sgd = MiniBatchRegressor(**sgd_params)
    pa = MiniBatchRegressor(**pa_params)

    sgd.fit(x_train_feat, y_train_bin)
    pa.fit(x_train_feat, y_train_bin)

    x_train_extended = extend_features(x_train_feat, sgd, pa)
    x_val_extended = extend_features(x_val_feat, sgd, pa)

    #   Now we train UNet
    numchannels = x_train_extended.shape[-1]
    unet_config = {
        "numchannels": numchannels,
        "epochs": 50,
        "callbacks": [],
        "weights": None
    }
    unet = UNet(**unet_config)

    if unet_config["weights"] is not None:
        unet.load_weights(unet_config["weights"])

    unet.fit(x_train_extended, y_train_bin, x_val_extended, y_val_bin)

    #   begin inference and print out test scores
    x_train_pred = unet.predict(x_train_extended)
    x_val_pred = unet.predict(x_val_extended)

    segmenter_params = {"nms_min_distance": 3, "watershed_line": True}
    segmenter = NucleiSegmenter(**segmenter_params)

    train_components = segmenter.fit_transform(x_train_pred, x_train_pre)
    val_components = segmenter.fit_transform(x_val_pred, x_val_pre)

    #postprocess(im, min_area=15) for im in x_predictions


def get_default_config():
    return {
        "train_dir": "",
        "val_dir": "",
        "test_dir": "",
        "feature_params": {
            "bilateral_d": 2,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            "equalize_hist_clip_limit": 0.03
        },
        "sgd_params": {
            "regressor__penalty": "elasticnet",
            "regressor__l1_ratio": 0.11,
            "batch_size": 1000,
            "num_iters": 25000
        },
        "pa_params": {
            "regressor__C": 0.2,
            "batch_size": 1000,
            "num_iters": 25000
        },
        "unet_params": {
            "weights": "",
            "architecture": "",
            "epochs": 50
        },
        "segmenter_params": {
            "nms_min_distance": 3,
            "watershed_line": True
        }
    }


def main():
    """
    Entry point of script.
    """

    parsed = parser.parse_args()
    if parsed.command == "generate_config" and parsed.config_file:
        with open(parsed.config_file, "w") as config_file:
            config_file.write(
                yaml.dump(get_default_config(), default_flow_style=False))

    else
