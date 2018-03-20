from scipy.ndimage import binary_fill_holes as fill_holes

from .analytical import *
from .clustering import *
from .computer_vision import *
from .predict import *


def predict(image):
    # if the image is sparse
    if is_sparse(image):
        features = basis_map(image)
        lda_output = lda_predict(features)
        return non_max_component_suppression(
            lda_output, min_area=5, percent=99)
    # if the image is dense
    else:
        # if the image is on a black/white scale
        if is_black_white(image):
            raise NotImplementedError("dense black/white")
        # if the image is on a color scale
        else:
            # if the image has a white background
            if has_white_background(image):
                if (has_purple_foreground(image)):
                    features = basis_map(1 - image)
                    svr_output = lr_predict(features)
                    return non_max_component_suppression(
                        (svr_output > .77) * svr_output, min_area=10, percent=95)
                else:
                    features = basis_map(1 - image)
                    lr_output = lr_predict(features)
                    return non_max_component_suppression(
                        lr_output, min_area=5, percent=95)
            # is the image has a purple background
            else:
                raise NotImplementedError("dense purple background")
    return fill_holes(predictions)
