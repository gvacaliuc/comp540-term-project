from analytical import *


def predict(X):
    predictions = []
    for image in X:
        # if the image is sparse
        if is_sparse(image):
            predictions.append(otsu(image, is_sparse=True))
        # if the image is dense
        else:
            # if the image is on a black/white scale
            if is_black_white(image):
                raise ValueError("not yet implemented")
            # if the image is on a color scale
            else:
                # if the image has a white background
                if has_white_background(image):
                    predictions.append(1 - otsu(image, is_sparse=False))
                # is the image has a purple background
                else:
                    raise ValueError("not yet implemented")
    return predictions
