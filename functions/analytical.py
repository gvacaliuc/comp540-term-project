import numpy as np
import numpy.core.multiarray
import cv2

'''
has_white_background
determines if an input image has a white background

parameters
__________
image : np.array
    the raw image

returns
__________
has_white_background : boolean
    true if the image has a white background, false otherwise
'''


def has_white_background(image):
    rgb_sums = np.sum(image, axis=2)
    if (rgb_sums > 620).sum() >= 14:
        return True
    else:
        return False


'''
is_black_white
determines if an input image is on a black/white scale

parameters
__________
image : np.array
    the raw image

returns
__________
is_black_white : boolean
    true if the image is on a black and white scale, false otherwise
'''


def is_black_white(image):
    R = [np.mean(image[:, :, 0])]
    G = [np.mean(image[:, :, 1])]
    B = [np.mean(image[:, :, 2])]
    if np.std([R, G, B]) < 1.7:
        return True
    else:
        return False


'''
is_sparse
determines if an input image is sparse or not

parameters
__________
image : np.array
    the raw image

returns
__________
is_sparse : boolean
    true if the image is sparse, false otherwise
'''


def is_sparse(image):
    if (np.mean(image[:, :, 2]) < 114):
        return True
    else:
        return False


'''
otsu
performs otsu's binarization on an image

parameters
__________
X : np.array
    the image you are trying to predict
sparse : boolean
    if the image is sparse

return
__________
predictions : np.array
    the thresholded image
'''


def otsu(X, sparse):
    if (len(X.shape) == 3):
        if sparse:
            img_grey = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(red_vals, (5, 5), 0)
            # TODO: mess with blur size
            return (cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] / 255).astype('int')
        else:
            red_vals = X[:, :, 2]
            # TODO: mess with blur size
            blur = cv2.GaussianBlur(red_vals, (11, 11), 0)
            return (cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] / 255).astype('int')
    else:
        raise ValueError("Array had " + str(len(X.shape)) +
                         "dimensions. Needs to have 2 dimensions. You can only predict a single image at a time.")
