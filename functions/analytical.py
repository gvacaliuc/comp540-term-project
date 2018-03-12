import numpy as np


def has_white_background(image):
    '''
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
    image = np.array(image) * 255
    rgb_sums = np.sum(image, axis=2)
    if (rgb_sums > 620).sum() >= 14:
        return True
    else:
        return False


def is_black_white(image):
    '''
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
    image = np.array(image) * 255
    R = [np.mean(image[:, :, 0])]
    G = [np.mean(image[:, :, 1])]
    B = [np.mean(image[:, :, 2])]
    if np.std([R, G, B]) < .2:
        return True
    else:
        return False


def has_purple_foreground(image):
    '''
    determines if an input image has a purple foreground

    parameters
    __________
    image : np.array
        the raw image

    returns
    __________
    boolean
        true if the image has a purple foreground, false otherwise
    '''
    image = np.array(image) * 255
    if np.mean(image[:, :, 1]) < 178:
        return True
    else:
        return False


def is_sparse(image):
    '''
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
    image = np.array(image) * 255
    if (np.mean(image[:, :, 2]) < 114):
        return True
    else:
        return False
