import numpy as np


def has_white_background(image):
    """
    determines if an input image has a white background

    parameters
    __________
    image : np.array
        the raw image

    returns
    __________
    has_white_background : boolean
        true if the image has a white background, false otherwise
    """
    image = np.array(image) * 255
    rgb_sums = np.sum(image, axis=2)
    if (rgb_sums > 620).sum() >= 14:
        return True
    else:
        return False


def is_black_white(image):
    """
    determines if an input image is on a black/white scale

    parameters
    __________
    image : np.array
        the raw image

    returns
    __________
    is_black_white : boolean
        true if the image is on a black and white scale, false otherwise
    """
    image = np.array(image) * 255
    R = [np.mean(image[:, :, 0])]
    G = [np.mean(image[:, :, 1])]
    B = [np.mean(image[:, :, 2])]
    if np.std([R, G, B]) < .2:
        return True
    else:
        return False


def has_purple_foreground(image):
    """
    determines if an input image has a purple foreground

    parameters
    __________
    image : np.array
        the raw image

    returns
    __________
    boolean
        true if the image has a purple foreground, false otherwise
    """
    image = np.array(image) * 255
    if np.mean(image[:, :, 1]) < 178:
        return True
    else:
        return False


def is_sparse(image):
    """
    determines if an input image is sparse or not

    parameters
    __________
    image : np.array
        the raw image

    returns
    __________
    is_sparse : boolean
        true if the image is sparse, false otherwise
    """
    image = np.array(image) * 255
    if (np.mean(image[:, :, 2]) < 114):
        return True
    else:
        return False

def cluster
_training_data(X_train, Y_train):
    """
    clusters the training data into modalities

    parameters
    __________
    x : np.array
        the training features
    y : np.array
        the training labels

    return
    __________
    python dictionary
        the keys are the modalities, and the values are a dictionary where the
        keys are "x" and "y" ("x" for data, "y" for labels)
    """
    X_train_sparse = []
    Y_train_sparse = []
    X_train_bw = []
    Y_train_bw = []
    X_train_wb = []
    Y_train_wb = []
    X_train_pb = []
    Y_train_pb = []
    X_train_pf = []
    Y_train_pf = []
    for i in range(len(X_train)):
        if is_sparse(X_train[i]):
            X_train_sparse.append(X_train[i])
            Y_train_sparse.append(Y_train[i])
        else:
            if is_black_white(X_train[i]):
                X_train_bw.append(X_train[i])
                Y_train_bw.append(Y_train[i])
            else:
                if has_white_background(X_train[i]):
                    if (has_purple_foreground(X_train[i])):
                        X_train_pf.append(X_train[i])
                        Y_train_pf.append(Y_train[i])
                    else:
                        X_train_wb.append(X_train[i])
                        Y_train_wb.append(Y_train[i])
                else:
                    X_train_pb.append(X_train[i])
                    Y_train_pb.append(Y_train[i])
    return {"sparse": {"x": X_train_sparse, "y": Y_train_sparse},
            "greyscale": {"x": X_train_bw, "y": Y_train_bw},
            "white_background": {"x": X_train_wb, "y": Y_train_wb},
            "purple_background": {"x": X_train_pb, "y": Y_train_pb},
            "purple_foreground": {"x": X_train_pf, "y": Y_train_pf}
            }


def cluster_test_data(X_test):
    """
    clusters the test data into modalities

    parameters
    __________
    x : np.array
        the test features

    return
    __________
    python dictionary
        the keys are the modalities, and the values are a list of examples
    """
    X_test_sparse = []
    X_test_bw = []
    X_test_wb = []
    X_test_pb = []
    X_test_pf = []
    for i in range(len(X_test)):
        if is_sparse(X_test[i]):
            X_test_sparse.append(X_test[i])
        else:
            if is_black_white(X_test[i]):
                X_test_bw.append(X_test[i])
            else:
                if has_white_background(X_test[i]):
                    if (has_purple_foreground(X_test[i])):
                        X_test_pf.append(X_test[i])
                    else:
                        X_test_wb.append(X_test[i])
                else:
                    X_test_pb.append(X_test[i])
    return {"sparse": X_test_sparse,
            "greyscale": X_test_bw,
            "white_background": X_test_wb,
            "purple_background": X_test_pb,
            "purple_foreground": X_test_pf
            }
