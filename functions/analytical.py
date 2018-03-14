import cv2
import numpy as np
from skimage import exposure


def basis_map(image):
    """
    maps each pixel of an image to a set of features

    parameters
    __________
    image : np.array (128 x 128)
        the input image

    returns
    __________
    np.array (128 x 128 x 9)
        the image features
    """
    new_image = np.zeros((image.shape[0], image.shape[1], 9))
    mean = np.mean(image)
    std = np.std(image)
    bilateral = cv2.bilateralFilter(
        np.mean(image * 255, axis=2).astype("uint8"), 2, 75, 75) / 255.0
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = np.mean(exposure.rescale_intensity(
        image, in_range=(p2, p98)), axis=2)
    equalize_hist = np.mean(exposure.equalize_adapthist(
        image, clip_limit=0.03), axis=2)
    dilate = np.mean(cv2.dilate(image, np.ones((5, 5)), iterations=2), axis=2)
    for i in range(len(image)):
        for j in range(len(image)):
            new_image[i][j][0] = image[i][j][0]
            new_image[i][j][1] = image[i][j][1]
            new_image[i][j][2] = image[i][j][2]
            new_image[i][j][3] = (np.mean(image[i][j]) - mean) / std
            new_image[i][j][4] = bilateral[i][j]
            new_image[i][j][5] = img_rescale[i][j]
            new_image[i][j][6] = equalize_hist[i][j]
            new_image[i][j][7] = dilate[i][j]
    for i in range(len(image)):
        for j in range(len(image)):
            lower_i = max(i - 1, 0)
            upper_i = min(i + 1, image.shape[0] - 1)
            lower_j = max(j - 1, 0)
            upper_j = min(j + 1, image.shape[1] - 1)
            new_image[i][j][8] = np.mean([np.linalg.norm(new_image[i][j] - new_image[neighbor_i][neighbor_j])
                                          for neighbor_i in range(lower_i, upper_i + 1) for neighbor_j in range(lower_j, upper_j + 1)])
    return np.nan_to_num(new_image)
