from itertools import product

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from .utils import image_predict
from .components import watershed_cc
from .computer_vision import ColorMatcher

def plot_prediction(model, features, mask):

    pred = image_predict(features, model)

    plt.figure()
    plt.subplot(131)
    plt.imshow(features[:, :, :3])
    plt.title("original image")
    plt.subplot(132)
    plt.imshow(pred)
    plt.title("predicted outputs")
    plt.subplot(133)
    plt.imshow(mask)
    plt.title("true mask")

def plot_segmentation(features, pred, mask):

    ccs, seg = watershed_cc(pred, original_image=features, nms_min_distance=7, watershed_line=True, return_mask=True)
    plt.figure()
    plt.subplot(131)
    plt.imshow(features[:, :, :3])
    plt.title("original image")
    plt.subplot(132)
    plt.imshow(seg)
    plt.title("indiv. nucleus segmentation")
    plt.subplot(133)
    plt.imshow(mask)
    plt.title("true mask")


def plot_color_transfer_results(trainImages, style_indices, grid_size=(5, 5)):
    """
    plots the results from our color transfer code.
    """

    num_images = np.prod(grid_size)
    idx = np.random.choice(
            np.arange(len(trainImages)), size=num_images, replace=False)
    style_images = trainImages[style_indices]
    content_images = trainImages[idx]

    cm = ColorMatcher()
    transformed_images = cm.fit_transform(style_images, content_images)

    expanded_grid_size = (grid_size[0], grid_size[1] * 2)

    gs = GridSpec(*expanded_grid_size)
    
    left_ims = content_images.reshape((*grid_size, *content_images.shape[1:]))
    right_ims = transformed_images.reshape((*grid_size, *transformed_images.shape[1:]))

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            left_ax = plt.subplot(gs[i, j])
            left_ax.imshow(left_ims[i, j])
            left_ax.axis('off')
            right_ax = plt.subplot(gs[i, j + grid_size[1]])
            right_ax.imshow(right_ims[i, j])
            right_ax.axis('off')


    plt.suptitle("Left: Original Images, Right: Images after Color Transfer")
    plt.show()
