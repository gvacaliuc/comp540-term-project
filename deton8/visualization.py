import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .utils import image_predict
from .components import watershed_cc

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
