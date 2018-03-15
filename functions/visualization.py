import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from main import predict


def viz_prediction(image, true_mask = None):
    """
    Visualizes an image, it's prediction under our pipeline, 
    and the true mask if given.  Will be plotted horizontally.

    :param image: the image to use
    :type image: ndarray, (height, width, channels)
    :return plt.figure: the figure which was plotted on
    """

    num_images = 2 if true_mask is None else 3
    images = [image, predict(image), true_mask][:num_images]
    titles = ["raw image", "prediction", "true nuclei"][:num_images]


    fig = plt.figure()
    gs = GridSpec(1, num_images)
    for ind, (im, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(gs[0, ind])
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)

    return fig
