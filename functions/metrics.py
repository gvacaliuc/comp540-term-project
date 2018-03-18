import tensorflow as tf
from keras import backend as K


def mean_iou(y_true, y_pred):
    """
    Calculates the mean IoU of our predicted mask and the true mask at 
    several different thresholds.  For more information refer to the 
    DS Bowl Evaluation Methodology: 
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_true_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(tf.convert_to_tensor(y_true),
                                            tf.convert_to_tensor(y_pred),
                                            2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0).eval(session=K.get_session())


def precision_recall_f1(labels, predictions):
    """
    Calculates precision, recall, and f1 metrics for our predicted mask and
    true mask.  Both matrices must have binary data.

    parameters
    __________
    labels : np.array (2d)
        the true labels
    predictions : np.array (2d)
        the predictions

    returns
    __________
    (precision, recall, f1) : tuple (double)
    """

    metrics = [image_precision, image_recall, image_f1]

    return tuple([met(labels, predictions) for met in metrics])


def image_f1(labels, predictions):
    return f1_score(labels.flatten(), predictions.flatten())


def image_precision(labels, predictions):
    return precision_score(labels.flatten(), predictions.flatten())


def image_recall(labels, predictions):
    return recall_score(labels.flatten(), predictions.flatten())
