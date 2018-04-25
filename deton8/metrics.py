import tensorflow as tf
from keras import backend as K


def dice_coef(y_true, y_pred):
    print(y_true)
    print(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    p = K.sum(y_true_f)

    return tp / (p + fp)


def mean_iou(y_true, y_pred):
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def dice_coef_loss(y_true, y_pred):
    return -1 * dice_coef(y_true, y_pred)


def f1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1_score = 2 * (prec * recall) / (prec + recall)
    return f1_score
