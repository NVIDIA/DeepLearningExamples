"""
Multiclass
from: 
https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

"""

__author__ = "Guillaume Genthial"

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix


def precision(labels, predictions, num_classes, pos_indices=None,
              weights=None, average='micro'):
    """Multi-class precision metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    pr, _, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    op, _, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (pr, op)


def recall(labels, predictions, num_classes, pos_indices=None, weights=None,
           average='micro'):
    """Multi-class recall metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, re, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    _, op, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (re, op)


def f1(labels, predictions, num_classes, pos_indices=None, weights=None,
       average='micro'):
    return fbeta(labels, predictions, num_classes, pos_indices, weights,
                 average)


def fbeta(labels, predictions, num_classes, pos_indices=None, weights=None,
          average='micro', beta=1):
    """Multi-class fbeta metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    beta : int, optional
        Weight of precision in harmonic mean
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, _, fbeta = metrics_from_confusion_matrix(
        cm, pos_indices, average=average, beta=beta)
    _, _, op = metrics_from_confusion_matrix(
        op, pos_indices, average=average, beta=beta)
    return (fbeta, op)


def safe_div(numerator, denominator):
    """Safe division, return 0 if denominator is 0"""
    numerator, denominator = tf.to_float(numerator), tf.to_float(denominator)
    zeros = tf.zeros_like(numerator, dtype=numerator.dtype)
    denominator_is_zero = tf.equal(denominator, zeros)
    return tf.where(denominator_is_zero, zeros, numerator / denominator)


def pr_re_fbeta(cm, pos_indices, beta=1):
    """Uses a confusion matrix to compute precision, recall and fbeta"""
    num_classes = cm.shape[0]
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0
    diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0
    tot_pred = tf.reduce_sum(cm * cm_mask)

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0
    tot_gold = tf.reduce_sum(cm * cm_mask)

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    fbeta = safe_div((1. + beta**2) * pr * re, beta**2 * pr + re)

    return pr, re, fbeta


def metrics_from_confusion_matrix(cm, pos_indices=None, average='micro',
                                  beta=1):
    """Precision, Recall and F1 from the confusion matrix
    Parameters
    ----------
    cm : tf.Tensor of type tf.int32, of shape (num_classes, num_classes)
        The streaming confusion matrix.
    pos_indices : list of int, optional
        The indices of the positive classes
    beta : int, optional
        Weight of precision in harmonic mean
    average : str, optional
        'micro', 'macro' or 'weighted'
    """
    num_classes = cm.shape[0]
    if pos_indices is None:
        pos_indices = [i for i in range(num_classes)]

    if average == 'micro':
        return pr_re_fbeta(cm, pos_indices, beta)
    elif average in {'macro', 'weighted'}:
        precisions, recalls, fbetas, n_golds = [], [], [], []
        for idx in pos_indices:
            pr, re, fbeta = pr_re_fbeta(cm, [idx], beta)
            precisions.append(pr)
            recalls.append(re)
            fbetas.append(fbeta)
            cm_mask = np.zeros([num_classes, num_classes])
            cm_mask[idx, :] = 1
            n_golds.append(tf.to_float(tf.reduce_sum(cm * cm_mask)))

        if average == 'macro':
            pr = tf.reduce_mean(precisions)
            re = tf.reduce_mean(recalls)
            fbeta = tf.reduce_mean(fbetas)
            return pr, re, fbeta
        if average == 'weighted':
            n_gold = tf.reduce_sum(n_golds)
            pr_sum = sum(p * n for p, n in zip(precisions, n_golds))
            pr = safe_div(pr_sum, n_gold)
            re_sum = sum(r * n for r, n in zip(recalls, n_golds))
            re = safe_div(re_sum, n_gold)
            fbeta_sum = sum(f * n for f, n in zip(fbetas, n_golds))
            fbeta = safe_div(fbeta_sum, n_gold)
            return pr, re, fbeta

    else:
        raise NotImplementedError()
