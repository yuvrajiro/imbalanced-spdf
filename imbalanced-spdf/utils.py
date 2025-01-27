import math
import random
from collections import Counter
from .tree import tree
import numpy as np


def bootstrap_sample(X, y):
    """
    This function creates a bootstrap sample of the data.
    :param X: The numpy array of the data.
    :param y: The numpy array of the labels.
    :return: X and y, the bootstrap sample of the data.
    """
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):

    """
    The most common label in the data.
    :param y: Predicted labels.
    :return: Most common label.
    """
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

def weight_update(amount_of_say, weight, ind, x_train, y_train):
    """
    This function updates the weight of the samples.
    :param amount_of_say: The amount of say of the tree.
    :param weight: weight of the samples.
    :param ind: index of tree
    :param x_train: x_train
    :param y_train: y_train
    :return: A dictionary containing the updated x_train and y_train.
    """

    wght = [w * math.exp(amount_of_say * i) for w, i in zip(weight, ind)]
    wght = wght / np.sum(wght)
    cum_wght = cumulative(wght)
    sam = np.zeros(len(y_train))
    for i in range(len(y_train)):
        u = random.random()
    sam[i] = np.min(np.where(cum_wght >= u))
    sam = np.array(sam, dtype=int)
    x_train = x_train[sam, :]
    y_train = y_train[sam]

    return {"x_train": x_train, "y_train": y_train}

def cumulative(freq):
    """
    This function calculates the cumulative frequency.
    :param freq:
    :return:
    """
    l = len(freq)
    vec = np.zeros(l)
    for i in range(l):
        vec[i] = np.sum(freq[0:i + 1])
    return (vec)

def penalty(X_train, y_train, times, n, n1train, pen_lst):
    """
    This function calculates the penalty for the tree.
    :param X_train: X_train a numpy array of the training data.
    :param y_train: y_train a numpy array of the training labels.
    :param times: times the weight of the minority class.
    :param n: n the number of samples.
    :param n1train: n1train the number of samples in minority class.
    :param pen_lst: list of penalties.
    :return: pen: the penalty for the tree.
    """
    F_lst = np.zeros(len(pen_lst))
    for k in range(len(pen_lst)):
        TP = 0
        FP = 0
        tr_svr = tree()
        tr_svr.fit_sv(X_train, y_train, pen_lst[k], weight=times + 1, maximal_leaves=2 * np.sqrt(n * 2 / 3))
        Y_pred_temp = tr_svr.predict(X_train)
        TP += np.sum(Y_pred_temp[np.flatnonzero(y_train)])
        FP += np.sum(Y_pred_temp[np.flatnonzero(y_train == 0)])
    if TP > 0:
        tpr = TP / n1train
        precision = TP / (TP + FP)
        F_lst[k] = 2 * tpr * precision / (tpr + precision)
    para_id = np.argmax(F_lst)
    pen = pen_lst[para_id]
    return (pen)