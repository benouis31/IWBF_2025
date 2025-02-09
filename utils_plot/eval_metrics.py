"""
This code from @UNIROMA3

performance evaluation functions.
"""
import numpy as np
from sklearn import metrics
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(distances, labels, best_threshold=None):
    """Computes TPR, FPR, FNR, verification accuracy and best verification threashold"""
    # Calculate evaluation metrics
    #fpr, tpr, thresholds = metrics.roc_curve(labels, distances,0,drop_intermediate=True)
    fpr, tpr, thresholds = metrics.roc_curve(labels, distances, pos_label=None, sample_weight=None, drop_intermediate=True)
    fpr_optimum, fnr_optimum = _calculate_eer(fpr, (1 - tpr))
    accuracy,best_threshold = _calculate_mean_acc_dist(thresholds, distances, labels, best_threshold)
    return tpr, fpr, (1-tpr), fpr_optimum, fnr_optimum, accuracy, best_threshold


def _calculate_mean_acc_dist(thresholds, distances, labels, best_threshold):
    if best_threshold is None:
        acc_train = np.zeros((len(thresholds)))
        for threshold_idx, threshold in enumerate(thresholds):
            acc_train[threshold_idx] = _calculate_accuracy(threshold, distances, labels)
            best_threshold_index = np.argmax(acc_train)
            best_threshold=thresholds[best_threshold_index]

    accuracy = _calculate_accuracy(best_threshold, distances, labels)

    return accuracy, best_threshold


def _calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    acc = float(tp+tn)/dist.size

    return  acc


def _calculate_eer(far, frr):
    """ Returns the most optimal FAR and FRR values """

    far_optimum = far[np.nanargmin(np.absolute((frr - far)))]
    frr_optimum = frr[np.nanargmin(np.absolute((frr - far)))]

    return far_optimum, frr_optimum

