# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    num_classes = pred.size(1)
    _, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target_label, pred_label):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        gt_positive = label == target.reshape(-1, 1)
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores
    
def precision_recall_f1_auc(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score, auc.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        gt_positive = label == target.reshape(-1, 1)
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    # compute auc
    y_true_bin = label_binarize(target, classes=[x for x in range(pred.shape[1])])

    # 计算每个类别的AUC
    result = 0
    aucs = []
    for i in range(y_true_bin.shape[1]):
      fpr, tpr, _ = roc_curve(y_true_bin[:, i], pred[:, i])
      auc = roc_auc_score(y_true_bin[:, i], pred[:, i])
      aucs.append(auc)

    # 计算micro-average AUC
    if average_mode == "micro":
        result = roc_auc_score(y_true_bin.ravel(), pred.ravel()) * 100
    
    # 计算macro-average AUC
    if average_mode == "macro":
        result = sum(aucs) / len(aucs) * 100

    if return_single:
        return precisions[0], recalls[0], f1_scores[0], result
    else:
        return precisions, recalls, f1_scores, result

def specificity_precision_recall_f1_auc(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score, auc.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    specificitys = []
    f1_scores = []
    aucs = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        pred_negative = ~pred_positive
        gt_positive = label == target.reshape(-1, 1)
        gt_negative = ~gt_positive
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        specificity = (pred_negative & gt_negative).sum(0) / np.maximum(
            gt_negative.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            specificity = float(specificity.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        specificitys.append(specificity)
        f1_scores.append(f1_score)

    # # compute auc
    # y_true_bin = label_binarize(target, classes=[x for x in range(pred.shape[1])])

    # # 计算每个类别的AUC
    # result = 0
    # aucs = []
    # for i in range(y_true_bin.shape[1]):
    #   fpr, tpr, _ = roc_curve(y_true_bin[:, i], pred[:, i])
    #   auc = roc_auc_score(y_true_bin[:, i], pred[:, i])
    #   aucs.append(auc)

    # # 计算micro-average AUC
    # if average_mode == "micro":
    #     result = roc_auc_score(y_true_bin.ravel(), pred.ravel()) * 100
    
    # # 计算macro-average AUC
    # if average_mode == "macro":
    #     result = sum(aucs) / len(aucs) * 100

    # ===== 修正 AUC 计算部分 =====
    y_true_bin = label_binarize(target, classes=range(pred.shape[1]))
    aucs_per_class = []  # 存储每个类别的AUC
    
    for i in range(pred.shape[1]):
        auc_class = roc_auc_score(y_true_bin[:, i], pred[:, i])
        aucs_per_class.append(auc_class * 100)  # 转换为百分比形式
        
    # 根据 average_mode 决定返回类型
    if average_mode == "none":
        result = np.array(aucs_per_class)  # 每个类别的AUC数组
    else:
        result = np.mean(aucs_per_class)   # 平均AUC（宏平均）
        
    if return_single:
        return precisions[0], recalls[0], specificitys[0], f1_scores[0], result
    else:
        return precisions, recalls, specificitys, f1_scores, result

def specificity_precision_recall_f1_auc_acc(pred, target, average_mode='macro', thrs=0.):
    """
    Calculate precision, recall and f1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score, auc.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    specificitys = []
    f1_scores = []
    aucs = []
    accs = []

    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        pred_negative = ~pred_positive
        gt_positive = label == target.reshape(-1, 1)
        gt_negative = ~gt_positive
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(gt_positive.sum(0), 1) * 100
        specificity = (pred_negative & gt_negative).sum(0) / np.maximum(gt_negative.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,1e-20)
        accuracy = (pred_positive & gt_positive).sum(0) + (pred_negative & gt_negative).sum(0)
        accuracy = accuracy / np.maximum(pred_positive.sum(0) + pred_negative.sum(0), 1) * 100

        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            specificity = float(specificity.mean())
            f1_score = float(f1_score.mean())
            accuracy = float(accuracy.mean())

        precisions.append(precision)
        recalls.append(recall)
        specificitys.append(specificity)
        f1_scores.append(f1_score)
        accs.append(accuracy)

    # ===== 修正 AUC 计算部分 =====
    y_true_bin = label_binarize(target, classes=range(pred.shape[1]))
    aucs_per_class = []  # 存储每个类别的AUC
    
    for i in range(pred.shape[1]):
        auc_class = roc_auc_score(y_true_bin[:, i], pred[:, i])
        aucs_per_class.append(auc_class * 100)  # 转换为百分比形式
        
    # 根据 average_mode 决定返回类型
    if average_mode == "none":
        result = np.array(aucs_per_class)  # 每个类别的AUC数组
    else:
        result = np.mean(aucs_per_class)   # 平均AUC（宏平均）
        
    if return_single:
        return precisions[0], recalls[0], specificitys[0], f1_scores[0], result, accs[0]
    else:
        return precisions, recalls, specificitys, f1_scores, result, accs

def multiclass_metrics(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array | list): The model predicted labels with shape (N,).
        target (torch.Tensor | np.array | list): The ground truth labels with shape (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score, auc.
    """
    
    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    # Convert lists to np.array if needed
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(target, list):
        target = np.array(target)

    # Check if pred and target are numpy arrays
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # If thrs is a number, convert it to tuple for consistency
    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    # Calculate precision, recall, specificity, and F1 score based on predicted labels
    precisions = []
    recalls = []
    specificitys = []
    f1_scores = []
    aucs = []
    
    # For each threshold
    for thr in thrs:
        # When working with predicted labels, we treat predictions as '1' for the predicted class and '0' for others
        pred_label = pred  # pred is now the predicted class (not score)
        
        # Calculate precision, recall, specificity, F1 score for each class
        precision = []
        recall = []
        specificity = []
        f1_score = []
        
        # For each class (for binary or multiclass)
        num_classes = np.unique(target)
        for label in num_classes:
            # True positives, false positives, true negatives, false negatives
            tp = np.sum((pred_label == label) & (target == label))
            fp = np.sum((pred_label == label) & (target != label))
            tn = np.sum((pred_label != label) & (target != label))
            fn = np.sum((pred_label != label) & (target == label))
            
            # Precision, Recall, Specificity, F1 score
            precision.append(tp / (tp + fp) * 100 if tp + fp > 0 else 0)
            recall.append(tp / (tp + fn) * 100 if tp + fn > 0 else 0)
            specificity.append(tn / (tn + fp) * 100 if tn + fp > 0 else 0)
            f1_score.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1] + 1e-20) if precision[-1] + recall[-1] > 0 else 0)

        # If averaging mode is 'macro', compute the mean of metrics for each class
        if average_mode == 'macro':
            precision = np.mean(precision)
            recall = np.mean(recall)
            specificity = np.mean(specificity)
            f1_score = np.mean(f1_score)
        precisions.append(precision)
        recalls.append(recall)
        specificitys.append(specificity)
        f1_scores.append(f1_score)

    # Compute AUC based on the true labels and predicted labels
    y_true_bin = label_binarize(target, classes=[x for x in range(np.max(target) + 1)])

    result = 0
    aucs = []
    for i in range(y_true_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], pred == i)  # For predicted labels, compare against each class
        auc = roc_auc_score(y_true_bin[:, i], pred == i)
        aucs.append(auc)

    # Compute micro-average AUC
    if average_mode == "micro":
        result = roc_auc_score(y_true_bin.ravel(), (pred == np.arange(len(np.unique(pred)))).ravel()) * 100

    # Compute macro-average AUC
    if average_mode == "macro":
        result = np.mean(aucs) * 100

    if return_single:
        return precisions[0], recalls[0], specificitys[0], f1_scores[0], result
    else:
        return precisions, recalls, specificitys, f1_scores, result

def auc(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: AUC.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    # confusion_matrix = calculate_confusion_matrix(pred, target)
    # with torch.no_grad():
    #     res = confusion_matrix.sum(1)
    #     if average_mode == 'macro':
    #         res = float(res.sum().numpy())
    #     elif average_mode == 'none':
    #         res = res.numpy()
    #     else:
    #         raise ValueError(f'Unsupport type of averaging {average_mode}.')  
    _, _, _, aucs = precision_recall_f1_auc(pred, target, average_mode)
    return aucs

def specificity(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: AUC.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    # confusion_matrix = calculate_confusion_matrix(pred, target)
    # with torch.no_grad():
    #     res = confusion_matrix.sum(1)
    #     if average_mode == 'macro':
    #         res = float(res.sum().numpy())
    #     elif average_mode == 'none':
    #         res = res.numpy()
    #     else:
    #         raise ValueError(f'Unsupport type of averaging {average_mode}.')  
    _, _, specificity, _, _ = specificity_precision_recall_f1_auc(pred, target, average_mode)
    return specificity

def precision(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Precision.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=0.):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Recall.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=0.):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: F1 score.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Support.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res

