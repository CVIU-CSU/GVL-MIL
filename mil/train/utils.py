import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from numbers import Number
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
import random
import torch
import torch.nn.functional as F
import os

def entropy(x, dim=-1, keepdim=False):
    """
    计算张量在指定维度上的熵
    x: shape (..., D)
    返回: shape (...,) 或 (..., 1)
    """
    # 先归一化为概率分布（softmax）
    prob = F.softmax(x, dim=dim)
    # 熵: H(p) = -sum(p * log(p))
    log_prob = F.log_softmax(x, dim=dim)
    ent = -torch.sum(prob * log_prob, dim=dim, keepdim=keepdim)
    return ent

def entropy_based_fusion(image_features, text_features, temp=1.0):
    """
    基于熵的加权融合，instance-level 独立
    
    Args:
        image_features: (B, N, D)
        text_features:  (B, N, D)
        temp: 温度系数，控制权重分布的平滑度
    
    Returns:
        fused_features: (B, N, D)
        weights:        (B, N, 2) [w_img, w_text]
    """
    B, N, D = image_features.shape
    
    # Step 1: 分别计算 image 和 text 特征的“不确定性”（熵）
    # 注意：我们是对 D 维做 softmax → 熵，衡量该向量的“分散程度”
    ent_img = entropy(image_features, dim=-1, keepdim=True)  # (B, N, 1)
    ent_txt = entropy(text_features,  dim=-1, keepdim=True)  # (B, N, 1)
    
    # Step 2: 熵越小 → 置信度越高 → 权重越高
    # 所以用 -entropy 作为 logits
    logits = torch.cat([-ent_img, -ent_txt], dim=-1)  # (B, N, 2)
    
    # Step 3: 温度缩放 + softmax 得到归一化权重
    # temp 越小，权重越偏向置信度高的模态
    weights = F.softmax(logits / temp, dim=-1)  # (B, N, 2)
    
    # Step 4: 加权融合
    # 扩展权重以匹配 D 维
    weights_img = weights[..., 0:1]  # (B, N, 1)
    weights_txt = weights[..., 1:2]  # (B, N, 1)
    
    fused = weights_img * image_features + weights_txt * text_features  # (B, N, D)
    
    return fused, weights


def entropy_weights_from_logits(logits_list, temp=1.0):
    """
    基于多个 logits 输出的熵来生成加权平均的权重。
    
    Args:
        logits_list: list of tensors, each shape (B, C)
        temp: 温度系数
    
    Returns:
        weights: tensor (B, len(logits_list))
    """
    # 计算每个模型输出 logits 的熵
    entropies = []
    for logits in logits_list:
        ent = entropy(logits, dim=-1, keepdim=True)  # (B, 1)
        entropies.append(ent)
    
    # stack: (B, N_models, 1)
    entropies = torch.stack(entropies, dim=1)  # (B, 2, 1)
    
    # 熵越小，权重越大
    logits_for_weight = -entropies.squeeze(-1)  # (B, 2)
    
    # softmax 归一化
    weights = F.softmax(logits_for_weight / temp, dim=-1)  # (B, 2)
    
    return weights

def print_metrics(train_metrics, test_metrics):
    metrics = [{"Set": "Train"}, {"Set": "Test"}]
    for key in train_metrics.keys():
        metrics[0][key] = train_metrics[key]
        metrics[1][key] = test_metrics[key]
    titles = ["Set", "AUC", "sensitivity", "specificity", "F1", "ACC", "precision"]
    headers = {title: title for title in titles}
    print(tabulate(metrics, headers=headers, tablefmt="fancy_grid"))

def setup_seed(seed=42, full_deterministic=False):
    # 基础固定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # GPU相关设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 关键修改点：
        torch.backends.cudnn.deterministic = full_deterministic  # 通常设为False
        torch.backends.cudnn.benchmark = not full_deterministic   # 通常设为True

def specificity_precision_recall_f1_auc_acc(pred, target, average_mode='macro', thrs=0.):
    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupported averaging mode {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert isinstance(pred, np.ndarray) and isinstance(target, np.ndarray), \
        'pred and target must be torch.Tensor or np.ndarray'

    if isinstance(thrs, Number):
        thrs = (thrs,)
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError('thrs must be a number or tuple')

    pred = softmax(pred, axis=1)
    label = np.indices(pred.shape)[1]
    pred_label = np.argmax(pred, axis=1)
    pred_score = np.max(pred, axis=1)

    precisions, recalls, specificitys, f1_scores, aucs, accs = [], [], [], [], [], []

    for thr in thrs:
        _pred_label = pred_label.copy()
        # if thr is not None:
        #     _pred_label[pred_score <= thr] = -1
        if thr is not None:
            # 统计低置信度样本数量
            low_conf_mask = pred_score < thr
            num_low_conf = np.sum(low_conf_mask)
            print(f"[DEBUG] Threshold {thr}: {num_low_conf}/{len(pred_score)} samples marked as 0")
            
            _pred_label[low_conf_mask] = 0

        pred_positive = label == _pred_label.reshape(-1, 1)
        pred_negative = ~pred_positive
        gt_positive = label == target.reshape(-1, 1)
        gt_negative = ~gt_positive

        precision = (pred_positive & gt_positive).sum(0) / np.maximum(pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(gt_positive.sum(0), 1) * 100
        specificity = (pred_negative & gt_negative).sum(0) / np.maximum(gt_negative.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall, 1e-20)
        accuracy = ((pred_positive & gt_positive).sum(0) +
                    (pred_negative & gt_negative).sum(0)) / \
                    np.maximum(pred_positive.sum(0) + pred_negative.sum(0), 1) * 100

        if average_mode == 'macro':
            precisions.append(float(precision.mean()))
            recalls.append(float(recall.mean()))
            specificitys.append(float(specificity.mean()))
            f1_scores.append(float(f1_score.mean()))
            accs.append(float(accuracy.mean()))
        else:  # 'none'
            precisions.append(precision)
            recalls.append(recall)
            specificitys.append(specificity)
            f1_scores.append(f1_score)
            accs.append(accuracy)

    # AUC calculation
    y_true_bin = label_binarize(target, classes=range(pred.shape[1]))
    aucs_per_class = [
        roc_auc_score(y_true_bin[:, i], pred[:, i]) * 100
        for i in range(pred.shape[1])
    ]
    aucs = np.mean(aucs_per_class) if average_mode == "macro" else np.array(aucs_per_class)

    if return_single:
        return (precisions[0], recalls[0], specificitys[0],
                f1_scores[0], aucs, accs[0])
    return precisions, recalls, specificitys, f1_scores, aucs, accs

def visualize_errors(matrix, save_path, labels=['NEG', 'RH', 'ROP', 'WS']):
    # print("confusion matrix")
    # print(matrix)
    matrix = np.array(matrix)
    # 创建画布
    plt.figure(figsize=(5,4))
    row_normalized = matrix.astype('float') / matrix.sum(axis=1, keepdims=True)
    # 创建热力图
    sns.heatmap(
        row_normalized, 
        annot=True, 
        fmt=".2f",
        cmap="rocket_r",
        xticklabels=labels,
        yticklabels=labels 
    )

    # 添加标签
    plt.title(f"Confusion Error", fontsize=14)
    plt.xlabel("Prediction", fontsize=12)
    plt.ylabel("Ground Truth", fontsize=12)

    # 显示图形
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()

def print_label_metrics(metrics, labels=['NEG', 'RH', 'ROP', 'WS']):
    table_data = []
    headers = ["Class(%)", "AUC", "Sensitivity", "Specificity","ACC", "F1", "Precision"]
    for i in range(len(labels)):
        table_data.append([labels[i]] + [metrics[h][i] for h in headers[1:]])

    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)