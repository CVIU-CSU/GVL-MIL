import argparse
import json
import collections
import random
import pandas as pd    
# criteria
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from tabulate import tabulate
from eval_metrics.glossary import *
from tqdm import tqdm
import os
import re
import jieba
from multiclass_metrics import multiclass_metrics
# from openpyxl import Workbook
# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# load_jsonl
def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

# 用于判定疾病类型，用于多分类
keywords = {
    "未见明显异常": 0,
    "血": 1,
    "分界线": 2,
    "附着": 2,
    "迂曲": 2,
    "渗出灶": 3
}

def extract_answers(text):
    # 定义正则表达式模式
    desc_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    # 提取内容
    desc_match = re.search(desc_pattern, text)
    answer_match = re.search(answer_pattern, text)
    result = {}
    if desc_match:
        result['description'] = desc_match.group(1).strip()
    if answer_match:
        result['answer'] = answer_match.group(1).strip()
    
    return result

# 使用 jieba 进行分词
def cut_words(sentence, remove_punc=False):
    res = []
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~,.，。、；：‘’“”？》《{}【】'''
    if remove_punc:
        for char in sentence:
            if char in punctuation:
                sentence = sentence.replace(char, " ")
    # 用 jieba 切分
    seg_list = jieba.cut(sentence, cut_all=False)
    if remove_punc:
        for s in seg_list:
            if s != " ":
                res.append(s)
    else:
        res = list(seg_list)
    return res

# print(cut_words("颞侧周边网膜上可见灰白色分界线改变，血管旁见红色不规则形区。"))

def confusion_mat(pred, gt):
    # 确保 pred 和 gt 长度一致
    assert len(pred) == len(gt), "预测和真实标签长度不一致！"

    # 生成混淆矩阵
    cm = confusion_matrix(gt, pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 按行归一化
    # 获取类别数（用于设置标签）
    # classes = sorted(list(set(gt) | set(pred)))  # 所有出现过的类别
    # num_classes = len(classes)
    # plt.rcParams.update({'font.size': 14})
    # 绘图
    plt.figure(figsize=(5,4))

    # 使用 seaborn 热力图绘制混淆矩阵
    ax = sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.3f', 
        cmap='PuBu',
        xticklabels=["NEG", "RH", "ROP", "WS"],
        yticklabels=["NEG", "RH", "ROP", "WS"],
        )

    # 设置标签
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14) 
    # 调整布局
    plt.tight_layout()
    # 显示图像
    plt.show()