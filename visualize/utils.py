import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import math

def visualize_instance_weights(
    image_paths,
    weights1,  # 第一组权重 [num_instances]
    weights2,  # 第二组权重 [num_instances]
    pred_labels,
    title=None,
    max_cols_per_row=15,  # 每行最大列数
    font_size=18,
    bar_width=0.35  # 每个bar的宽度
):
    """
    Visualize a MIL bag with two sets of weights side by side for each instance.
    If more than 12 instances, split into two rows.
    """
    pred_labels = int(pred_labels)
    
    # 处理权重
    weights1 = weights1[0,:,pred_labels]
    weights1 = weights1 / sum(weights1)  # 归一化
    weights2 = weights2[0,:,pred_labels]
    weights2 = weights2 / sum(weights2)  # 归一化
    
    if torch.is_tensor(weights1):
        weights1 = weights1.detach().cpu().numpy()
    if torch.is_tensor(weights2):
        weights2 = weights2.detach().cpu().numpy()

    N = len(image_paths)
    
    # 判断是否需要分两行
    if N > 15:
        # 分两行显示
        rows = 2
        cols_per_row = math.ceil(N / rows)
        fig_height = 6  # 两行时增加高度
    else:
        # 单行显示
        rows = 1
        cols_per_row = min(max_cols_per_row, N)
        fig_height = 3
    
    fig_width = cols_per_row * 1.2  # 每个实例需要两个bar的宽度
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    max_y = max(max(weights1), max(weights2)) * 1.5
    if cols_per_row > 10:
        font_size += 2
    if rows == 2:
        # 创建两行的子图布局
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # 第一行：前一半实例
        split_point = math.ceil(N / 2)
        weights1_1 = weights1[:split_point]
        weights2_1 = weights2[:split_point]
        x1 = np.arange(1, len(weights1_1) + 1)
        
        # 绘制第一组权重
        ax1.bar(x1 - bar_width/2, weights1_1, width=bar_width, color="#ccbee6", label="Image Modality")
        # 绘制第二组权重
        ax1.bar(x1 + bar_width/2, weights2_1, width=bar_width, color="#54a56c", label="Text Modality")
        
        ax1.set_xlabel("Instance Index", fontsize=font_size)
        ax1.set_ylabel("Weight", fontsize=font_size)
        ax1.tick_params(axis='x', labelsize=font_size)
        ax1.tick_params(axis='y', labelsize=font_size)
        ax1.set_xticks(x1)
        ax1.set_yticks(np.arange(0, max_y * 1.2, 0.05))
        ax1.legend(fontsize=font_size-4)
        
        # 第二行：后一半实例
        weights1_2 = weights1[split_point:]
        weights2_2 = weights2[split_point:]
        x2 = np.arange(split_point + 1, split_point + len(weights1_2) + 1)
        
        # 绘制第一组权重
        ax2.bar(x2 - bar_width/2, weights1_2, width=bar_width, color="#ccbee6", label="Image Modality")
        # 绘制第二组权重
        ax2.bar(x2 + bar_width/2, weights2_2, width=bar_width, color="#54a56c", label="Text Modality")
        
        ax2.set_xlabel("Instance Index", fontsize=font_size)
        ax2.set_ylabel("Weight", fontsize=font_size)
        ax2.tick_params(axis='x', labelsize=font_size)
        ax2.tick_params(axis='y', labelsize=font_size)
        ax2.set_xticks(x2)
        ax2.set_yticks(np.arange(0, max_y * 1.2, 0.05))
        ax2.legend(fontsize=font_size-4)
        
    else:
        # 单行显示
        ax_bar = fig.add_subplot(111)
        x = np.arange(1, N + 1)
        
        # 绘制第一组权重
        ax_bar.bar(x - bar_width/2, weights1, width=bar_width, color="#ccbee6", label="Image Modality")
        # 绘制第二组权重
        ax_bar.bar(x + bar_width/2, weights2, width=bar_width, color="#54a56c", label="Text Modality")
        
        ax_bar.set_xlabel("Instance Index", fontsize=font_size)
        ax_bar.set_ylabel("Weight", fontsize=font_size)
        ax_bar.tick_params(axis='x', labelsize=font_size)
        ax_bar.tick_params(axis='y', labelsize=font_size)
        ax_bar.set_xticks(x)
        ax_bar.set_yticks(np.arange(0, max_y * 1.2, 0.1))
        ax_bar.legend(fontsize=font_size-2)

    # if title is not None:
    #     plt.suptitle(title, fontsize=font_size, y=1)
    plt.tight_layout()
    plt.savefig(os.path.join('../../asserts', "_".join(title.split())))
    plt.show()


# 假设 attention 是你的注意力矩阵，形状为 [1, 729]
# 示例：attention = torch.randn(1, 729).numpy()
# 若是 torch.Tensor，请先 attention = attention.cpu().numpy()

# attention = np.random.rand(1, 729)  # 替换成你的实际数据
def visualize_instance(attention, instance_idx=None, title=None):
    if instance_idx is None:
        instance_idx = 2
    if attention.ndim == 3:
        attention = attention[:, instance_idx, :]
    # reshape 成 27 × 27
    att_map = attention.reshape(27, 27)

    plt.figure(figsize=(5,4))
    plt.imshow(att_map, cmap='CMRmap_r')
    # plt.imshow(att_map, cmap='jet', interpolation='nearest')
    plt.colorbar(shrink=0.7)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # 修改标题和轴字体
    # if title is not None:
    #     plt.title(title, fontsize=16)
    # else:
    #     plt.title("Attention Heatmap (27×27)", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join('../../asserts', "_".join(title.split())))
    plt.show()

# visualize_instance(outputs['token attention'])