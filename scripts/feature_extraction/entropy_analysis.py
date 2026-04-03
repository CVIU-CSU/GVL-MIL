import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

def plot_layer_entropy_stats(
    *entropy_datas,  # 支持多个熵数据集
    labels=None,
    colors=None,
    layer_names: list = None,
    title='Image Entropy Statistics by Layer',
    font_size=20,
    show_legend=True
):
    """
    绘制各层熵统计数据图表，展示均值及标准差。
    :param entropy_datas: 可变数量的熵数据数组
    :param layer_names: 图层名称列表
    :param title: 主图标题
    :param font_size: 字体大小
    :param show_legend: 是否显示图例
    :return: 返回生成的图形对象
    """
    if labels is None:
        labels = [f"Line {i}" for i in range(len(entropy_datas))]
    if colors is None:
        colors = viridis(np.linspace(0, 1, len(entropy_datas)))  # 根据数据集数量分配颜色
    
    fig1, ax1 = plt.subplots(figsize=(14,6), dpi=100)
    for idx, entropy_data in enumerate(entropy_datas):
        if isinstance(entropy_data, list) and len(entropy_data[0]) == 0:
            entropy_data = entropy_data[1:]
        
        entropy_data = np.array(entropy_data)
        means = np.mean(entropy_data, axis=1)
        std_devs = np.std(entropy_data, axis=1)
        
        # 绘制均值线
        mean_line = ax1.plot(
            range(1, len(means) + 1), means, 'o-', 
            color=colors[idx], linewidth=2, markersize=4, 
            label=f'{labels[idx]} Mean', zorder=5)
        
        # 绘制标准差区间
        ax1.fill_between(
            range(1, len(means) + 1), 
            means - std_devs, means + std_devs, 
            alpha=0.25, color=colors[idx], 
            label=f'{labels[idx]} Dev')
    
    # 创建默认图层名称（如果未提供）
    if layer_names is None:
        layer_names = [i-1 for i in range(len(entropy_data))]
    
    # 设置主图属性
    # ax1.set_title(title, fontsize=font_size+2, pad=15)
    ax1.set_ylabel('Entropy Value', fontsize=font_size+2)
    ax1.set_xlabel('Layer', fontsize=font_size+2)
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0.5, len(layer_names) + 0.5)
    
    # 设置X轴刻度
    ax1.set_xticks(range(1, len(layer_names) + 1))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=font_size)
    
    # 添加图例
    if show_legend:
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
                  ncol=len(entropy_datas), fontsize=font_size-6, framealpha=0.9)
    
    # 调整布局
    plt.subplots_adjust(top=0.8, bottom=0.15)
    plt.savefig("Entropy_based_analysis.png")
    plt.show()