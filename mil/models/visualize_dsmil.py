import os
from typing import Optional, List, Tuple, Dict, Union
import pathlib
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import cv2
from PIL import Image
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel
import sys 
sys.path.insert(0, "/root/userfolder/MIL/VL-MIL")
from mil.models.mil_template import MIL
from mil.models.layers import create_mlp
from mil.losses import kl_align_loss


# --- Visualization Tools ---

class DSMILVisualizer:
    """可视化工具类，用于DSMIL模型的可视化分析"""
    
    def __init__(self, model: 'DSMIL', device: torch.device = None):
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        
    def create_attention_heatmap(self, 
                                h: torch.Tensor, 
                                attention_weights: torch.Tensor,
                                original_patch_coords: List[Tuple[int, int]] = None,
                                slide_size: Tuple[int, int] = (1000, 1000),
                                return_fig: bool = True):
        """
        创建注意力热力图
        
        Args:
            h: 输入特征 [B, M, D]
            attention_weights: 注意力权重 [B, M, C]
            original_patch_coords: 原始贴片坐标列表 [(x1, y1), (x2, y2), ...]
            slide_size: 整个WSI的尺寸 (width, height)
            return_fig: 是否返回matplotlib figure对象
        """
        batch_size, num_instances, num_classes = attention_weights.shape
        
        figures = []
        
        for batch_idx in range(batch_size):
            # 获取当前batch的注意力权重
            batch_attention = attention_weights[batch_idx]  # [M, C]
            
            fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 5))
            
            # 对每个类别创建热力图
            for class_idx in range(num_classes):
                class_attention = batch_attention[:, class_idx].cpu().numpy()
                
                if original_patch_coords is not None:
                    # 创建基于坐标的热力图
                    heatmap = np.zeros(slide_size)
                    count_map = np.zeros(slide_size)
                    
                    for patch_idx, (x, y) in enumerate(original_patch_coords):
                        if patch_idx < len(class_attention):
                            # 简化处理：假设每个贴片是固定大小
                            patch_size = 224
                            heatmap[y:y+patch_size, x:x+patch_size] += class_attention[patch_idx]
                            count_map[y:y+patch_size, x:x+patch_size] += 1
                    
                    # 平均处理
                    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map!=0)
                    
                    im = axes[class_idx].imshow(heatmap, cmap='hot', interpolation='bilinear')
                    axes[class_idx].set_title(f'Class {class_idx} Attention')
                    plt.colorbar(im, ax=axes[class_idx])
                else:
                    # 简单的注意力分布图
                    axes[class_idx].bar(range(len(class_attention)), class_attention)
                    axes[class_idx].set_title(f'Class {class_idx} Attention')
                    axes[class_idx].set_xlabel('Instance Index')
                    axes[class_idx].set_ylabel('Attention Weight')
            
            # 总体注意力分布（所有类别的平均）
            overall_attention = batch_attention.mean(dim=1).cpu().numpy()
            if original_patch_coords is not None:
                heatmap = np.zeros(slide_size)
                count_map = np.zeros(slide_size)
                
                for patch_idx, (x, y) in enumerate(original_patch_coords):
                    if patch_idx < len(overall_attention):
                        patch_size = 224
                        heatmap[y:y+patch_size, x:x+patch_size] += overall_attention[patch_idx]
                        count_map[y:y+patch_size, x:x+patch_size] += 1
                
                heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map!=0)
                
                im = axes[-1].imshow(heatmap, cmap='hot', interpolation='bilinear')
                axes[-1].set_title('Overall Attention')
                plt.colorbar(im, ax=axes[-1])
            else:
                axes[-1].bar(range(len(overall_attention)), overall_attention)
                axes[-1].set_title('Overall Attention')
                axes[-1].set_xlabel('Instance Index')
                axes[-1].set_ylabel('Attention Weight')
            
            plt.tight_layout()
            figures.append(fig)
            
            if not return_fig:
                plt.close(fig)
        
        return figures if batch_size > 1 else figures[0]
    
    def visualize_feature_space(self, 
                              h: torch.Tensor,
                              attention_weights: torch.Tensor,
                              instance_predictions: torch.Tensor,
                              method: str = 'tsne',
                              **kwargs):
        """
        可视化特征空间，展示双流一致性
        
        Args:
            h: 实例特征 [B, M, D]
            attention_weights: 注意力权重 [B, M, C]
            instance_predictions: 实例级预测 [B, M, C]
            method: 降维方法 'tsne' 或 'umap'
        """
        batch_size, num_instances, _ = h.shape
        
        figures = []
        
        for batch_idx in range(batch_size):
            features = h[batch_idx].detach().cpu().numpy()  # [M, D]
            attention = attention_weights[batch_idx].detach().cpu().numpy()  # [M, C]
            instance_preds = instance_predictions[batch_idx].detach().cpu().numpy()  # [M, C]
            
            # 降维
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, **kwargs)
            else:  # umap
                reducer = umap.UMAP(n_components=2, random_state=42, **kwargs)
            
            embeddings_2d = reducer.fit_transform(features)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1. 按注意力权重着色（使用最大注意力类别）
            max_attention_class = np.argmax(attention, axis=1)
            scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=max_attention_class, cmap='tab10', alpha=0.7)
            axes[0].set_title('Colored by Max Attention Class')
            plt.colorbar(scatter1, ax=axes[0])
            
            # 2. 按实例预测着色（使用最大预测概率类别）
            max_pred_class = np.argmax(instance_preds, axis=1)
            scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=max_pred_class, cmap='tab10', alpha=0.7)
            axes[1].set_title('Colored by Instance Prediction')
            plt.colorbar(scatter2, ax=axes[1])
            
            # 3. 按注意力权重大小着色（透明度表示权重）
            max_attention_values = np.max(attention, axis=1)
            scatter3 = axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=max_attention_values, cmap='YlOrRd', alpha=0.7)
            axes[2].set_title('Colored by Attention Strength')
            plt.colorbar(scatter3, ax=axes[2])
            
            plt.tight_layout()
            figures.append(fig)
        
        return figures if batch_size > 1 else figures[0]
    
    def visualize_top_instances(self,
                              h: torch.Tensor,
                              attention_weights: torch.Tensor,
                              instance_images: List[Image.Image] = None,
                              top_k: int = 10,
                              class_index: int = None):
        """
        可视化Top-K注意力实例
        
        Args:
            h: 实例特征
            attention_weights: 注意力权重
            instance_images: 对应的实例图像列表
            top_k: 显示前K个实例
            class_index: 指定类别，如果为None则使用最大注意力
        """
        batch_size, num_instances, num_classes = attention_weights.shape
        
        if class_index is None:
            # 使用总体注意力（各类别平均）
            attention_scores = attention_weights.mean(dim=2)  # [B, M]
        else:
            attention_scores = attention_weights[:, :, class_index]  # [B, M]
        
        all_figures = []
        
        for batch_idx in range(batch_size):
            batch_scores = attention_scores[batch_idx]
            top_indices = torch.topk(batch_scores, min(top_k, len(batch_scores)), 
                                   largest=True, sorted=True).indices
            
            fig, axes = plt.subplots(2, (top_k + 1) // 2, figsize=(15, 8))
            axes = axes.flatten() if top_k > 1 else [axes]
            
            for idx, (ax, instance_idx) in enumerate(zip(axes, top_indices)):
                if instance_images is not None and instance_idx < len(instance_images):
                    # 显示图像
                    ax.imshow(instance_images[instance_idx])
                    ax.set_title(f'Rank {idx+1}\nAttn: {batch_scores[instance_idx]:.3f}')
                else:
                    # 显示特征统计信息
                    instance_feat = h[batch_idx, instance_idx]
                    ax.hist(instance_feat.detach().cpu().numpy(), bins=20, alpha=0.7)
                    ax.set_title(f'Rank {idx+1}\nAttn: {batch_scores[instance_idx]:.3f}\nFeature Histogram')
                
                ax.axis('off')
            
            # 隐藏多余的子图
            for idx in range(len(top_indices), len(axes)):
                axes[idx].axis('off')
            
            class_info = f"Class {class_index}" if class_index is not None else "Overall"
            plt.suptitle(f'Top-{len(top_indices)} Instances by Attention ({class_info})', fontsize=16)
            plt.tight_layout()
            all_figures.append(fig)
        
        return all_figures if batch_size > 1 else all_figures[0]
    
    def create_attention_distribution_plot(self, attention_weights: torch.Tensor):
        """创建注意力分布直方图和排序图"""
        batch_size, num_instances, num_classes = attention_weights.shape
        
        figures = []
        
        for batch_idx in range(batch_size):
            batch_attention = attention_weights[batch_idx]  # [M, C]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 注意力分布直方图
            for class_idx in range(num_classes):
                class_attention = batch_attention[:, class_idx].cpu().numpy()
                axes[0, 0].hist(class_attention, bins=20, alpha=0.7, 
                              label=f'Class {class_idx}')
            axes[0, 0].set_xlabel('Attention Weight')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Attention Distribution by Class')
            axes[0, 0].legend()
            
            # 2. 注意力排序图（衰减曲线）
            for class_idx in range(num_classes):
                class_attention = batch_attention[:, class_idx].cpu().numpy()
                sorted_attention = np.sort(class_attention)[::-1]  # 降序排列
                axes[0, 1].plot(range(len(sorted_attention)), sorted_attention, 
                              label=f'Class {class_idx}', alpha=0.7)
            axes[0, 1].set_xlabel('Instance Rank')
            axes[0, 1].set_ylabel('Attention Weight')
            axes[0, 1].set_title('Attention Sorted (Descending)')
            axes[0, 1].legend()
            axes[0, 1].set_yscale('log')
            
            # 3. 各类别注意力相关性热力图
            attention_corr = np.corrcoef(batch_attention.cpu().numpy().T)
            im = axes[1, 0].imshow(attention_corr, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_xticks(range(num_classes))
            axes[1, 0].set_yticks(range(num_classes))
            axes[1, 0].set_xlabel('Class')
            axes[1, 0].set_ylabel('Class')
            axes[1, 0].set_title('Inter-class Attention Correlation')
            plt.colorbar(im, ax=axes[1, 0])
            
            # 4. 实例级注意力统计
            instance_total_attention = batch_attention.sum(dim=1).cpu().numpy()
            axes[1, 1].hist(instance_total_attention, bins=20, alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Total Attention (Sum over Classes)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Instance Total Attention Distribution')
            
            plt.tight_layout()
            figures.append(fig)
        
        return figures if batch_size > 1 else figures[0]
    
    def generate_comprehensive_report(self, 
                                   h: torch.Tensor,
                                   attention_weights: torch.Tensor,
                                   instance_predictions: torch.Tensor,
                                   bag_prediction: torch.Tensor,
                                   instance_images: List[Image.Image] = None,
                                   top_k: int = 10):
        """生成综合可视化报告"""
        print("=== DSMIL Model Visualization Report ===")
        
        # 1. 注意力热力图
        print("\n1. Generating attention heatmaps...")
        heatmap_fig = self.create_attention_heatmap(h, attention_weights)
        
        # 2. 特征空间可视化
        print("2. Generating feature space visualization...")
        feature_fig = self.visualize_feature_space(h, attention_weights, instance_predictions)
        
        # 3. Top-K实例可视化
        print("3. Generating top instances visualization...")
        topk_fig = self.visualize_top_instances(h, attention_weights, instance_images, top_k=top_k)
        
        # 4. 注意力分布分析
        print("4. Generating attention distribution analysis...")
        dist_fig = self.create_attention_distribution_plot(attention_weights)
        
        # 5. 预测结果摘要
        print("5. Generating prediction summary...")
        bag_probs = F.softmax(bag_prediction, dim=1)
        pred_class = torch.argmax(bag_probs, dim=1)
        
        print(f"Bag Prediction: {pred_class.cpu().numpy()}")
        print(f"Prediction Probabilities: {bag_probs.detach().cpu().numpy()}")
        
        return {
            'heatmap': heatmap_fig,
            'feature_space': feature_fig,
            'top_instances': topk_fig,
            'attention_distribution': dist_fig,
            'predictions': {
                'class': pred_class.cpu().numpy(),
                'probabilities': bag_probs.detach().cpu().numpy()
            }
        }


# --- Core Model Components (保持不变) ---

class IClassifier(nn.Module):
    """Instance-level classifier."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.inst_classifier = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B x M x D]
        c = self.inst_classifier(h)  # B x M x C
        return c


class BClassifier(nn.Module):
    """Bag-level classifier with attention."""

    def __init__(self, in_dim: int, attn_dim: int = 384, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)
        self.fcc = nn.Conv1d(3, 3, kernel_size=in_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor, attn_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        device = h.device
        V = self.v(h)  # B x M x D
        Q = self.q(h)  # B x M x D_attn

        # Sort instances by class scores to find critical instances
        _, m_indices = torch.sort(c, dim=1, descending=True)

        # Select features of top instances for each class
        m_feats = torch.stack(
            [torch.index_select(h_i, dim=0, index=m_indices_i[0, :]) for h_i, m_indices_i in zip(h, m_indices)], 0
        )

        q_max = self.q(m_feats)  # B x C x D_attn
        # Attention mechanism: I think this could be the error?
        A = torch.bmm(Q, q_max.transpose(1, 2))  # B x M x C
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=2) * torch.finfo(A.dtype).min

        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=device)),
                      dim=1)  # Softmax over M

        # Aggregate features

        B = torch.bmm(A.transpose(1, 2), V)  # B x C x D

        B = self.norm(B)
        return B, A


# --- Main DSMIL Module (添加可视化功能) ---

class DSMIL(MIL):
    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            dropout_v: float = 0.0,
            num_classes: int = 2
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False
        )
        self.i_classifier = IClassifier(in_dim=embed_dim, num_classes=num_classes)
        self.b_classifier = BClassifier(in_dim=embed_dim, attn_dim=attn_dim, dropout=dropout_v)
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=embed_dim)
        
        # 添加可视化器
        self.visualizer = None
        self.initialize_weights()

    def setup_visualizer(self, device: torch.device = None):
        """设置可视化器"""
        if device is None:
            device = next(self.parameters()).device
        self.visualizer = DSMILVisualizer(self, device)

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = False, calc_kibsp=False, label=None) -> tuple[
        torch.Tensor, dict]:
        h = self.patch_embed(h)
        instance_classes = self.i_classifier(h)
        slide_feats, attention = self.b_classifier(h, instance_classes, attn_mask=attn_mask)
        intermeds = {'instance_classes': instance_classes}
        if return_attention:
            intermeds['attention'] = attention
        # new loss
        # ---- Compute BP loss ----
        if calc_kibsp:
            from mil.losses import compute_bp_loss
            B, C, D = slide_feats.shape
            weights = attention.mean(dim=-1) if attention is not None else torch.ones(B, h.size(1), device=h.device)
            masks = torch.ones_like(weights)
            bp_loss = compute_bp_loss(
                bag_features=h,   # 每个实例的嵌入特征
                weights=weights,  # 注意力权重
                masks=masks,      # 有效mask
                labels=label,     # bag标签
                k=3               # top-k 超参数
            )
        else:
            bp_loss = None
        return slide_feats, intermeds, bp_loss

    def visualize(self, 
                 h: torch.Tensor, 
                 instance_images: List[Image.Image] = None,
                 patch_coords: List[Tuple[int, int]] = None,
                 **kwargs) -> Dict:
        """
        执行完整的可视化分析
        
        Args:
            h: 输入特征
            instance_images: 实例图像列表（可选）
            patch_coords: 贴片坐标列表（可选，用于热力图）
            **kwargs: 其他可视化参数
        """
        if self.visualizer is None:
            self.setup_visualizer()
        
        # 前向传播获取中间结果
        with torch.no_grad():
            slide_feats, intermeds, _ = self.forward_features(h, return_attention=True)
            bag_logits = self.forward_head(slide_feats)
            instance_logits = intermeds['instance_classes']
            attention_weights = intermeds['attention']
        
        # 生成可视化报告
        report = self.visualizer.generate_comprehensive_report(
            h=h,
            attention_weights=attention_weights,
            instance_predictions=instance_logits,
            bag_prediction=bag_logits,
            instance_images=instance_images,
            **kwargs
        )
        
        return report

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        pass

    def initialize_classifier(self, num_classes: Optional[int] = None):
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=self.embed_dim)

    def forward_head(self, slide_feats: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(slide_feats)  # B x C x 1
        return logits.squeeze(-1)

    def forward(
            self, 
            h: torch.Tensor, 
            loss_fn: nn.Module = None,
            label: torch.LongTensor = None, 
            attn_mask=None, 
            calc_kibsp: bool=False,
            return_attention: bool = False,
            return_slide_feats: bool = False) -> tuple[dict, dict]:
        slide_feats, intermeds, bp_loss = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention, calc_kibsp=calc_kibsp, label=label)
        max_instance_logits, _ = torch.max(intermeds['instance_classes'], 1)
        bag_logits = self.forward_head(slide_feats)
        logits = 0.5 * (bag_logits + max_instance_logits)
        cls_loss = self.compute_loss(loss_fn, logits, label)

        # kl loss between instance stream and bag stream
        kl_loss = kl_align_loss(
            bag_logits=bag_logits.detach(),  # detach to avoid double backprop
            inst_logits=intermeds['instance_classes'],
            topk=1
        )
        w_kl = 0.05

        results_dict = {'logits': logits, 'loss': cls_loss, 'bp_loss': bp_loss}
        log_dict = {'loss': cls_loss.item() if cls_loss is not None else -1}

        # # add kl align loss
        results_dict['loss'] += kl_loss * w_kl
        log_dict['loss'] += kl_loss.item() * w_kl

        if bp_loss is not None:
            results_dict['loss'] += bp_loss * 0.1
            log_dict['loss'] += bp_loss.item() * 0.1
        
        if not return_attention and 'attention' in log_dict:
            del log_dict['attention']
        if return_slide_feats:
            log_dict['slide_feats'] = slide_feats

        return results_dict, log_dict


# --- Hugging Face Compatible Configuration and Model (保持不变) ---

class DSMILConfig(PretrainedConfig):
    model_type = 'dsmil'

    def __init__(self,
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 attn_dim: int = 384,
                 dropout_v: float = 0.0,
                 num_classes: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.attn_dim = attn_dim
        self.dropout_v = dropout_v
        self.num_classes = num_classes


class DSMILModel(PreTrainedModel):
    config_class = DSMILConfig

    def __init__(self, config: DSMILConfig, **kwargs):
        # Override config with any kwargs provided
        self.config = config
        for key, value in kwargs.items():
            setattr(config, key, value)

        super().__init__(config)
        self.model = DSMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            dropout_v=config.dropout_v,
            num_classes=config.num_classes
        )
        # Expose the inner model's forward methods
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.initialize_classifier = self.model.initialize_classifier
        self.visualize = self.model.visualize
        self.setup_visualizer = self.model.setup_visualizer


# Register the model with Hugging Face AutoClass
AutoConfig.register(DSMILConfig.model_type, DSMILConfig)
AutoModel.register(DSMILConfig, DSMILModel)


# --- 使用示例 ---
def example_usage():
    """使用示例"""
    # 初始化模型
    model = DSMIL(in_dim=1024, embed_dim=512, num_classes=4)
    model.setup_visualizer()
    
    # 假设有一些测试数据
    batch_size, num_instances, feat_dim = 2, 100, 1024
    h = torch.randn(batch_size, num_instances, feat_dim)
    
    # 执行可视化
    report = model.visualize(
        h=h,
        top_k=2  # 显示前5个最重要的实例
    )
    os.makedirs("visualize", exist_ok=True)
    # 保存可视化结果
    for name, fig_list in report.items():
        if name != 'predictions':
            if not isinstance(fig_list, list):
                fig_list = [fig_list]

            for idx, fig in enumerate(fig_list):
                fig.savefig(f'visualize/{name}_visualization_{idx}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    print("Visualization completed and saved!")

if __name__ == "__main__":
    example_usage()