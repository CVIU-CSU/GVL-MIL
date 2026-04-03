import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn.functional as F

def kl_align_loss(bag_logits, inst_logits, topk=5):
    """
    KL-based alignment loss between bag-level and instance-level predictions.

    Args:
        bag_logits: Tensor [B, C] — bag-level logits
        inst_logits: Tensor [B, M, C] — instance-level logits
        topk: int — number of top instances to select per bag

    Returns:
        scalar loss (float tensor)
    """
    B, M, C = inst_logits.shape
    loss = 0.0
    valid_bags = 0  # count how many bags are actually used

    # Loop over each bag
    for b in range(B):
        m = M if inst_logits[b].ndim == 2 else inst_logits[b].shape[0]
        if m < topk:
            # Skip this bag entirely if insufficient instances
            continue

        with torch.no_grad():
            # Select top-k instances based on max class probability
            inst_probs = F.softmax(inst_logits[b], dim=-1)  # [M, C]
            conf, _ = inst_probs.max(dim=1)
            topk_idx = torch.topk(conf, topk).indices

        # Compute bag and instance distributions
        bag_prob = F.softmax(bag_logits[b], dim=-1)  # [C]
        inst_topk = inst_logits[b][topk_idx]         # [k, C]
        inst_prob = F.softmax(inst_topk, dim=-1)     # [k, C]

        # Compute KL divergence (symmetrized)
        kl_bi = F.kl_div(inst_prob.log(), bag_prob.unsqueeze(0).expand_as(inst_prob),
                         reduction='batchmean') + \
                F.kl_div(bag_prob.log().unsqueeze(0).expand_as(inst_prob),
                         inst_prob, reduction='batchmean')

        loss += kl_bi
        valid_bags += 1

    if valid_bags == 0:
        return torch.tensor(0.0, device=bag_logits.device)
    else:
        return loss / valid_bags
    

# def compute_bp_loss(bag_features, weights, masks, labels, k=3, margin=0.3):
#     """
#     Improved Bag-Positive alignment loss.
#     Args:
#         bag_features: [B, M, D]   bag instance features
#         weights: [B, M]           instance importance weights (e.g., attention)
#         masks: [B, M]             binary mask for valid instances
#         labels: [B]               0/1 labels
#         k: int                    top-k selection
#         margin: float             margin for contrastive loss
#     Returns:
#         bp_loss: scalar tensor
#     """
#     B, M, D = bag_features.shape
#     device = bag_features.device

#     # --- 1. Mask invalid instances ---
#     weights = weights.masked_fill(masks == 0, float('-inf'))

#     # --- 2. Sort by importance ---
#     sorted_indices = torch.argsort(weights, dim=-1, descending=True)  # [B, M]
#     sorted_feats = torch.gather(
#         bag_features, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D)
#     )

#     # --- 3. Dynamically handle small bags ---
#     valid_counts = masks.sum(dim=-1).long()
#     eff_k = torch.minimum(valid_counts // 2, torch.tensor(k, device=device))
#     eff_k = torch.clamp(eff_k, min=1)  # at least 1

#     # --- 4. Select top-k and bottom-k per bag ---
#     positives, negatives = [], []
#     for i in range(B):
#         k_i = eff_k[i].item()
#         if valid_counts[i] < 2:  # skip degenerate bags
#             continue
#         positives.append(sorted_feats[i, :k_i])
#         negatives.append(sorted_feats[i, -k_i:])
#     if len(positives) == 0:
#         return torch.tensor(0.0, device=device)

#     pos_feats = torch.stack(positives)  # [B, k, D]
#     neg_feats = torch.stack(negatives)  # [B, k, D]

#     # --- 5. Normalize ---
#     pos_norm = F.normalize(pos_feats, p=2, dim=-1)
#     neg_norm = F.normalize(neg_feats, p=2, dim=-1)

#     # --- 6. Compute cosine similarity ---
#     sim = torch.einsum('bkd,bkd->bk', pos_norm, neg_norm)  # [B, k]
#     sim_mean = sim.mean(dim=-1)  # [B]

#     # --- 7. Contrastive-style loss (margin-based) ---
#     labels = labels.float()
#     pos_loss = (1 - labels) * F.relu(sim_mean - margin)   # background bags: force sim < margin
#     neg_loss = labels * F.relu(margin - sim_mean)         # positive bags: force sim > margin
#     loss = (pos_loss + neg_loss).mean()

#     return loss

# mil/losses.py
import torch
import torch.nn.functional as F

def compute_bp_loss(
    bag_features: torch.Tensor,   # [B, M, D]
    weights: torch.Tensor,        # [B, M], attention weights (or mean over classes)
    masks: torch.Tensor,          # [B, M], valid instance mask (optional)
    labels: torch.LongTensor,     # [B], 0 for negative, 1 for positive
    k: int = 3                     # top-k for selecting instances
) -> torch.Tensor:
    """
    Compute Binary Pattern (BP) loss based on attention-weighted instance selection.

    Args:
        bag_features: [B, M, D] - features of instances
        weights: [B, M] - attention weights (e.g., from attention mechanism)
        masks: [B, M] - binary mask indicating valid instances
        labels: [B] - bag labels (0: negative, 1: positive)
        k: number of top instances to consider

    Returns:
        bp_loss: scalar tensor
    """
    B, M, D = bag_features.shape
    assert weights.shape == (B, M), "weights must be [B, M]"
    assert masks.shape == (B, M), "masks must be [B, M]"
    assert labels.shape == (B,), "labels must be [B]"

    # Apply mask and normalize weights
    weights = weights * masks  # zero out invalid instances
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
    weights = weights / weights_sum  # normalize per bag

    # Get top-k and bottom-k indices
    _, top_indices = torch.topk(weights, k=k, dim=1)  # [B, k]
    _, bot_indices = torch.topk(-weights, k=k, dim=1)  # [B, k]

    # Select top and bottom instances
    top_feats = torch.stack([bag_features[i, top_indices[i]] for i in range(B)], dim=0)  # [B, k, D]
    bot_feats = torch.stack([bag_features[i, bot_indices[i]] for i in range(B)], dim=0)  # [B, k, D]

    # Compute cosine similarity between top and bottom features
    # Use mean across k instances as representative
    top_mean = top_feats.mean(dim=1)  # [B, D]
    bot_mean = bot_feats.mean(dim=1)  # [B, D]

    # Normalize for cosine similarity
    top_norm = F.normalize(top_mean, dim=1)
    bot_norm = F.normalize(bot_mean, dim=1)

    s = torch.sum(top_norm * bot_norm, dim=1)  # [B], cosine similarity

    # Construct BP loss
    # For negative bags: maximize s → minimize (1 - s)
    # For positive bags: minimize s → minimize (1 + s)
    bp_loss = torch.where(labels == 0, 1 - s, 1 + s)
    bp_loss = bp_loss.mean()  # average over batch

    return bp_loss

class KIBSPLoss(nn.Module):
    """
    Ground Truth-aware Key-instance Guided Intra-Bag Semantic Prior (GT-K-IBSP)
    Ensures intra-bag semantic consistency w.r.t. ground truth class.
    
    Args:
        K (int): Number of key instances to select per bag.
        lambda_max (float): Weight for max deviation regularization.
        delta (float): Margin for max deviation loss.
        distance_metric (str): 'cosine' or 'l2'
    """
    def __init__(self, K=2, lambda_max=0.1, delta=0.5, distance_metric='cosine'):
        super(KIBSPLoss, self).__init__()
        self.K = K
        self.lambda_max = lambda_max
        self.delta = delta
        assert distance_metric in ['cosine', 'l2']
        self.distance_metric = distance_metric

    def compute_attributions(self, features, labels, classifier_head):
        """
        Compute attribution scores using gradient w.r.t. GT class logit.
        Assumes logits are raw outputs (before softmax).
        
        Args:
            features: (N, D) tensor of image features for a single bag
            labels: scalar (GT class index) for this bag
            classifier_head: callable, the classifier head to compute logits
        
        Returns:
            attributions: (N,) tensor of attribution scores
        """
        # Detach features to avoid backprop through encoder
        features = features.detach().requires_grad_(True)
        
        # Recompute logit for GT class
        pred_logits = classifier_head(features).squeeze(-1)  # (N, C) or (N,)
        # print(pred_logits)
        # if pred_logits.dim() == 2:
        #     pred_logits = pred_logits[:, labels]  # (N,)
        
        # Compute gradient of GT logit w.r.t. features
        grads = torch.autograd.grad(
            outputs=pred_logits.sum(),
            inputs=features,
            retain_graph=True,
            create_graph=False
        )[0]  # (N, D)

        # Attribution: gradient × feature
        attributions = (grads * features).sum(dim=1)  # (N,)
        attributions = attributions.detach()
        
        return attributions

    def forward(self, features, labels, classifier_head):
        """
        Forward pass for GT-K-IBSP loss with batch support.
        
        Args:
            features: (B, N, D) tensor of image features (B=batch size, N=instances per bag, D=feature dim)
            labels: (B,) tensor of GT class indices for each bag
            classifier_head: callable, the classifier head to compute logits
        
        Returns:
            loss: scalar tensor, GT-K-IBSP loss (averaged over batch)
        """
        B, N, D = features.shape
        batch_losses = []
        
        for i in range(B):
            # Extract single bag
            bag_features = features[i]  # (N, D)
            bag_label = labels[i]      # scalar
            
            # Skip if insufficient instances
            if N <= self.K:
                batch_losses.append(torch.tensor(0.0, device=features.device))
                continue
                
            # Compute attributions for this bag
            attributions = self.compute_attributions(bag_features, bag_label, classifier_head)
            
            # Select Top-K key instances
            _, topk_indices = torch.topk(attributions, k=self.K, largest=True)
            mask_key = torch.zeros(N, dtype=torch.bool, device=features.device)
            mask_key[topk_indices] = True
            
            # Compute key prototype
            f_key = bag_features[mask_key]  # (K, D)
            mu_key = f_key.mean(dim=0)      # (D,)
            
            # Compute distances for non-key instances
            if self.distance_metric == 'cosine':
                mu_key_norm = F.normalize(mu_key.unsqueeze(0), dim=1).squeeze(0)
                f_others = bag_features[~mask_key]  # (N-K, D)
                f_others_norm = F.normalize(f_others, dim=1)
                dists = 1 - torch.sum(f_others_norm * mu_key_norm.unsqueeze(0), dim=1)  # (N-K,)
            elif self.distance_metric == 'l2':
                f_others = bag_features[~mask_key]
                dists = torch.norm(f_others - mu_key.unsqueeze(0), dim=1)  # (N-K,)
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
            # Consistency loss (weighted by attention)
            if len(dists) > 0:
                weights = F.softmax(attributions[~mask_key], dim=0)
                loss_consistency = (weights * dists).sum()
            else:
                loss_consistency = torch.tensor(0.0, device=features.device)
            
            # Max deviation regularization
            if len(dists) > 0:
                loss_max = F.relu(dists.max() - self.delta).mean()
            else:
                loss_max = torch.tensor(0.0, device=features.device)
            
            # Total loss for this bag
            loss = loss_consistency + self.lambda_max * loss_max
            batch_losses.append(loss)
        
        # Return average loss over batch
        return torch.stack(batch_losses).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    
    Args:
        alpha: Weight for each class (positive/negative samples)
        gamma: Focusing parameter
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


LOSS_DICT = {
    'BCE': nn.BCELoss(),
    'CE': nn.CrossEntropyLoss(),
    'CBCE': nn.CrossEntropyLoss(weight=torch.tensor([0.48, 0.59, 0.90, 1.00])),
    'NLL': nn.NLLLoss(),
    'Focal': FocalLoss(alpha=0.25, gamma=2.0),
    'KIBSP': KIBSPLoss(K=2, lambda_max=0.1, delta=0.5, distance_metric='cosine')
}