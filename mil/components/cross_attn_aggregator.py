import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_TYPE = 'xattn_aggregator'


class XattnAggregator(nn.Module):
    """
    Cross-Attention based Aggregator for multi-instance learning.

    This module uses a learnable query to attend to instance features via multi-head cross-attention,
    producing a query-based aggregated representation for each bag.

    Args:
        feature_dim (int): Input feature dimension of instances (D).
        hidden_dim (int): Hidden dimension for attention computation (H).
        num_heads (int): Number of attention heads (default: 8).
        dropout_prob (float): Dropout probability in attention and layers (default: 0.1).
        num_queries (int): Number of learnable query vectors (default: 1).
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout_prob: float = 0.25,
        num_queries: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.num_queries = num_queries

        # Learnable query: (Q, H)
        self.query = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Key and value projections
        self.k_proj = nn.Linear(feature_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(feature_dim, embed_dim, bias=True)

        # Multi-head cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )

        # Output projection and layer norms
        self.out_proj = nn.Linear(embed_dim, feature_dim, bias=True)
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.post_norm = nn.LayerNorm(feature_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with appropriate distributions."""
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

        # LayerNorm initialization
        nn.init.constant_(self.pre_norm.weight, 1.0)
        nn.init.constant_(self.pre_norm.bias, 0.0)
        nn.init.constant_(self.post_norm.weight, 1.0)
        nn.init.constant_(self.post_norm.bias, 0.0)

        # Initialize attention projections
        for name, param in self.attn.named_parameters():
            if 'in_proj_weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'in_proj_bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'out_proj.weight' in name:
                nn.init.xavier_normal_(param)
            elif 'out_proj.bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(
        self,
        feature_map: torch.Tensor,
        instance_mask: torch.Tensor = None,
        visualize: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the cross-attention aggregator.

        Args:
            feature_map (torch.Tensor): Input tensor of shape [B, K, L, D], where
                B is batch size, K is number of instances, L is number of patches per instance,
                and D is feature dimension.
            instance_mask (torch.Tensor, optional): Binary mask of shape [B, K], where 1 indicates
                valid instances and 0 indicates invalid/missing ones.
            visualize (bool): If True, returns attention weights over patches.

        Returns:
            torch.Tensor: Aggregated features of shape [B, K, D]. If visualize is True,
                returns tuple (aggregated_features, attention_weights).
        """
        B, K, L, D = feature_map.shape
        device = feature_map.device

        # Flatten instance dimension: (B*K, L, D)
        x_flat = feature_map.view(B * K, L, D)

        # Determine valid instances
        if instance_mask is not None:
            assert instance_mask.shape == (B, K), f"Mask shape {instance_mask.shape} != (B={B}, K={K})"
            valid_mask_flat = instance_mask.view(-1).bool()  # (B*K,)
        else:
            valid_mask_flat = torch.ones(B * K, dtype=torch.bool, device=device)

        # Filter valid instances
        x_valid = x_flat[valid_mask_flat]  # (N_valid, L, D)
        N_valid = x_valid.size(0)

        # Pre-LN
        x_valid = self.pre_norm(x_valid)

        # Project keys and values
        keys = self.k_proj(x_valid)  # (N_valid, L, H)
        values = self.v_proj(x_valid)  # (N_valid, L, H)

        # Repeat query for each valid instance: (N_valid, Q, H)
        queries = self.query.expand(N_valid, -1, -1)

        # Cross-attention: (N_valid, Q, H), (N_valid, Q, L)
        attn_out, attn_weights = self.attn(
            query=queries,
            key=keys,
            value=values,
            need_weights=True
        )

        # Output projection and post-LN
        attn_out = self.out_proj(attn_out)  # (N_valid, Q, D)
        attn_out = self.post_norm(attn_out)

        # Pool over queries
        attn_out = attn_out.mean(dim=1)  # (N_valid, D)

        # Scatter back to full shape
        result = torch.zeros(B * K, D, device=device, dtype=attn_out.dtype)
        result[valid_mask_flat] = attn_out

        if visualize:
            # Expand attention weights back to full shape: (N_valid, L) → (B*K, L)
            full_attn_weights = torch.zeros(B * K, L, device=device, dtype=attn_weights.dtype)
            full_attn_weights[valid_mask_flat] = attn_weights.squeeze(1)  # assume Q=1
            full_attn_weights = full_attn_weights.view(B, K, L)  # (B, K, L)
            return result.view(B, K, D), full_attn_weights

        return result.view(B, K, D), None