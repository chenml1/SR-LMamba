from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttention(nn.Module):
    """Cross Attention for lane detection.
    十字交叉注意力机制，使用两个不同序列计算注意力
    
    Args:
        embed_dim: 特征通道维度
        num_heads: 注意力头数量
        dropout: dropout比率
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 dropout: float = 0.0, ):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embed_dim ({embed_dim}) must be "
                             f"divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim **-0.5 

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                query: Tensor,
                context: Tensor,
                attn_mask: Optional[Tensor] = None,
                tau: float = 1.0):

        bs, n_q, _ = query.shape  
        _, n_c, _ = context.shape  
        

        q = self.q_proj(query).view(bs, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        

        k = self.k_proj(context).view(bs, n_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(bs, n_c, self.num_heads, self.head_dim).transpose(1, 2)
        

        attn_weight = (q @ k.transpose(-2, -1)) * self.scale
        

        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        

        attn_weight = attn_weight.div(tau).softmax(dim=-1)
        attn_weight = self.dropout_layer(attn_weight)
        

        context_output = attn_weight @ v
        

        context_output = context_output.transpose(1, 2).contiguous().view(bs, n_q, self.embed_dim)
        

        updated_query = self.out_proj(context_output)
        
        return updated_query, attn_weight

    @staticmethod
    def loss(pred_lanes: Tensor,
             target_lanes: Tensor,
             pred_attn_weight: Tensor):

        if len(target_lanes) == 0:
            return torch.tensor(0.0, device=pred_attn_weight.device)
            
        target_lanes = target_lanes.detach().clone().flip(-1)  # (n_pos, 72)
        pred_lanes = pred_lanes.clone().flip(-1)
        
        bs, groups, n_pos, n_prior = pred_attn_weight.shape
        

        target_lanes = target_lanes.reshape(n_pos, groups, -1).permute(1, 0, 2)  # (groups, n_pos, 72//groups)
        pred_lanes = pred_lanes.reshape(n_prior, groups, -1).permute(1, 0, 2)  # (groups, n_prior, 72//groups)
        

        valid_mask = (0 <= target_lanes) & (target_lanes < 1)

        dist = ((pred_lanes.unsqueeze(1) - target_lanes.unsqueeze(2)).abs()
                ) * valid_mask.unsqueeze(2)  # (groups, n_pos, n_prior, 72//groups)
        dist = dist.sum(-1) / (valid_mask.sum(-1).unsqueeze(2) + 1e-6)  # (groups, n_pos, n_prior)
        

        _, indices = dist.min(-1)  # (groups, n_pos)
        valid_mask = valid_mask.any(-1)  # (groups, n_pos)
        indices[~valid_mask] = 255 

        pred_attn_weight = torch.clamp(pred_attn_weight, 1e-6, 1 - 1e-6)

        loss = F.nll_loss(
            torch.log(pred_attn_weight.transpose(2, 3).mean(dim=0)),  
            indices.long(),
            ignore_index=255
        )
        
        return loss


