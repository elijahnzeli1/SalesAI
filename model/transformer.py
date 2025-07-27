import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from config import SalesAConfig
from model.moe import MoELayer

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with cross-modal support"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        assert self.head_dim * self.num_heads == self.hidden_dim, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Cross-modal attention weights
        self.cross_modal_weights = nn.Parameter(torch.ones(3, 3))  # text, vision, audio

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                modality_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply cross-modal attention weights if modality info is provided
        if modality_info is not None and modality_info.numel() > 0:
            # Ensure modality_info has the right shape
            if modality_info.dim() == 1:
                # If 1D, expand to match sequence length
                modality_info = modality_info.unsqueeze(0).expand(batch_size, -1)
            
            # Ensure modality_info matches sequence length
            if modality_info.shape[1] != seq_len:
                # Pad or truncate to match sequence length
                if modality_info.shape[1] < seq_len:
                    # Pad with zeros (text modality)
                    pad_size = seq_len - modality_info.shape[1]
                    modality_info = torch.cat([
                        modality_info, 
                        torch.zeros(batch_size, pad_size, dtype=modality_info.dtype, device=modality_info.device)
                    ], dim=1)
                else:
                    # Truncate
                    modality_info = modality_info[:, :seq_len]
            
            # Apply cross-modal weights
            cross_weights = self.cross_modal_weights[modality_info.unsqueeze(-1), modality_info.unsqueeze(-2)]
            scores = scores + cross_weights.unsqueeze(1).unsqueeze(1)  # Add to attention scores

        # Apply mask if provided and has correct dimensions
        if mask is not None and mask.numel() > 0:
            # Ensure mask has correct dimensions by checking and resizing
            if mask.shape[1] == seq_len and mask.shape[2] == seq_len:
                # Expand mask to match scores dimensions [batch_size, num_heads, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                scores = scores.masked_fill(mask == 0, -1e9)
            else:
                # If mask dimensions don't match, create a simple causal mask
                causal_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=scores.device))
                causal_mask = causal_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                scores = scores.masked_fill(causal_mask == 0, -1e9)

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Concatenate heads - fix the reshaping
        # context shape: [batch_size, num_heads, seq_len, head_dim]
        context = context.transpose(1, 2).contiguous()
        # Now context shape: [batch_size, seq_len, num_heads, head_dim]
        
        # Calculate the correct size for reshaping
        total_size = context.numel()
        expected_size = batch_size * seq_len * self.hidden_dim
        
        if total_size != expected_size:
            # If sizes don't match, reshape to the correct size
            context = context.view(batch_size, seq_len, -1)
            # Ensure the last dimension matches hidden_dim
            if context.size(-1) != self.hidden_dim:
                # Pad or truncate to match hidden_dim
                if context.size(-1) < self.hidden_dim:
                    # Pad with zeros
                    pad_size = self.hidden_dim - context.size(-1)
                    context = torch.cat([context, torch.zeros(batch_size, seq_len, pad_size, device=context.device)], dim=-1)
                else:
                    # Truncate
                    context = context[:, :, :self.hidden_dim]
        else:
            context = context.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.output(context)

        return output

class TransformerBlock(nn.Module):
    """Transformer block with MoE integration and proper normalization"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config

        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_dim)

        # MoE layer instead of standard FFN
        self.moe = MoELayer(config)
        self.moe_norm = nn.LayerNorm(config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                modality_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual connection
        attention_output = self.attention(x, mask, modality_info)
        x = self.attention_norm(x + self.dropout(attention_output))

        # MoE with residual connection
        moe_output = self.moe(x)
        x = self.moe_norm(x + self.dropout(moe_output))

        return x