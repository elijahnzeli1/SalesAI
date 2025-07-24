import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import SalesAConfig

class Expert(nn.Module):
    """Individual expert in the MoE layer"""
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Router(nn.Module):
    """Router for selecting experts"""
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            gates: Softmax probabilities for each expert
            indices: Selected expert indices
        """
        router_logits = self.router(x)
        gates = F.softmax(router_logits, dim=-1)
        # Select top-k experts (hard-coded to 2 by default)
        top_k_gates, top_k_indices = torch.topk(gates, k=2, dim=-1)
        return top_k_gates, top_k_indices

class MoELayer(nn.Module):
    """Mixture of Experts layer with vectorized dispatch"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.experts = nn.ModuleList([
            Expert(config.hidden_dim, config.intermediate_dim, config.dropout_rate)
            for _ in range(config.num_experts)
        ])
        self.router = Router(config.hidden_dim, config.num_experts)
        self.register_buffer('expert_usage', torch.zeros(config.num_experts))
        self.expert_usage: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vectorized MoE dispatch
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        gates, expert_indices = self.router(x)
        k = self.top_k
        gates_flat = gates.view(-1, k)
        indices_flat = expert_indices.view(-1, k)
        output_flat = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            mask = indices_flat.eq(expert_id)
            if not mask.any():
                continue
            positions, slots = mask.nonzero(as_tuple=True)
            weights = gates_flat[positions, slots].unsqueeze(1)
            inputs = x_flat[positions]
            outs = self.experts[expert_id](inputs)
            output_flat.index_add_(0, positions, outs * weights)
            self.expert_usage[expert_id] += mask.sum().item()
        return output_flat.view(B, S, D)

    def get_load_balancing_loss(self) -> torch.Tensor:
        """Calculate load balancing loss to encourage even expert usage"""
        usage_normalized = self.expert_usage / (self.expert_usage.sum() + 1e-8)
        mean_usage = usage_normalized.mean()
        std_usage = usage_normalized.std()
        load_balance_loss = (std_usage / (mean_usage + 1e-8)) ** 2
        return load_balance_loss 