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
    """Router for selecting experts with improved routing logic"""
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            gates: Softmax probabilities for each expert (batch_size, seq_len, top_k)
            indices: Selected expert indices (batch_size, seq_len, top_k)
        """
        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        
        # Apply softmax to top-k gates
        gates = F.softmax(top_k_gates, dim=-1)
        
        return gates, top_k_indices

class MoELayer(nn.Module):
    """Mixture of Experts layer with improved vectorized dispatch"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.experts = nn.ModuleList([
            Expert(config.hidden_dim, config.intermediate_dim, config.dropout_rate)
            for _ in range(config.num_experts)
        ])
        self.router = Router(config.hidden_dim, config.num_experts, config.top_k)
        
        # Register expert usage tracking
        self.register_buffer('expert_usage', torch.zeros(config.num_experts))
        self.expert_usage: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved MoE dispatch
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            Output tensor of same shape as input
        """
        B, S, D = x.shape
        
        # Get routing decisions
        gates, expert_indices = self.router(x)  # (B, S, top_k), (B, S, top_k)
        
        # Flatten for processing
        x_flat = x.view(-1, D)  # (B*S, D)
        gates_flat = gates.view(-1, self.top_k)  # (B*S, top_k)
        indices_flat = expert_indices.view(-1, self.top_k)  # (B*S, top_k)
        
        # Initialize output
        output_flat = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_id in range(self.num_experts):
            # Find positions where this expert is selected
            mask = indices_flat.eq(expert_id)  # (B*S, top_k)
            
            if not mask.any():
                continue
                
            # Get positions and corresponding weights
            positions, slots = mask.nonzero(as_tuple=True)
            weights = gates_flat[positions, slots].unsqueeze(1)  # (num_selected, 1)
            inputs = x_flat[positions]  # (num_selected, D)
            
            # Process through expert
            expert_outputs = self.experts[expert_id](inputs)  # (num_selected, D)
            
            # Weight and accumulate
            weighted_outputs = expert_outputs * weights
            output_flat.index_add_(0, positions, weighted_outputs)
            
            # Update usage statistics
            self.expert_usage[expert_id] += mask.sum().item()
        
        # Reshape back to original dimensions
        return output_flat.view(B, S, D)

    def get_load_balancing_loss(self) -> torch.Tensor:
        """Calculate load balancing loss to encourage even expert usage"""
        if self.expert_usage.sum() == 0:
            return torch.tensor(0.0, device=self.expert_usage.device)
            
        usage_normalized = self.expert_usage / (self.expert_usage.sum() + 1e-8)
        mean_usage = usage_normalized.mean()
        std_usage = usage_normalized.std()
        
        # Coefficient of variation squared
        load_balance_loss = (std_usage / (mean_usage + 1e-8)) ** 2
        
        return load_balance_loss

    def reset_usage_stats(self):
        """Reset expert usage statistics"""
        self.expert_usage.zero_() 