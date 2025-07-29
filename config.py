"""
Core configuration for SalesA AI - Simple config format expected by model components
This provides the simplified config class that the model components expect.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SalesAConfig:
    """
    Simple configuration class for SalesA AI model components
    This is the config format expected by the model, encoders, transformers, etc.
    """
    # Model metadata
    model_name: str = "SalesA AI"
    model_author: str = "Created by N.E.N (Nthuku Elijah Nzeli) and SalesA Team"
    
    # Core architecture parameters
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_dim: int = 1024
    max_seq_len: int = 2048
    
    # MoE configuration
    num_experts: int = 4
    expert_capacity: int = 2
    top_k: int = 2
    
    # Multimodal dimensions
    vision_dim: int = 224
    audio_dim: int = 80
    vision_patch_size: int = 16
    audio_patch_size: int = 4
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    num_epochs: int = 10
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 0
    
    # Optimization flags
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = True
    
    # Data parameters
    max_text_length: int = 1024
    max_audio_length: int = 16000
    action_dim: int = 10
    
    # Utility methods for dynamic access
    def get(self, key: str, default=None):
        """Get configuration value with default fallback"""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return getattr(self, key)
    
    def __contains__(self, key: str):
        """Allow 'in' operator usage"""
        return hasattr(self, key)
