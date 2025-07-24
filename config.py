from dataclasses import dataclass
import math

@dataclass
class SalesAConfig:
    """Configuration class for SalesA AI"""
    # Model architecture
    vocab_size: int = 32000
    hidden_dim: int = 512  # Reduced for CPU efficiency
    num_layers: int = 8    # Reduced for CPU efficiency
    num_heads: int = 8
    intermediate_dim: int = 1024
    max_seq_len: int = 2048

    # MoE configuration
    num_experts: int = 4   # Reduced for CPU efficiency
    expert_capacity: int = 2
    top_k: int = 2

    # Multimodal dimensions
    vision_dim: int = 224
    audio_dim: int = 80
    vision_patch_size: int = 16
    audio_patch_size: int = 4

    # Training parameters
    batch_size: int = 4    # Small batch for CPU
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1

    # Optimization for CPU
    use_mixed_precision: bool = False  # CPU doesn't support AMP well
    gradient_checkpointing: bool = True

    # Data parameters
    max_text_length: int = 1024
    max_audio_length: int = 16000  # 1 second at 16kHz
    action_dim: int = 10  # Number of possible actions for robotics

    model_name: str = "SalesA AI"
    model_author: str = "Created by N.E.N (Nthuku Elijah Nzeli) and SalesA Team"

    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts" 