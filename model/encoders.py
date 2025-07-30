import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SalesAConfig

class TextEncoder(nn.Module):
    """Text encoder with token embedding and positional encoding"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(config.max_seq_len, config.hidden_dim)
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text tokens"""
        seq_len = input_ids.shape[1]
        embeddings = self.embedding(input_ids)
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0)
        embeddings = embeddings + pos_encoding
        return self.dropout(embeddings)

class VisionEncoder(nn.Module):
    """Vision encoder using patch-based approach"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.vision_patch_size
        self.num_patches = (config.vision_dim // config.vision_patch_size) ** 2
        self.patch_projection = nn.Linear(
            3 * config.vision_patch_size * config.vision_patch_size,
            config.hidden_dim
        )
        # Fixed: Use only num_patches for positional embeddings (no CLS token for multimodal)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, config.hidden_dim)
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patches = self.extract_patches(images)
        patch_embeddings = self.patch_projection(patches)
        # Add positional embeddings directly to patch embeddings
        embeddings = patch_embeddings + self.pos_embedding
        return self.dropout(embeddings)

    def extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = images.shape
        patch_size = self.patch_size
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(
            batch_size, channels, -1, patch_size, patch_size
        )
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(batch_size, -1, channels * patch_size * patch_size)
        return patches

class AudioEncoder(nn.Module):
    """Audio encoder using 1D convolutions with improved processing"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        
        # Improved CNN architecture for audio processing
        self.conv1d_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ),
        ])
        
        # Projection layers for better feature alignment
        self.pre_projection = nn.Linear(256, 512)
        self.projection = nn.Linear(512, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Handle different input shapes
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        x = audio
        
        # Apply CNN layers
        for conv_layer in self.conv1d_layers:
            x = conv_layer(x)
        
        # Global pooling to get fixed-size representation
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch_size, 256)
        
        # Feature transformation
        x = F.relu(self.pre_projection(x))
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Add sequence dimension for consistency with other encoders
        # Match the sequence length of text encoder (which can vary)
        # For now, we'll create a fixed sequence of length 1
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        return x
