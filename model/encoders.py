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
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.hidden_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patches = self.extract_patches(images)
        patch_embeddings = self.patch_projection(patches)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        embeddings = embeddings + self.pos_embedding
        return embeddings

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
    """Audio encoder using 1D convolutions"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
        ])
        self.projection = nn.Linear(256, config.hidden_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        for conv in self.conv1d_layers:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        x = self.projection(x)
        return x 