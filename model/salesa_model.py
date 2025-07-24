import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import numpy as np

from config import SalesAConfig
from model.encoders import TextEncoder, VisionEncoder, AudioEncoder
from model.transformer import TransformerBlock

class SalesAModel(nn.Module):
    """Main SalesA AI model with multimodal capabilities"""
    def __init__(self, config: SalesAConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.model_author = config.model_author

        # Multimodal encoders
        self.text_encoder = TextEncoder(config)
        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output heads for different tasks
        self.text_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.vision_head = nn.Linear(config.hidden_dim, config.vocab_size)  # For vision-to-text
        self.audio_head = nn.Linear(config.hidden_dim, config.vocab_size)   # For audio-to-text
        self.code_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)  # For robotics actions

        # TTS integration (placeholder, can be replaced with real TTS model)
        self.tts = None  # Placeholder for TTS module

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None,
                task_type: str = "text",
                return_loss: bool = False,
                labels: Optional[torch.Tensor] = None,
                action_labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass through SalesA AI

        Args:
            input_ids: Text token IDs
            images: Image tensors
            audio: Audio waveforms
            task_type: Type of task ("text", "vision", "audio", "code", "action")
            return_loss: Whether to compute loss
            labels: Ground truth labels for loss computation
            action_labels: Action labels for robotics tasks
        """
        # Encode inputs based on modality
        embeddings = []

        if input_ids is not None:
            text_embeddings = self.text_encoder(input_ids)
            embeddings.append(text_embeddings)

        if images is not None:
            vision_embeddings = self.vision_encoder(images)
            embeddings.append(vision_embeddings)

        if audio is not None:
            audio_embeddings = self.audio_encoder(audio)
            embeddings.append(audio_embeddings)

        # Concatenate all embeddings
        if len(embeddings) == 1:
            x = embeddings[0]
        else:
            x = torch.cat(embeddings, dim=1)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Generate outputs based on task type
        if task_type == "text":
            logits = self.text_head(x)
        elif task_type == "vision":
            logits = self.vision_head(x)
        elif task_type == "audio":
            logits = self.audio_head(x)
        elif task_type == "code":
            logits = self.code_head(x)
        elif task_type == "action":
            logits = self.action_head(x)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Calculate loss if requested
        loss = None
        if return_loss:
            if task_type == "action" and action_labels is not None:
                # For discrete actions, use cross-entropy
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), action_labels.view(-1))
            elif labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": x
        }

    def tts_generate(self, text: str) -> bytes:
        """Text-to-speech: returns waveform bytes (placeholder, replace with real TTS)"""
        # Placeholder: return silence or use torchaudio TTS if available
        # You can integrate a real TTS model here (e.g., TTS from Coqui, torchaudio pipelines, etc.)
        sr = 22050
        duration = 2  # seconds
        waveform = np.zeros(int(sr * duration), dtype=np.float32)  # Silence
        return waveform.tobytes()

    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 do_sample: bool = True,
                 top_k: int = 50) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.forward(generated, task_type="text")
                logits = outputs["logits"][:, -1, :]  # Get last token predictions

                # Apply temperature
                logits = logits / temperature

                # Sample next token
                if do_sample:
                    # Top-k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        probs = F.softmax(top_k_logits, dim=-1)
                        next_token_idx = torch.multinomial(probs, 1)
                        next_token = top_k_indices.gather(1, next_token_idx)
                    else:
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if we generate end token (assuming 2 is end token)
                if next_token.item() == 2:
                    break

        return generated

    def get_name(self):
        return self.model_name

    def get_author(self):
        return self.model_author 