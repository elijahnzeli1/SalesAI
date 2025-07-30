import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List
import numpy as np

# Make torchaudio import optional
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None
    print("⚠️  TorchAudio not available - audio features will be limited")

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

        # Modality projection layers to align dimensions
        self.text_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.vision_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.audio_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

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

        # Modality embeddings
        self.text_modality_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.vision_modality_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.audio_modality_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

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
                text: Optional[torch.Tensor] = None,
                image: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                task_type: str = "text",
                return_loss: bool = False,
                labels: Optional[torch.Tensor] = None,
                action_labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
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
        # Handle parameter mapping (text -> input_ids, image -> images)
        if text is not None and input_ids is None:
            input_ids = text
        if image is not None and images is None:
            images = image
            
        # Encode inputs based on modality
        embeddings = []
        modality_info = []

        if input_ids is not None:
            text_embeddings = self.text_encoder(input_ids)
            text_embeddings = self.text_projection(text_embeddings)
            text_embeddings = text_embeddings + self.text_modality_embedding
            embeddings.append(text_embeddings)
            modality_info.extend([0] * text_embeddings.shape[1])

        if images is not None:
            vision_embeddings = self.vision_encoder(images)
            vision_embeddings = self.vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings + self.vision_modality_embedding
            embeddings.append(vision_embeddings)
            modality_info.extend([1] * vision_embeddings.shape[1])

        if audio is not None:
            audio_embeddings = self.audio_encoder(audio)
            audio_embeddings = self.audio_projection(audio_embeddings)
            audio_embeddings = audio_embeddings + self.audio_modality_embedding
            embeddings.append(audio_embeddings)
            modality_info.extend([2] * audio_embeddings.shape[1])

        # Concatenate all embeddings (handle different sequence lengths)
        if len(embeddings) == 1:
            x = embeddings[0]
        else:
            # Check if all embeddings have the same batch size
            batch_size = embeddings[0].shape[0]
            for emb in embeddings[1:]:
                if emb.shape[0] != batch_size:
                    # Expand to match batch size if needed
                    emb = emb.expand(batch_size, -1, -1)
            x = torch.cat(embeddings, dim=1)

        # Create attention mask and modality info AFTER concatenation
        # This ensures the mask matches the actual sequence length
        batch_size, seq_len, _ = x.shape
        
        attention_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=x.device))
        
        # Create modality tensor with proper dimensions
        modality_tensor = torch.tensor(modality_info, dtype=torch.long, device=x.device)
        if batch_size > 1:
            modality_tensor = modality_tensor.unsqueeze(0).expand(batch_size, -1)
        else:
            modality_tensor = modality_tensor.unsqueeze(0)

        # Pass through transformer blocks
        total_load_balance_loss = 0.0
        for block in self.transformer_blocks:
            x = block(x, attention_mask, modality_tensor)
            # Accumulate load balancing loss from MoE layers
            if hasattr(block.moe, 'get_load_balancing_loss'):
                total_load_balance_loss += block.moe.get_load_balancing_loss()

        # Generate outputs based on task type
        # Use the first task type in the list for simplicity, since we are processing a batch
        if isinstance(task_type, list):
            task_type = task_type[0]
        
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
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                
                # Ignore padding tokens in loss calculation
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            # Add load balancing loss
            if total_load_balance_loss > 0:
                loss = loss + 0.01 * total_load_balance_loss  # Weight factor for load balancing

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": x,
            "load_balance_loss": total_load_balance_loss
        }

    def tts_generate(self, text: str) -> bytes:
        """Text-to-speech: returns waveform bytes"""
        try:
            # Try to use torchaudio's TTS pipeline if available
            if TORCHAUDIO_AVAILABLE and hasattr(torchaudio.pipelines, 'TTS'):
                # Use torchaudio's built-in TTS
                tts_pipeline = torchaudio.pipelines.TTS()
                waveform, sample_rate = tts_pipeline(text)
                return waveform.numpy().tobytes()
            else:
                # Fallback: generate a simple sine wave with varying frequency based on text
                return self._generate_sine_wave_tts(text)
        except Exception as e:
            print(f"TTS generation failed: {e}")
            # Ultimate fallback: return silence
            return self._generate_silence()
    
    def _generate_sine_wave_tts(self, text: str) -> bytes:
        """Generate a simple sine wave TTS as fallback"""
        sr = 22050
        duration = len(text) * 0.1  # 0.1 seconds per character
        duration = max(1.0, min(duration, 5.0))  # Between 1-5 seconds
        
        # Generate frequency based on text content
        base_freq = 220  # A3 note
        freq_variation = hash(text) % 200  # Simple hash-based variation
        frequency = base_freq + freq_variation
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sr * duration), False)
        waveform = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% amplitude
        
        # Add some variation to make it less monotonous
        for i, char in enumerate(text[:10]):  # Use first 10 characters
            if i < len(t):
                char_freq = base_freq + (ord(char) % 100)
                start_idx = int(i * sr * duration / len(text))
                end_idx = int((i + 1) * sr * duration / len(text))
                if end_idx < len(t):
                    t_segment = t[start_idx:end_idx]
                    waveform[start_idx:end_idx] += np.sin(2 * np.pi * char_freq * t_segment) * 0.1
        
        # Normalize and convert to float32
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
        return waveform.tobytes()
    
    def _generate_silence(self) -> bytes:
        """Generate silence as ultimate fallback"""
        sr = 22050
        duration = 2  # seconds
        waveform = np.zeros(int(sr * duration), dtype=np.float32)
        return waveform.tobytes()

    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """Generate text using greedy decoding"""
        self.eval()
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self(input_ids=current_ids, task_type="text")
                logits = outputs["logits"]
                
                # Get next token (greedy decoding)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Check for end of sequence token (assuming token ID 2 is EOS)
                if next_token.item() == 2:  # EOS token
                    break
                
                # Append next token
                current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=-1)
                
                # Check if we've reached max length
                if current_ids.shape[1] >= max_length:
                    break
        
        self.train()
        return current_ids

    def get_name(self):
        return self.model_name

    def get_author(self):
        return self.model_author 