"""
Audio Processing Utilities for SalesAI
Alternative audio processing without torchcodec dependency
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, Union
import os

logger = logging.getLogger(__name__)

# Import audio libraries with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available - some audio features will be limited")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("SoundFile not available - audio file reading will be limited")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("TorchAudio not available - some audio features will be limited")

class AudioProcessor:
    """Audio processing utilities without torchcodec dependency"""
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        
    def load_audio(self, file_path: str) -> Optional[torch.Tensor]:
        """Load audio file using available libraries"""
        try:
            if TORCHAUDIO_AVAILABLE:
                # Use torchaudio for loading
                waveform, sr = torchaudio.load(file_path)
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                return waveform.squeeze()
            
            elif SOUNDFILE_AVAILABLE:
                # Use soundfile for loading
                waveform, sr = sf.read(file_path)
                waveform = torch.tensor(waveform, dtype=torch.float32)
                if sr != self.sample_rate:
                    # Simple resampling (for better quality, use librosa)
                    if LIBROSA_AVAILABLE:
                        waveform = librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=self.sample_rate)
                        waveform = torch.tensor(waveform, dtype=torch.float32)
                return waveform
            
            else:
                logger.error("No audio loading library available")
                return None
                
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features"""
        try:
            if TORCHAUDIO_AVAILABLE:
                # Use torchaudio for mel spectrogram
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                mel_spec = mel_transform(audio)
                # Convert to log scale
                mel_spec = torch.log(mel_spec + 1e-9)
                return mel_spec
            
            elif LIBROSA_AVAILABLE:
                # Use librosa for mel spectrogram
                audio_np = audio.numpy()
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_np,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
                mel_spec = torch.log(mel_spec + 1e-9)
                return mel_spec
            
            else:
                # Fallback: simple FFT-based features
                return self._extract_simple_features(audio)
                
        except Exception as e:
            logger.error(f"Error extracting mel spectrogram: {e}")
            return self._extract_simple_features(audio)
    
    def _extract_simple_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract simple FFT-based features as fallback"""
        # Simple FFT-based feature extraction
        fft = torch.fft.fft(audio)
        magnitude = torch.abs(fft)
        
        # Take first n_mels frequency bins
        features = magnitude[:self.n_mels]
        
        # Normalize
        features = features / (torch.norm(features) + 1e-9)
        
        return features.unsqueeze(0)  # Add time dimension
    
    def extract_mfcc(self, audio: torch.Tensor, n_mfcc: int = 13) -> torch.Tensor:
        """Extract MFCC features"""
        try:
            if TORCHAUDIO_AVAILABLE:
                # Use torchaudio for MFCC
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=self.sample_rate,
                    n_mfcc=n_mfcc,
                    melkwargs={'n_fft': self.n_fft, 'hop_length': self.hop_length}
                )
                mfcc = mfcc_transform(audio)
                return mfcc
            
            elif LIBROSA_AVAILABLE:
                # Use librosa for MFCC
                audio_np = audio.numpy()
                mfcc = librosa.feature.mfcc(
                    y=audio_np,
                    sr=self.sample_rate,
                    n_mfcc=n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mfcc = torch.tensor(mfcc, dtype=torch.float32)
                return mfcc
            
            else:
                # Fallback: return zeros
                logger.warning("MFCC extraction not available")
                return torch.zeros(n_mfcc, 1)
                
        except Exception as e:
            logger.error(f"Error extracting MFCC: {e}")
            return torch.zeros(n_mfcc, 1)
    
    def extract_spectral_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract spectral features (centroid, bandwidth, etc.)"""
        try:
            if LIBROSA_AVAILABLE:
                audio_np = audio.numpy()
                
                # Spectral centroid
                centroid = librosa.feature.spectral_centroid(
                    y=audio_np, sr=self.sample_rate, hop_length=self.hop_length
                )
                
                # Spectral bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio_np, sr=self.sample_rate, hop_length=self.hop_length
                )
                
                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(
                    y=audio_np, sr=self.sample_rate, hop_length=self.hop_length
                )
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio_np, hop_length=self.hop_length)
                
                # Combine features
                features = torch.tensor(np.vstack([centroid, bandwidth, rolloff, zcr]), dtype=torch.float32)
                return features
            
            else:
                # Fallback: simple features
                return self._extract_simple_spectral_features(audio)
                
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return self._extract_simple_spectral_features(audio)
    
    def _extract_simple_spectral_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract simple spectral features as fallback"""
        # Simple zero crossing rate
        zcr = torch.sum(torch.sign(audio[1:]) != torch.sign(audio[:-1])) / len(audio)
        
        # Simple energy
        energy = torch.mean(audio ** 2)
        
        # Simple spectral centroid approximation
        fft = torch.fft.fft(audio)
        freqs = torch.fft.fftfreq(len(audio), 1/self.sample_rate)
        magnitude = torch.abs(fft)
        centroid = torch.sum(freqs * magnitude) / (torch.sum(magnitude) + 1e-9)
        
        features = torch.tensor([zcr, energy, centroid], dtype=torch.float32)
        return features.unsqueeze(1)  # Add time dimension
    
    def process_audio(self, audio: Union[torch.Tensor, str]) -> torch.Tensor:
        """Main audio processing function"""
        # Load audio if file path is provided
        if isinstance(audio, str):
            audio = self.load_audio(audio)
            if audio is None:
                return torch.zeros(1, self.n_mels)
        
        # Ensure audio is the right length
        target_length = self.sample_rate * 4  # 4 seconds
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = F.pad(audio, (0, padding))
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio)
        mfcc = self.extract_mfcc(audio)
        spectral = self.extract_spectral_features(audio)
        
        # Ensure all features have compatible dimensions for concatenation
        # Mel spectrogram: (n_mels, time_steps)
        # MFCC: (n_mfcc, time_steps) 
        # Spectral: (n_features, time_steps)
        
        # Get the minimum time dimension
        min_time = min(mel_spec.shape[1], mfcc.shape[1], spectral.shape[1])
        
        # Truncate all features to the same time dimension
        mel_spec = mel_spec[:, :min_time]
        mfcc = mfcc[:, :min_time]
        spectral = spectral[:, :min_time]
        
        # Combine features along feature dimension
        features = torch.cat([mel_spec, mfcc, spectral], dim=0)
        
        return features

class AudioDatasetProcessor:
    """Process audio datasets without torchcodec"""
    
    def __init__(self, sample_rate: int = 16000):
        self.processor = AudioProcessor(sample_rate=sample_rate)
        self.sample_rate = sample_rate
    
    def process_audio_sample(self, audio_data) -> Optional[torch.Tensor]:
        """Process a single audio sample from dataset"""
        try:
            if isinstance(audio_data, dict):
                # Handle different audio formats
                if 'array' in audio_data:
                    audio = torch.tensor(audio_data['array'], dtype=torch.float32)
                elif 'path' in audio_data:
                    audio = self.processor.load_audio(audio_data['path'])
                    if audio is None:
                        return None
                else:
                    logger.warning(f"Unknown audio format: {audio_data.keys()}")
                    return None
            elif isinstance(audio_data, (list, np.ndarray)):
                audio = torch.tensor(audio_data, dtype=torch.float32)
            elif isinstance(audio_data, torch.Tensor):
                audio = audio_data
            else:
                logger.warning(f"Unsupported audio data type: {type(audio_data)}")
                return None
            
            # Process audio
            features = self.processor.process_audio(audio)
            return features
            
        except Exception as e:
            logger.error(f"Error processing audio sample: {e}")
            return None
    
    def create_audio_features(self, audio_samples: list) -> torch.Tensor:
        """Create audio features from a list of samples"""
        features_list = []
        
        for audio_sample in audio_samples:
            features = self.process_audio_sample(audio_sample)
            if features is not None:
                features_list.append(features)
        
        if features_list:
            return torch.stack(features_list)
        else:
            # Return dummy features if no valid audio
            return torch.zeros(len(audio_samples), 1, 128) 