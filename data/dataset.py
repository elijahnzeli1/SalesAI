import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import logging
import random
from typing import Dict, List, Optional, Union, Any
import os
from collections import deque
import threading
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset, IterableDataset

# Import torchaudio with error handling
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TorchAudio not available: {e}")
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None
except Exception as e:
    print(f"⚠️  TorchAudio import error: {e}")
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None

# Import PIL with error handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  PIL not available: {e}")
    PIL_AVAILABLE = False
    Image = None

# Import soundfile with error handling
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SoundFile not available: {e}")
    SOUNDFILE_AVAILABLE = False
    sf = None

# Import our custom audio processor
try:
    from utils.audio_processor import AudioDatasetProcessor
    AUDIO_PROCESSOR_AVAILABLE = True
    print("✅ Custom audio processor available")
except ImportError as e:
    print(f"⚠️  Custom audio processor not available: {e}")
    AUDIO_PROCESSOR_AVAILABLE = False

# Conditionally disable TorchCodec only if not properly installed
try:
    import torchcodec
    # Test if torchcodec.decoders is available
    try:
        from torchcodec.decoders import AudioDecoder # type: ignore
        os.environ['TORCHCODEC_DISABLE'] = '0'  # Enable if available
        TORCHCODEC_AVAILABLE = True
    except ImportError:
        # torchcodec is installed but decoders module is missing
        os.environ['TORCHCODEC_DISABLE'] = '1'  # Disable if decoders not available
        TORCHCODEC_AVAILABLE = False
        print("⚠️  TorchCodec decoders not available - using custom audio processor")
except ImportError:
    os.environ['TORCHCODEC_DISABLE'] = '1'  # Disable if not available
    TORCHCODEC_AVAILABLE = False
    print("⚠️  TorchCodec not available - using custom audio processor")
except Exception as e:
    os.environ['TORCHCODEC_DISABLE'] = '1'  # Disable on any error
    TORCHCODEC_AVAILABLE = False
    print(f"⚠️  TorchCodec error: {e} - using custom audio processor")

from config import SalesAConfig
from tokenizer import SalesATokenizer

logger = logging.getLogger(__name__)

class StreamingDatasetWrapper:
    """Wrapper for streaming datasets with caching and prefetching"""
    
    def __init__(self, dataset, max_cache_size=1000):
        self.dataset = dataset
        self.cache = deque(maxlen=max_cache_size)
        self.cache_lock = threading.Lock()
        self.prefetch_thread = None
        self.stop_prefetch = False
        self._iterator = None
        
    def start_prefetching(self):
        """Start background prefetching of samples with timeout protection"""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.stop_prefetch = False
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
            self.prefetch_thread.daemon = True
            self.prefetch_thread.start()
            
            # Add a timeout mechanism to prevent hanging
            def timeout_monitor():
                time.sleep(30)  # Wait 30 seconds
                if self.prefetch_thread.is_alive():
                    logger.warning("Prefetch worker taking too long, stopping...")
                    self.stop_prefetch = True
            
            timeout_thread = threading.Thread(target=timeout_monitor, daemon=True)
            timeout_thread.start()
    
    def stop_prefetching(self):
        """Stop background prefetching"""
        self.stop_prefetch = True
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
    
    def _prefetch_worker(self):
        """Background worker for prefetching samples"""
        try:
            # Handle IterableDatasets properly
            if isinstance(self.dataset, IterableDataset):
                for item in self.dataset:
                    if self.stop_prefetch:
                        break
                    try:
                        # Try to process the item to catch any torchcodec errors early
                        if hasattr(item, 'get') and item.get('audio') is not None:
                            # Skip audio items that might cause torchcodec issues
                            continue
                        with self.cache_lock:
                            if len(self.cache) < self.cache.maxlen:
                                self.cache.append(item)
                    except Exception as item_error:
                        # Log but continue with other items
                        if "register_fake" in str(item_error) or "torchcodec" in str(item_error).lower():
                            logger.debug(f"Skipping audio item due to torchcodec issue: {item_error}")
                            continue
                        else:
                            logger.warning(f"Error processing item in prefetch worker: {item_error}")
                            continue
                    time.sleep(0.001)  # Small delay to prevent overwhelming
            else:
                # Regular dataset
                for item in self.dataset:
                    if self.stop_prefetch:
                        break
                    try:
                        # Try to process the item to catch any torchcodec errors early
                        if hasattr(item, 'get') and item.get('audio') is not None:
                            # Skip audio items that might cause torchcodec issues
                            continue
                        with self.cache_lock:
                            if len(self.cache) < self.cache.maxlen:
                                self.cache.append(item)
                    except Exception as item_error:
                        # Log but continue with other items
                        if "register_fake" in str(item_error) or "torchcodec" in str(item_error).lower():
                            logger.debug(f"Skipping audio item due to torchcodec issue: {item_error}")
                            continue
                        else:
                            logger.warning(f"Error processing item in prefetch worker: {item_error}")
                            continue
                    time.sleep(0.001)  # Small delay to prevent overwhelming
        except Exception as e:
            logger.warning(f"Prefetch worker error: {e}")
            # Don't let the error crash the entire process
            if "register_fake" in str(e) or "torchcodec" in str(e).lower():
                logger.warning("Prefetch worker stopped due to torchcodec compatibility issues")
            else:
                logger.warning("Prefetch worker stopped due to unexpected error")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        with self.cache_lock:
            if self.cache:
                return self.cache.popleft()
        
        # If cache is empty, get directly from dataset
        # Handle both regular datasets and IterableDatasets
        if isinstance(self.dataset, IterableDataset):
            if self._iterator is None:
                self._iterator = iter(self.dataset)
            return next(self._iterator)
        else:
            return next(self.dataset)

class MultimodalDataset(Dataset):
    """Dataset for multimodal training with hybrid streaming approach"""

    def __init__(self, config: SalesAConfig, tokenizer: SalesATokenizer, split: str = "train", dataset_name: Union[str, List[str]] = "auto", task_type: Optional[str] = None, use_streaming: bool = True, cache_size: int = 1000):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.use_streaming = use_streaming
        self.cache_size = cache_size
        self.streaming_datasets = []
        self.processed_samples = []  # Add the missing attribute
        
        # Initialize audio processor if available and torchcodec is working
        if AUDIO_PROCESSOR_AVAILABLE and TORCHCODEC_AVAILABLE:
            self.audio_processor = AudioDatasetProcessor(sample_rate=16000)
            self.audio_available = True
        else:
            self.audio_processor = None
            self.audio_available = False
            if not TORCHCODEC_AVAILABLE:
                logger.warning("TorchCodec not available - disabling audio datasets to prevent compatibility issues")
        
        # Set maximum samples per dataset
        self.max_samples_per_dataset = getattr(config, 'max_samples_per_dataset', 1000)
        
        # Initialize datasets
        self._initialize_streaming_datasets()
        
        # Start prefetching for all streaming datasets with delay for audio datasets
        for wrapper in self.streaming_datasets:
            # Delay prefetching for audio datasets to avoid torchcodec issues
            if wrapper.dataset_config.get('has_audio', False):
                def delayed_start(wrapper=wrapper):
                    time.sleep(5)  # Wait 5 seconds before starting audio dataset prefetching
                    if not wrapper.stop_prefetch:
                        wrapper.start_prefetching()
                
                delay_thread = threading.Thread(target=delayed_start, daemon=True)
                delay_thread.start()
            else:
                wrapper.start_prefetching()

    def _initialize_streaming_datasets(self):
        """Initialize streaming datasets without downloading"""
        # Define available datasets with their configurations
        dataset_configs = {
            # Tri‑modal: audio + text (no image)
            "AnimeVox": {
                "name": "taresh18/AnimeVox",
                "config": "default",  # Use the only available config
                "has_image": False,
                "has_text": True,
                "has_audio": True,
                "audio_key": "audio",
                "text_key": "english_transcription",
                "limit": 2000 if self.split == "train" else 500,
                "max_samples": 1500,
                "note": "AnimeVox audio dataset with transcriptions"
            },
            "open_platypus": {
                "name": "garage-bAInd/Open-Platypus",
                "config": None,
                "has_image": False,
                "has_text": True,
                "has_audio": False,
                "text_key": "input",
                "labels_key": "output",
                "limit": 10000 if self.split == "train" else 2000,
                "max_samples": 10000
            },
            "humaneval": {
                "name": "code-rag-bench/humaneval",
                "config": None,
                "has_image": False,
                "has_text": True,
                "has_audio": False,
                "text_key": "prompt",
                "labels_key": "canonical_solution",
                "limit": 1000 if self.split == "train" else 200,
                "max_samples": 2000
            },
            # vision + text
            "beans": {
                "name": "beans",
                "config": None,
                "has_image": True,
                "has_text": True,  # Beans has text labels
                "has_audio": False,
                "image_key": "image",
                "text_key": "label",  # This is the text label for classification
                "labels_key": "label",  # Add labels_key for consistency
                "limit": 1000 if self.split == "train" else 200,
                "max_samples": 1000,
                "note": "Image classification dataset for bean disease detection"
            },
            #text only
            "prosocial_dialog": {
                "name": "allenai/prosocial-dialog",
                "config": None,
                "has_image": False,
                "has_text": True,
                "has_audio": False,
                "text_key": "context",
                "labels_key": "response",
                "limit": 2000 if self.split == "train" else 500,
                "max_samples": 2000,
                "note": "Prosocial dialog dataset for safe AI conversations"
            },
        }

        # --- Enhanced automatic selection logic ---
        if self.dataset_name == "auto":
            # Select default dataset(s) based on task_type
            if hasattr(self, 'task_type') and self.task_type:
                if self.task_type in ["code", "code-generation"]:
                    selected_datasets = ["humaneval"]
                elif self.task_type == "vision":
                    selected_datasets = ["beans"]
                elif self.task_type == "audio":
                    # Include audio datasets if custom processor is available
                    if self.audio_available:
                        selected_datasets = ["AnimeVox"]
                    else:
                        logger.warning("Audio task requested but audio processor not available. Using text dataset instead.")
                        selected_datasets = ["open_platypus"]
                elif self.task_type in ["financial", "stock"]:
                    selected_datasets = ["financial_phrasebank"]
                elif self.task_type == "text":
                    selected_datasets = ["open_platypus"]
                else:
                    selected_datasets = ["open_platypus"]
            else:
                # Fallback to general text dataset
                selected_datasets = ["open_platypus"]
        elif self.dataset_name == "all":
            # Include all datasets, including audio if processor is available
            if self.audio_available:
                selected_datasets = list(dataset_configs.keys())
            else:
                selected_datasets = [key for key in dataset_configs.keys() if key != "AnimeVox"]
                logger.warning("Audio processor not available - excluding audio datasets from 'all' selection")
        elif isinstance(self.dataset_name, list):
            # Include audio datasets if processor is available
            if self.audio_available:
                selected_datasets = self.dataset_name
            else:
                selected_datasets = [name for name in self.dataset_name if name != "AnimeVox"]
                if "AnimeVox" in self.dataset_name:
                    logger.warning("Audio processor not available - excluding AnimeVox from dataset list")
        else:
            # Single dataset selection
            if self.dataset_name == "AnimeVox" and not self.audio_available:
                logger.warning("AnimeVox requested but audio processor not available. Using open_platypus instead.")
                selected_datasets = ["open_platypus"]
            else:
                selected_datasets = [self.dataset_name]

        # Initialize streaming datasets
        for dataset_key in selected_datasets:
            if dataset_key not in dataset_configs:
                logger.warning(f"Unknown dataset: {dataset_key}")
                continue

            dataset_config = dataset_configs[dataset_key]
            try:
                streaming_dataset = self._create_streaming_dataset(dataset_config)
                if streaming_dataset:
                    self.streaming_datasets.append(streaming_dataset)
                    logger.info(f"Initialized streaming dataset: {dataset_config['name']}")
            except Exception as e:
                logger.error(f"Failed to initialize streaming dataset {dataset_key}: {e}")

        # If no streaming datasets were created, fall back to synthetic data
        if not self.streaming_datasets:
            logger.warning("No streaming datasets initialized. Will use synthetic data.")
            self._generate_synthetic_data()

    def _create_streaming_dataset(self, dataset_config: Dict) -> Optional[StreamingDatasetWrapper]:
        """Create a streaming dataset wrapper"""
        try:
            # Skip audio datasets only if no audio processor is available
            if dataset_config['name'] == "taresh18/AnimeVox" and not self.audio_available:
                logger.warning(f"Skipping {dataset_config['name']} - Audio processor not available for audio processing")
                return None
                
            # Handle special split cases
            if dataset_config['name'] == "taresh18/AnimeVox" and self.split == "validation":
                logger.warning("AnimeVox doesn't have validation split. Using test split instead.")
                split_to_use = "test"
            elif dataset_config['name'] == "garage-bAInd/Open-Platypus" and self.split == "validation":
                 split_to_use = "train"
            elif dataset_config['name'] == "beans" and self.split == "validation":
                logger.warning("Beans dataset doesn't have validation split. Using test split instead.")
                split_to_use = "test"
            else:
                split_to_use = self.split

            # Create streaming dataset
            if self.use_streaming:
                if dataset_config['config']:
                    dataset = load_dataset(
                        dataset_config['name'],
                        dataset_config['config'],
                        split=split_to_use,
                        streaming=True
                    )
                else:
                    dataset = load_dataset(
                        dataset_config['name'],
                        split=split_to_use,
                        streaming=True
                    )
            else:
                # Fallback to non-streaming for datasets that don't support streaming
                if dataset_config['config']:
                    dataset = load_dataset(
                        dataset_config['name'],
                        dataset_config['config'],
                        split=split_to_use,
                        streaming=False
                    )
                else:
                    dataset = load_dataset(
                        dataset_config['name'],
                        split=split_to_use,
                        streaming=False
                    )

            # Wrap with streaming wrapper
            wrapper = StreamingDatasetWrapper(dataset, max_cache_size=self.cache_size)
            wrapper.dataset_config = dataset_config
            wrapper.samples_processed = 0
            
            return wrapper

        except Exception as e:
            logger.error(f"Error creating streaming dataset {dataset_config['name']}: {e}")
            # If it's a torchcodec-related error, log it specifically
            if "register_fake" in str(e) or "torchcodec" in str(e).lower():
                logger.warning(f"Skipping {dataset_config['name']} due to TorchCodec compatibility issues")
            return None

    def _load_data(self) -> List[Dict]:
        """This method is now deprecated - data is loaded on-demand via streaming"""
        logger.warning("_load_data is deprecated. Use streaming approach instead.")
        return []

    def _ensure_local_dataset(self, repo_id, config=None):
        """This method is now deprecated - we use streaming instead of local downloads"""
        logger.warning("_ensure_local_dataset is deprecated. Using streaming approach instead.")
        return None

    def _load_single_dataset(self, dataset_config: Dict) -> List[Dict]:
        """This method is now deprecated - datasets are handled via streaming"""
        logger.warning("_load_single_dataset is deprecated. Use streaming approach instead.")
        return []

    def _get_next_sample(self) -> Optional[Dict]:
        """Get the next sample from any available streaming dataset"""
        for wrapper in self.streaming_datasets:
            if wrapper.samples_processed >= self.max_samples_per_dataset:
                continue
                
            try:
                item = next(wrapper)
                if item is None:
                    continue
                
                # Skip audio items that might cause torchcodec issues
                if hasattr(item, 'get') and item.get('audio') is not None:
                    logger.debug(f"Skipping audio item from {wrapper.dataset_config['name']} due to torchcodec compatibility")
                    continue
                    
                sample = self._process_sample(item, wrapper.dataset_config)
                if sample:
                    wrapper.samples_processed += 1
                    return sample
            except StopIteration:
                # This dataset is exhausted
                continue
            except Exception as e:
                # Handle torchcodec errors gracefully
                if "register_fake" in str(e) or "torchcodec" in str(e).lower():
                    logger.warning(f"Skipping {wrapper.dataset_config['name']} due to torchcodec compatibility issues: {e}")
                    # Mark this dataset as exhausted to avoid repeated errors
                    wrapper.samples_processed = self.max_samples_per_dataset
                    continue
                else:
                    logger.warning(f"Error getting sample from {wrapper.dataset_config['name']}: {e}")
                    continue
                continue
        
        return None

    def _process_sample(self, item: Dict, dataset_config: Dict) -> Optional[Dict]:
        """Process a single sample from the dataset"""
        try:
            processed_item = {}
            
            # Handle text data with proper encoding
            if dataset_config.get("has_text", False):
                text_key = dataset_config.get("text_key", "text")
                if text_key in item:
                    try:
                        # Handle different text formats and encoding issues
                        text_data = item[text_key]
                        if isinstance(text_data, bytes):
                            text_data = text_data.decode('utf-8', errors='ignore')
                        elif isinstance(text_data, list):
                            text_data = " ".join([str(t) for t in text_data])
                        elif not isinstance(text_data, str):
                            text_data = str(text_data)
                        
                        # Clean and validate text
                        if text_data and len(text_data.strip()) > 0:
                            processed_item["text"] = text_data.strip()
                    except (UnicodeDecodeError, AttributeError) as e:
                        logger.warning(f"Text processing error: {e}")
                        return None
            
            # Handle audio data with improved processing
            if dataset_config.get("has_audio", False):
                audio_key = dataset_config.get("audio_key", "audio")
                if audio_key in item:
                    try:
                        audio_data = item[audio_key]
                        
                        # Use our custom audio processor if available
                        if self.audio_processor is not None:
                            processed_audio = self.audio_processor.process_audio_sample(audio_data)
                            if processed_audio is not None:
                                processed_item["audio"] = processed_audio
                            else:
                                logger.warning("Audio processing failed - skipping audio")
                        else:
                            # Fallback to basic audio handling
                            if hasattr(audio_data, 'array'):
                                # AudioArray format
                                processed_item["audio"] = audio_data.array
                            elif hasattr(audio_data, 'numpy'):
                                # Convert to numpy array
                                processed_item["audio"] = audio_data.numpy()
                            elif isinstance(audio_data, (list, tuple)):
                                # List/tuple format
                                processed_item["audio"] = np.array(audio_data)
                            elif isinstance(audio_data, np.ndarray):
                                # Already numpy array
                                processed_item["audio"] = audio_data
                            elif isinstance(audio_data, torch.Tensor):
                                # PyTorch tensor
                                processed_item["audio"] = audio_data
                            else:
                                logger.warning(f"Audio format not supported: {type(audio_data)}")
                            
                    except Exception as e:
                        logger.warning(f"Audio processing error (skipping audio): {e}")
                        # Continue without audio - don't fail the entire sample
                        pass
            
            # Handle image data
            if dataset_config.get("has_image", False):
                image_key = dataset_config.get("image_key", "image")
                if image_key in item:
                    try:
                        image_data = item[image_key]
                        if PIL_AVAILABLE and hasattr(image_data, 'convert'):
                            processed_item["image"] = image_data
                        elif isinstance(image_data, (torch.Tensor, np.ndarray)):
                            # Handle tensor/array format
                            processed_item["image"] = image_data
                        else:
                            # Try to import torchvision locally if needed
                            try:
                                import torchvision.transforms as transforms
                                # If we get here, torchvision is available
                                if hasattr(image_data, 'convert'):
                                    processed_item["image"] = image_data
                                else:
                                    logger.warning(f"Image format not supported: {type(image_data)}")
                            except ImportError:
                                logger.warning(f"TorchVision not available, skipping image processing")
                            except Exception as e:
                                logger.warning(f"Image processing error: {e}")
                    except Exception as e:
                        logger.warning(f"Image processing error: {e}")
                        # Continue without image
            
            # Add labels if available
            labels_key = dataset_config.get("labels_key")
            if labels_key and labels_key in item:
                try:
                    label_data = item[labels_key]
                    if isinstance(label_data, bytes):
                        label_data = label_data.decode('utf-8', errors='ignore')
                    
                    # Try to preserve numeric labels as numbers
                    if isinstance(label_data, (int, float)):
                        processed_item["labels"] = label_data
                    elif isinstance(label_data, str):
                        # Try to convert string to number if possible
                        try:
                            if '.' in label_data:
                                processed_item["labels"] = float(label_data)
                            else:
                                processed_item["labels"] = int(label_data)
                        except (ValueError, TypeError):
                            # Keep as string if conversion fails
                            processed_item["labels"] = label_data
                    else:
                        processed_item["labels"] = label_data
                except (UnicodeDecodeError, AttributeError) as e:
                    logger.warning(f"Label processing error: {e}")
            
            # Add answer if available (for QA datasets)
            answer_key = dataset_config.get("answer_key")
            if answer_key and answer_key in item:
                try:
                    answer_data = item[answer_key]
                    if isinstance(answer_data, bytes):
                        answer_data = answer_data.decode('utf-8', errors='ignore')
                    processed_item["answer"] = str(answer_data)
                except (UnicodeDecodeError, AttributeError) as e:
                    logger.warning(f"Answer processing error: {e}")
            
            # Only return if we have at least text or audio data
            if "text" in processed_item or "audio" in processed_item:
                return processed_item
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Sample processing error: {e}")
            return None

    def _generate_synthetic_data(self):
        """Generate synthetic data as fallback"""
        logger.info("Generating synthetic multimodal data")

        num_samples = 1000 if self.split == "train" else 200

        for i in range(num_samples):
            text_length = random.randint(10, 50)
            # Handle both simple and complex config formats
            vocab_size = getattr(self.config, 'vocab_size', None)
            if vocab_size is None:
                vocab_size = getattr(self.config.model, 'vocab_size', 32000)
            text_tokens = torch.randint(1, vocab_size, (text_length,))

            # Generate realistic image tensors
            vision_dim = getattr(self.config, 'vision_dim', None)
            if vision_dim is None:
                vision_dim = getattr(self.config.model, 'vision_dim', 224)
            image = torch.randn(3, vision_dim, vision_dim)
            image = torch.clamp(image, -2, 2)  # Normalize range

            # Generate realistic audio
            # Handle both simple and complex config formats
            max_audio_length = getattr(self.config, 'max_audio_length', None)
            if max_audio_length is None:
                max_audio_length = getattr(self.config.data, 'max_audio_length', 16000)
            audio = torch.randn(max_audio_length) * 0.1

            task_type = random.choice(["text", "vision", "audio", "vision_text", "audio_text"])
            labels = text_tokens.clone() if task_type in ["text", "vision_text", "audio_text"] else None

            sample = {
                "text": text_tokens if task_type in ["text", "vision_text", "audio_text"] else None,
                "image": image if task_type in ["vision", "vision_text"] else None,
                "audio": audio if task_type in ["audio", "audio_text"] else None,
                "task_type": task_type,
                "labels": labels
            }

            # Ensure at least one modality is present
            if sample["text"] is not None or sample["image"] is not None or sample["audio"] is not None:
                self.processed_samples.append(sample)

    def __len__(self):
        """Return estimated length based on configured limits"""
        total_length = 0
        for wrapper in self.streaming_datasets:
            total_length += min(wrapper.dataset_config['limit'], self.max_samples_per_dataset)
        
        # Add synthetic data length if no streaming datasets
        if not self.streaming_datasets:
            total_length = 1000 if self.split == "train" else 200
            
        return total_length

    @property
    def data(self):
        """Property to access processed samples for vocabulary building"""
        # Ensure we have some data loaded
        if not self.processed_samples:
            # Load a few samples to build vocabulary
            for i in range(min(100, len(self))):
                try:
                    sample = self[i]
                    if sample and "text" in sample:
                        self.processed_samples.append(sample)
                except Exception as e:
                    logger.warning(f"Error loading sample {i}: {e}")
                    continue
        
        return self.processed_samples

    def __getitem__(self, idx):
        """Get item by index - implements hybrid approach"""
        # If we have processed samples in memory, return from there
        if idx < len(self.processed_samples):
            return self.processed_samples[idx]
        
        # Otherwise, get from streaming datasets
        if self.streaming_datasets:
            sample = self._get_next_sample()
            if sample:
                self.processed_samples.append(sample)
                return sample
        
        # Fallback to synthetic data
        if not self.processed_samples:
            self._generate_synthetic_data()
            if idx < len(self.processed_samples):
                return self.processed_samples[idx]
        
        # If all else fails, return a synthetic sample
        return self._generate_single_synthetic_sample()

    def _generate_single_synthetic_sample(self):
        """Generate a single synthetic sample on-demand"""
        text_length = random.randint(10, 50)
        # Handle both simple and complex config formats
        vocab_size = getattr(self.config, 'vocab_size', None)
        if vocab_size is None:
            vocab_size = getattr(self.config.model, 'vocab_size', 32000)
        text_tokens = torch.randint(1, vocab_size, (text_length,))
        
        sample = {
            "text": text_tokens,
            "image": None,
            "audio": None,
            "task_type": "text",
            "labels": text_tokens.clone()
        }
        
        return sample

    def __del__(self):
        """Cleanup streaming datasets"""
        for wrapper in self.streaming_datasets:
            wrapper.stop_prefetching()