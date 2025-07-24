import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchaudio
from PIL import Image
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
import soundfile as sf
import io
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

from config import SalesAConfig
from tokenizer import SalesATokenizer

logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
    """Dataset for multimodal training with real datasets and action support"""

    def __init__(self, config: SalesAConfig, tokenizer: SalesATokenizer, split: str = "train", dataset_name: Union[str, List[str]] = "auto", task_type: Optional[str] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.data = self._load_data()   

    def _load_data(self) -> List[Dict]:
        """Load and prepare multimodal data from real datasets"""
        data = []

        # Define available datasets with their configurations
        dataset_configs = {
            # Audio–text question answering
            "clotho_aqa": {
                "name": "CLAPv2/ClothoAQA",
                "config": None,
                "has_image": False,
                "has_text": True,
                "has_audio": True,
                "audio_key": "audio",
                "text_key": "question",
                "answer_key": "answer",
                "limit": 1500 if self.split == "train" else 500,
                "max_samples": 1500 
            },
            # Tri‑modal: image + audio + text
            "luma": {
                "name": "bezirganyan/LUMA",
                "config": None,
                "has_image": True,
                "has_text": True,
                "has_audio": True,
                "image_key": "image",
                "audio_key": "audio",
                "text_key": "text",
                "limit": 2000 if self.split == "train" else 500,
                "max_samples": 1500
            },
            # Beans you already have:
            "beans": {
                "name": "AI-Lab-Makerere/beans",
                "config": None,
                "has_image": True,
                "has_text": False,
                "has_audio": False,
                "image_key": "image",
                "text_key": None,
                "limit": 1000 if self.split == "train" else 200,
                "max_samples": 1000
            },
            "prosocial_dialog": {
                "name": "allenai/prosocial-dialog",
                "config": None,
                "has_image": False,
                "has_text": True,
                "has_audio": False,
                "text_key": "text",
                "labels_key": "labels",
                "answer_key": None,
                "limit": 5000 if self.split == "train" else 1000,
                "max_samples": 2000
            },
            "logic_reasoning": {
                "name": "LogiQA",
                "config": None,
                "has_text": True,
                "task": "multiple_choice",
                "limit": 8000 if self.split=="train" else 2000,
                "max_samples": 10000
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
            "financial_phrasebank": {
                "name": "atrost/financial_phrasebank",
                "config": "sentences_50agree",
                "has_image": False,
                "has_text": True,
                "has_audio": False,
                "text_key": "sentence",
                "labels_key": "label",
                "limit": 5000 if self.split == "train" else 1000,
                "max_samples": 5000
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
            "ds1000": {
                "name": "code-rag-bench/ds1000",
                "config": None,
                "has_image": False,
                "has_text": True,
                "has_audio": False,
                "text_key": "prompt",
                "labels_key": "solution",
                "limit": 1000 if self.split == "train" else 200,
                "max_samples": 10000
            },
        }

        # --- Enhanced automatic selection logic ---
        if self.dataset_name == "auto":
            # Select default dataset(s) based on task_type
            if hasattr(self, 'task_type') and self.task_type:
                if self.task_type in ["code", "code-generation"]:
                    selected_datasets = ["humaneval", "ds1000"]
                elif self.task_type == "vision":
                    selected_datasets = ["beans"]
                elif self.task_type == "audio":
                    selected_datasets = ["clotho_aqa"]
                elif self.task_type in ["financial", "stock"]:
                    selected_datasets = ["financial_phrasebank"]
                elif self.task_type == "text":
                    selected_datasets = ["prosocial_dialog"]
                else:
                    selected_datasets = ["logic_reasoning"]
            else:
                # Fallback to general text dataset
                selected_datasets = ["open_platypus"]
        elif self.dataset_name == "all":
            selected_datasets = list(dataset_configs.keys())
        elif isinstance(self.dataset_name, list):
            selected_datasets = self.dataset_name
        else:
            selected_datasets = [self.dataset_name]

        # Load each selected dataset
        for dataset_key in selected_datasets:
            if dataset_key not in dataset_configs:
                logger.warning(f"Unknown dataset: {dataset_key}")
                continue

            dataset_config = dataset_configs[dataset_key]
            try:
                # Attempt to load the dataset
                loaded_data = self._load_single_dataset(dataset_config)
                data.extend(loaded_data)
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_key}: {e}")
                # Continue to the next dataset even if one fails

        # If no real data was loaded, fall back to synthetic data
        if not data:
            logger.warning("No real datasets loaded successfully. Generating synthetic data.")
            data = self._generate_synthetic_data()

        logger.info(f"Total samples loaded: {len(data)}")
        return data

    def _ensure_local_dataset(self, repo_id, config=None):
        """Ensure the dataset is downloaded locally and return the local path. Skip download if already present."""
        import os
        local_dir = os.path.join("datasets", repo_id.replace('/', '__'))
        marker_file = os.path.join(local_dir, ".download_complete")
        if not os.path.exists(local_dir) or not os.path.exists(marker_file):
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
            # Create marker file to indicate download is complete
            os.makedirs(local_dir, exist_ok=True)
            with open(marker_file, "w") as f:
                f.write("downloaded")
        return local_dir

    def _load_single_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load a single dataset based on its configuration, preferring local download."""
        data = []
        logger.info(f"Loading dataset: {dataset_config['name']}")
        try:
            # Handle special split cases
            if dataset_config['name'] == "bezirganyan/LUMA" and self.split == "validation":
                logger.warning("LUMA doesn't have validation split. Using test split instead.")
                split_to_use = "test"
            elif dataset_config['name'] == "garage-bAInd/Open-Platypus" and self.split == "validation":
                 split_to_use = "train"
            else:
                split_to_use = self.split

            # Limit download size by max_samples if present
            max_samples = dataset_config.get('max_samples', None)
            if max_samples:
                split_str = f"{split_to_use}[:{max_samples}]"
            else:
                split_str = split_to_use

            # Download and load from local directory if not already local
            if not os.path.exists(dataset_config['name']) and not dataset_config['name'].startswith("./"):
                local_dir = self._ensure_local_dataset(dataset_config['name'], dataset_config.get('config'))
                if dataset_config['config']:
                    dataset = load_dataset(
                        local_dir,
                        dataset_config['config'],
                        split=split_str,
                        streaming=False
                    )
                else:
                    dataset = load_dataset(
                        local_dir,
                        split=split_str,
                        streaming=False
                    )
            else:
                # Already local or custom path
                if dataset_config['config']:
                    dataset = load_dataset(
                        dataset_config['name'],
                        dataset_config['config'],
                        split=split_str,
                        streaming=False
                    )
                else:
                    dataset = load_dataset(
                        dataset_config['name'],
                        split=split_str,
                        streaming=False
                    )

            # Process samples
            count = 0
            first_sample = None
            for item in dataset:
                if count == 0:
                    first_sample = item
                if count >= dataset_config['limit']:
                    break
                if not isinstance(item, dict):
                    continue
                try:
                    sample = self._process_sample(item, dataset_config)
                    if sample:
                        data.append(sample)
                        count += 1
                        if count % 1000 == 0:
                            logger.info(f"Processed {count} samples from {dataset_config['name']}")
                except Exception as e:
                    logger.warning(f"Error processing sample {count} from {dataset_config['name']}: {e}")
                    continue
            logger.info(f"Successfully loaded {len(data)} samples from {dataset_config['name']}")
            if count == 0:
                logger.warning(f"No samples loaded from {dataset_config['name']} (split: {split_to_use}). Check split/key names.")
                if first_sample is not None:
                    if isinstance(first_sample, dict):
                        logger.warning(f"First sample keys: {list(first_sample.keys())}")
                    else:
                        logger.warning(f"First sample type: {type(first_sample)}, value: {first_sample}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_config['name']}: {e}")
            # Generate synthetic data for this dataset instead of failing completely
            logger.info(f"Generating synthetic data for {dataset_config['name']}")
            synthetic_data = self._generate_synthetic_data_for_dataset(dataset_config)
            data.extend(synthetic_data)

        logger.info(f"Successfully loaded {len(data)} samples from {dataset_config['name']}")
        return data

    def _generate_synthetic_data_for_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Generate synthetic data for a specific dataset configuration"""
        num_samples = dataset_config['limit']
        synthetic_data = []

        logger.info(f"Generating {num_samples} synthetic samples for {dataset_config['name']}")

        for i in range(num_samples):
            # Create sample based on dataset configuration
            sample = {
                "text": None,
                "image": None,
                "audio": None,
                "task_type": "text",
                "labels": None,
                "action_labels": None  # For robotics
            }

            # Generate text if needed
            if dataset_config['has_text']:
                text_length = random.randint(10, 50)
                text_tokens = torch.randint(1, self.config.vocab_size, (text_length,))
                sample["text"] = text_tokens
                sample["labels"] = text_tokens.clone()

            # Generate image if needed
            if dataset_config['has_image']:
                image = torch.randn(3, self.config.vision_dim, self.config.vision_dim)
                image = torch.clamp(image, -2, 2)
                sample["image"] = image
                # --- FIX: Always generate a valid label for vision samples ---
                sample["labels"] = torch.tensor([random.randint(0, 9)], dtype=torch.long)  # 10-class synthetic label

            # Generate audio if needed
            if dataset_config['has_audio']:
                audio = torch.randn(self.config.max_audio_length) * 0.1
                sample["audio"] = audio

            # Generate action label for robotics
            if random.random() < 0.2:  # 20% of samples are action samples
                sample["action_labels"] = torch.tensor([random.randint(0, self.config.action_dim - 1)], dtype=torch.long)
                sample["task_type"] = "action"

            # Determine task type
            if sample["image"] is not None and sample["text"] is not None:
                sample["task_type"] = "vision_text"
            elif sample["audio"] is not None and sample["text"] is not None:
                sample["task_type"] = "audio_text"
            elif sample["text"] is not None:
                sample["task_type"] = "text"
            elif sample["image"] is not None:
                sample["task_type"] = "vision"
            elif sample["audio"] is not None:
                sample["task_type"] = "audio"

            synthetic_data.append(sample)

        return synthetic_data

    def _process_sample(self, item: Dict, dataset_config: Dict) -> Optional[Dict]:
        """Process a single sample from a dataset"""

        # Initialize sample
        sample = {
            "text": None,
            "image": None,
            "audio": None,
            "task_type": "text",
            "labels": None
        }

        # Process text
        if dataset_config['has_text'] and dataset_config['text_key'] in item:
            text_content = item[dataset_config['text_key']]
            # Prepend <STOCK> token for financial_phrasebank
            if dataset_config.get('name', '') == 'financial_phrasebank':
                text_content = "<STOCK> " + text_content
            # Handle different text formats
            if isinstance(text_content, list):
                # For datasets with multiple captions
                text_content = random.choice(text_content)
            if isinstance(text_content, dict):
                text_content = text_content.get('raw', str(text_content))
            if text_content and isinstance(text_content, str) and len(text_content.strip()) > 0:
                try:
                    text_tokens = torch.tensor(self.tokenizer.encode(text_content), dtype=torch.long)
                    sample["text"] = text_tokens
                    sample["labels"] = text_tokens.clone()
                except Exception as e:
                    logger.warning(f"Error tokenizing text: {e}")
            # For financial_phrasebank, set sentiment label as classification label
            if dataset_config.get('name', '') == 'financial_phrasebank' and 'labels_key' in dataset_config and dataset_config['labels_key'] in item:
                sample["labels"] = torch.tensor([int(item[dataset_config['labels_key']])], dtype=torch.long)

        # Process images
        if dataset_config['has_image'] and dataset_config['image_key'] in item:
            try:
                image_data = item[dataset_config['image_key']]

                # Handle different image formats
                if isinstance(image_data, str):
                    # Skip URL images for now
                    logger.warning(f"Skipping image with URL: {image_data}")
                elif hasattr(image_data, 'convert'):
                    # PIL Image
                    img = image_data.convert("RGB")
                    transform = transforms.Compose([
                        transforms.Resize((self.config.vision_dim, self.config.vision_dim)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    sample["image"] = transform(img)
                elif isinstance(image_data, np.ndarray):
                    # Array format
                    img = Image.fromarray(image_data).convert("RGB")
                    transform = transforms.Compose([
                        transforms.Resize((self.config.vision_dim, self.config.vision_dim)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    sample["image"] = transform(img)
                else:
                    logger.warning(f"Unsupported image format: {type(image_data)}")

            except Exception as e:
                logger.warning(f"Error processing image: {e}")

        # Process audio - using soundfile as a fallback
        if dataset_config['has_audio'] and dataset_config['audio_key'] in item:
            try:
                audio_data = item[dataset_config['audio_key']]
                waveform = None
                sample_rate = 16000

                # Handle different audio formats
                if isinstance(audio_data, dict) and 'array' in audio_data:
                    # Standard audio dictionary format
                    waveform = torch.tensor(audio_data['array'], dtype=torch.float32)
                    sample_rate = audio_data.get('sampling_rate', 16000)
                elif isinstance(audio_data, bytes):
                    # Audio bytes
                    with io.BytesIO(audio_data) as f:
                        data, sample_rate = sf.read(f)
                        waveform = torch.tensor(data, dtype=torch.float32)
                elif isinstance(audio_data, str):
                    # File path - skip for now
                    logger.warning(f"Skipping audio file path: {audio_data}")
                elif isinstance(audio_data, np.ndarray):
                    # Numpy array
                    waveform = torch.tensor(audio_data, dtype=torch.float32)

                if waveform is not None:
                    # Resample if necessary
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=16000
                        )
                        waveform = resampler(waveform)

                    # Ensure mono
                    if waveform.dim() > 1:
                        waveform = waveform.mean(dim=0)

                    # Pad or truncate
                    if len(waveform) > self.config.max_audio_length:
                        waveform = waveform[:self.config.max_audio_length]
                    else:
                        waveform = F.pad(waveform, (0, self.config.max_audio_length - len(waveform)))

                    sample["audio"] = waveform

            except Exception as e:
                logger.warning(f"Error processing audio: {e}")

        # --- FIX: For vision-only samples, ensure a valid label is present ---
        if dataset_config.get('has_image', False) and not dataset_config.get('has_text', False):
            # Try to get a label for image classification
            label = item.get('label') or item.get('labels')
            if label is not None:
                # If label is a string, map to int or use tokenizer if vision-to-text
                if isinstance(label, str):
                    # Use tokenizer if vision-to-text, else hash to int
                    label_id = self.tokenizer.encode(label)[0] if hasattr(self, 'tokenizer') else abs(hash(label)) % 1000
                else:
                    label_id = int(label)
                sample["labels"] = torch.tensor([label_id], dtype=torch.long)
            else:
                # If no label, skip this sample
                return None

        # Handle VQA specific format
        if 'answer_key' in dataset_config and dataset_config['answer_key'] in item:
            answer = item[dataset_config['answer_key']]
            if sample["text"] is not None:
                try:
                    question_text = self.tokenizer.decode(sample["text"].tolist())
                    combined_text = f"Question: {question_text} Answer: {answer}"
                    combined_tokens = torch.tensor(self.tokenizer.encode(combined_text), dtype=torch.long)
                    sample["labels"] = combined_tokens
                except:
                    pass

        # Determine the primary task type
        if sample["image"] is not None and sample["text"] is not None:
            sample["task_type"] = "vision_text"
        elif sample["audio"] is not None and sample["text"] is not None:
            sample["task_type"] = "audio_text"
        elif sample["text"] is not None:
            sample["task_type"] = "text"
        elif sample["image"] is not None:
            sample["task_type"] = "vision"
        elif sample["audio"] is not None:
            sample["task_type"] = "audio"
        else:
            # If no valid modality data, skip
            return None

        # Ensure labels are set
        if sample["task_type"] in ["text", "vision_text", "audio_text"] and sample["labels"] is None:
            if sample["text"] is not None:
                sample["labels"] = sample["text"].clone()
            else:
                return None

        return sample

    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic data as fallback"""
        logger.info("Generating synthetic multimodal data")

        data = []
        num_samples = 1000 if self.split == "train" else 200

        for i in range(num_samples):
            text_length = random.randint(10, 50)
            text_tokens = torch.randint(1, self.config.vocab_size, (text_length,))

            # Generate realistic image tensors
            image = torch.randn(3, self.config.vision_dim, self.config.vision_dim)
            image = torch.clamp(image, -2, 2)  # Normalize range

            # Generate realistic audio
            audio = torch.randn(self.config.max_audio_length) * 0.1

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
                data.append(sample)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 