# LUMA Dataset Usage Guide

## Overview

The LUMA (Learning from Uncertain and Multimodal Data) dataset is a multimodal dataset designed for benchmarking multimodal learning and uncertainty quantification. It contains audio, text, and image modalities.

## Important Note About Images

**⚠️ CRITICAL**: The Hugging Face version of LUMA (`bezirganyan/LUMA`) does **NOT** include images. The documentation clearly states:

> "This repository provides the Audio and Text modalities. The image modality consists of images from CIFAR-10/100 datasets. To download the image modality and compile the dataset with a specified amount of uncertainties, please use the [LUMA compilation tool](https://github.com/bezirganyan/LUMA)."

## Available Versions

### 1. Hugging Face Version (Audio + Text Only)
- **Source**: `bezirganyan/LUMA` on Hugging Face
- **Modalities**: Audio + Text only
- **Images**: ❌ Not included
- **Usage**: Direct download from Hugging Face

### 2. Full Compiled Version (Audio + Text + Images)
- **Source**: LUMA compilation tool
- **Modalities**: Audio + Text + Images
- **Images**: ✅ Included (CIFAR-10/100 + generated images)
- **Usage**: Requires separate compilation

## Implementation in SalesAI

The `MultimodalDataset` class has been corrected to properly handle both versions:

### Hugging Face Version Usage

```python
from data.dataset import MultimodalDataset
from config import SalesAConfig
from tokenizer import SalesATokenizer

# Initialize
config = SalesAConfig()
tokenizer = SalesATokenizer(config)

# Load LUMA from Hugging Face (audio + text only)
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    split="train",
    dataset_name="luma"
)

print(f"Loaded {len(dataset)} samples")
# Note: All samples will have has_image=False
```

### Full Compiled Version Usage

```python
# Load full LUMA dataset with images
compiled_data = dataset.load_luma_with_images("./path/to/luma_compiled")

if compiled_data:
    print(f"Loaded {len(compiled_data)} full multimodal samples")
    # Samples will have audio + text + images
```

## Getting the Full Dataset with Images

To get the complete LUMA dataset with images:

1. **Clone the LUMA compilation tool**:
   ```bash
   git clone https://github.com/bezirganyan/LUMA
   cd LUMA
   ```

2. **Follow the compilation instructions** in the LUMA repository

3. **Use the compiled dataset** in your SalesAI training:
   ```python
   # After compilation, load the full dataset
   compiled_data = dataset.load_luma_with_images("./path/to/compiled/luma")
   ```

## Dataset Structure

### Hugging Face Version
- **Audio**: WAV files from multiple sources (Common Voice, Spoken Wikipedia, LibriSpeech, etc.)
- **Text**: Generated text passages using Gemma 7B LLM
- **Labels**: 42 classes for training/testing, 8 classes for OOD

### Full Compiled Version
- **Audio**: Same as HF version
- **Text**: Same as HF version  
- **Images**: CIFAR-10/100 images + EDM-generated images
- **Images file**: `edm_images.pickle` (pandas DataFrame)

## Configuration

The corrected configuration in `dataset.py`:

```python
"luma": {
    "name": "bezirganyan/LUMA",
    "config": None,
    "has_image": False,  # Images not in HF dataset
    "has_text": True,
    "has_audio": True,
    "image_key": "image",
    "audio_key": "audio", 
    "text_key": "text",
    "limit": 2000 if self.split == "train" else 500,
    "max_samples": 1500,
    "note": "LUMA images require separate compilation tool. Only audio+text available in HF dataset."
}
```

## Testing

Run the test script to verify the implementation:

```bash
python test_luma_dataset.py
```

This will:
1. Test the Hugging Face version (audio + text)
2. Test the compiled version (if available)
3. Show sample data structure
4. Provide guidance on getting full multimodal data

## Troubleshooting

### Images Not Loading
- **Problem**: Images are `None` in samples
- **Solution**: Use the LUMA compilation tool to get the full dataset with images

### Dataset Not Found
- **Problem**: `bezirganyan/LUMA` not found
- **Solution**: Check internet connection and Hugging Face access

### Compilation Tool Issues
- **Problem**: Can't compile full dataset
- **Solution**: Check the LUMA repository for latest instructions and dependencies

## References

- **Paper**: [LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data](https://arxiv.org/abs/2406.09864)
- **Hugging Face**: [bezirganyan/LUMA](https://huggingface.co/datasets/bezirganyan/LUMA)
- **Compilation Tool**: [LUMA GitHub Repository](https://github.com/bezirganyan/LUMA)
- **Conference**: Accepted to SIGIR 2025

## License

- **Dataset**: CC BY-SA 4.0
- **Images**: Apache-2.0 (from DM-Improves-AT)
- **Audio sources**: Various (CC0, CC BY-SA 4.0, CC BY 4.0) 