# Streaming Dataset Implementation

This document describes the new hybrid streaming approach for loading datasets in the SalesAI project, which eliminates the need to download datasets to local storage.

## Overview

The new implementation replaces the previous approach of downloading datasets locally with a hybrid streaming system that:

- **Streams data directly** from Hugging Face datasets without local storage
- **Uses background prefetching** for better performance
- **Implements intelligent caching** to reduce network overhead
- **Provides automatic fallback** to synthetic data if streaming fails
- **Supports multiple datasets** with automatic selection based on task type

## Key Features

### 1. StreamingDatasetWrapper
A wrapper class that provides:
- **Thread-safe caching** with configurable cache size
- **Background prefetching** to improve data loading speed
- **Automatic cleanup** of resources when the dataset is destroyed

### 2. Hybrid Loading Approach
The `MultimodalDataset` class now supports:
- **Streaming mode** (default): Loads data directly from Hugging Face
- **Non-streaming mode**: Fallback for datasets that don't support streaming
- **Synthetic data generation**: Automatic fallback when streaming fails

### 3. Smart Dataset Selection
Automatic dataset selection based on task type:
- `code` → HumanEval (code generation)
- `vision` → Beans (image classification)
- `audio` → LUMA (audio-text)
- `text` → Open-Platypus (general text)
- `financial` → Financial PhraseBank (sentiment analysis)

## Usage

### Basic Usage

```python
from data.dataset import MultimodalDataset
from config import SalesAConfig
from tokenizer import SalesATokenizer

# Create dataset with streaming (default)
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    split="train",
    dataset_name="open_platypus",
    use_streaming=True,  # Default
    cache_size=1000      # Configurable cache size
)
```

### Advanced Usage

```python
# Multiple datasets with automatic selection
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    split="train",
    dataset_name="all",  # Load all available datasets
    task_type="code",    # Automatically select code-related datasets
    use_streaming=True,
    cache_size=2000
)

# Custom dataset list
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    split="train",
    dataset_name=["humaneval", "open_platypus"],
    use_streaming=True
)
```

## Configuration Options

### Constructor Parameters

- `use_streaming` (bool): Enable/disable streaming (default: True)
- `cache_size` (int): Number of samples to cache in memory (default: 1000)
- `dataset_name`: Can be:
  - `"auto"`: Automatic selection based on task_type
  - `"all"`: Load all available datasets
  - `str`: Single dataset name
  - `List[str]`: Multiple dataset names
- `task_type`: Influences automatic dataset selection

### Performance Tuning

```python
# For memory-constrained environments
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    cache_size=100,  # Smaller cache
    use_streaming=True
)

# For high-performance environments
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    cache_size=5000,  # Larger cache
    use_streaming=True
)
```

## Supported Datasets

### Currently Supported
1. **LUMA** (`bezirganyan/LUMA`)
   - Modalities: Audio + Text
   - Note: Images require separate compilation tool

2. **Open-Platypus** (`garage-bAInd/Open-Platypus`)
   - Modalities: Text only
   - Task: Instruction following

3. **HumanEval** (`code-rag-bench/humaneval`)
   - Modalities: Text only
   - Task: Code generation

### Previously Supported (Commented Out)
- ClothoAQA (audio-text QA)
- Beans (image classification)
- Prosocial Dialog (text)
- LogicQA (reasoning)
- Financial PhraseBank (sentiment)
- DS1000 (code generation)

## Architecture

### Data Flow
```
Hugging Face Dataset
        ↓
StreamingDatasetWrapper
        ↓
Background Prefetching
        ↓
Thread-Safe Cache
        ↓
MultimodalDataset.__getitem__()
        ↓
Processed Sample
```

### Thread Safety
- **Cache access** is protected by locks
- **Prefetching** runs in background threads
- **Cleanup** is handled automatically in `__del__`

## Benefits

### Storage Efficiency
- **No local downloads** required
- **Minimal disk usage** (only cache)
- **Automatic cleanup** of temporary data

### Performance
- **Background prefetching** reduces latency
- **Intelligent caching** balances memory and speed
- **Streaming** enables handling of large datasets

### Reliability
- **Automatic fallback** to synthetic data
- **Error handling** for network issues
- **Graceful degradation** when datasets are unavailable

### Flexibility
- **Multiple dataset support** with automatic selection
- **Configurable caching** for different environments
- **Easy extension** for new datasets

## Testing

Run the test script to verify the streaming functionality:

```bash
python test_streaming_dataset.py
```

This will test:
1. Streaming mode with real datasets
2. Non-streaming fallback
3. Automatic dataset selection
4. Error handling and fallback mechanisms

## Migration from Old Implementation

### Before (Local Download)
```python
# Old approach - downloads to local storage
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    split="train",
    dataset_name="open_platypus"
)
```

### After (Streaming)
```python
# New approach - streams directly
dataset = MultimodalDataset(
    config=config,
    tokenizer=tokenizer,
    split="train",
    dataset_name="open_platypus",
    use_streaming=True  # Default behavior
)
```

### Backward Compatibility
The old methods are deprecated but still available:
- `_load_data()` → Returns empty list, logs warning
- `_ensure_local_dataset()` → Returns None, logs warning
- `_load_single_dataset()` → Returns empty list, logs warning

## Troubleshooting

### Common Issues

1. **Network Connectivity**
   - Check internet connection
   - Verify Hugging Face access
   - Dataset will fall back to synthetic data

2. **Memory Issues**
   - Reduce `cache_size` parameter
   - Monitor memory usage during training
   - Consider using smaller datasets

3. **Performance Issues**
   - Increase `cache_size` for better performance
   - Check network bandwidth
   - Monitor prefetching thread activity

### Debug Mode
Enable detailed logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

dataset = MultimodalDataset(...)
```

## Future Enhancements

1. **Dataset Caching**: Persistent cache across sessions
2. **Compression**: Compress cached samples to save memory
3. **Priority Queuing**: Prioritize important samples in cache
4. **Distributed Streaming**: Support for distributed training
5. **Custom Dataset Support**: Easy integration of new datasets

## Contributing

To add support for new datasets:

1. Add dataset configuration to `dataset_configs` in `_initialize_streaming_datasets()`
2. Implement appropriate processing in `_process_sample()`
3. Add to automatic selection logic if needed
4. Test with the provided test script 