#!/usr/bin/env python3
"""
Test script for the new streaming dataset functionality.
This demonstrates how the hybrid streaming approach works without downloading datasets locally.
"""

import torch
import logging
from config import SalesAConfig
from tokenizer import SalesATokenizer
from data.dataset import MultimodalDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_streaming_dataset():
    """Test the streaming dataset functionality"""
    
    # Create a minimal config for testing
    config = SalesAConfig()
    
    # Create a simple tokenizer
    tokenizer = SalesATokenizer()
    
    print("=== Testing Streaming Dataset ===")
    print("This will load datasets directly from Hugging Face without downloading to local storage.")
    print()
    
    # Test with streaming enabled (default)
    print("1. Testing with streaming enabled:")
    try:
        dataset = MultimodalDataset(
            config=config,
            tokenizer=tokenizer,
            split="train",
            dataset_name="open_platypus",  # Use a smaller dataset for testing
            use_streaming=True,
            cache_size=100
        )
        
        print(f"   Dataset length: {len(dataset)}")
        print(f"   Number of streaming datasets: {len(dataset.streaming_datasets)}")
        
        # Get a few samples
        for i in range(3):
            sample = dataset[i]
            print(f"   Sample {i}: task_type={sample['task_type']}, text_length={len(sample['text']) if sample['text'] is not None else 0}")
        
        print("   ✓ Streaming test passed!")
        
    except Exception as e:
        print(f"   ✗ Streaming test failed: {e}")
    
    print()
    
    # Test with streaming disabled (fallback)
    print("2. Testing with streaming disabled (fallback to synthetic data):")
    try:
        dataset_no_streaming = MultimodalDataset(
            config=config,
            tokenizer=tokenizer,
            split="train",
            dataset_name="open_platypus",
            use_streaming=False,
            cache_size=100
        )
        
        print(f"   Dataset length: {len(dataset_no_streaming)}")
        print(f"   Number of streaming datasets: {len(dataset_no_streaming.streaming_datasets)}")
        
        # Get a few samples
        for i in range(3):
            sample = dataset_no_streaming[i]
            print(f"   Sample {i}: task_type={sample['task_type']}, text_length={len(sample['text']) if sample['text'] is not None else 0}")
        
        print("   ✓ Non-streaming test passed!")
        
    except Exception as e:
        print(f"   ✗ Non-streaming test failed: {e}")
    
    print()
    
    # Test automatic dataset selection
    print("3. Testing automatic dataset selection:")
    try:
        dataset_auto = MultimodalDataset(
            config=config,
            tokenizer=tokenizer,
            split="train",
            dataset_name="auto",
            task_type="code",  # This should select humaneval
            use_streaming=True,
            cache_size=100
        )
        
        print(f"   Dataset length: {len(dataset_auto)}")
        print(f"   Number of streaming datasets: {len(dataset_auto.streaming_datasets)}")
        
        if dataset_auto.streaming_datasets:
            dataset_name = dataset_auto.streaming_datasets[0].dataset_config['name']
            print(f"   Selected dataset: {dataset_name}")
        
        print("   ✓ Auto-selection test passed!")
        
    except Exception as e:
        print(f"   ✗ Auto-selection test failed: {e}")
    
    print()
    print("=== Streaming Dataset Test Complete ===")
    print()
    print("Key benefits of the new streaming approach:")
    print("- No local storage required for datasets")
    print("- Data is loaded on-demand during training")
    print("- Background prefetching for better performance")
    print("- Automatic fallback to synthetic data if streaming fails")
    print("- Configurable cache size for memory management")
    print("- Thread-safe prefetching with proper cleanup")

if __name__ == "__main__":
    test_streaming_dataset() 