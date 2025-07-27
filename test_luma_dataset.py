#!/usr/bin/env python3
"""
Test script for LUMA dataset loading
Demonstrates the corrected implementation and shows how to use both versions.
"""

import logging
import torch
from data.dataset import MultimodalDataset
from config import SalesAConfig
from tokenizer import SalesATokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_luma_hf_version():
    """Test loading LUMA dataset from Hugging Face (audio + text only)"""
    print("=" * 60)
    print("Testing LUMA Dataset - Hugging Face Version (Audio + Text Only)")
    print("=" * 60)
    
    # Initialize config and tokenizer
    config = SalesAConfig()
    tokenizer = SalesATokenizer(vocab_size=config.vocab_size)
    
    # Load LUMA dataset from Hugging Face
    dataset = MultimodalDataset(
        config=config,
        tokenizer=tokenizer,
        split="train",
        dataset_name="luma",
        task_type="multimodal"
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Check a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Task type: {sample['task_type']}")
        print(f"  Has text: {sample['text'] is not None}")
        print(f"  Has audio: {sample['audio'] is not None}")
        print(f"  Has image: {sample['image'] is not None}")
        
        if sample['text'] is not None:
            text_decoded = tokenizer.decode(sample['text'].tolist())
            print(f"  Text preview: {text_decoded[:100]}...")
        
        if sample['audio'] is not None:
            print(f"  Audio shape: {sample['audio'].shape}")
    
    print("\nNote: Images are not available in the Hugging Face version.")
    print("To get full multimodal data with images, use the LUMA compilation tool.")

def test_luma_compiled_version():
    """Test loading full LUMA dataset with images (if available)"""
    print("\n" + "=" * 60)
    print("Testing LUMA Dataset - Compiled Version (Full Multimodal)")
    print("=" * 60)
    
    # Initialize config and tokenizer
    config = SalesAConfig()
    tokenizer = SalesATokenizer(vocab_size=config.vocab_size)
    
    # Try to load compiled version
    compiled_path = "./luma_compiled"  # Adjust path as needed
    
    try:
        # Create a dataset instance to access the method
        dataset = MultimodalDataset(
            config=config,
            tokenizer=tokenizer,
            split="train",
            dataset_name="luma"
        )
        
        # Try to load compiled version
        compiled_data = dataset.load_luma_with_images(compiled_path)
        
        if compiled_data:
            print(f"Compiled dataset loaded: {len(compiled_data)} samples")
            
            # Check a few samples
            for i in range(min(3, len(compiled_data))):
                sample = compiled_data[i]
                print(f"\nCompiled Sample {i}:")
                print(f"  Task type: {sample['task_type']}")
                print(f"  Has text: {sample['text'] is not None}")
                print(f"  Has audio: {sample['audio'] is not None}")
                print(f"  Has image: {sample['image'] is not None}")
                
                if sample['text'] is not None:
                    text_decoded = tokenizer.decode(sample['text'].tolist())
                    print(f"  Text preview: {text_decoded[:100]}...")
                
                if sample['image'] is not None:
                    print(f"  Image shape: {sample['image'].shape}")
                
                if sample['audio'] is not None:
                    print(f"  Audio shape: {sample['audio'].shape}")
        else:
            print("Compiled dataset not found or failed to load.")
            print("To get the full LUMA dataset with images:")
            print("1. Clone the LUMA compilation tool: https://github.com/bezirganyan/LUMA")
            print("2. Follow the compilation instructions")
            print("3. Update the path in this script to point to your compiled dataset")
    
    except Exception as e:
        print(f"Error testing compiled version: {e}")

def main():
    """Main test function"""
    print("LUMA Dataset Test Script")
    print("This script demonstrates the corrected LUMA dataset implementation.")
    
    # Test Hugging Face version (always available)
    test_luma_hf_version()
    
    # Test compiled version (if available)
    test_luma_compiled_version()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Hugging Face version: Audio + Text only (no images)")
    print("- Compiled version: Full multimodal (Audio + Text + Images)")
    print("- Use the LUMA compilation tool for full multimodal data")
    print("=" * 60)

if __name__ == "__main__":
    main() 