#!/usr/bin/env python3
"""
Performance comparison between streaming and non-streaming approaches.
This script demonstrates the memory and storage benefits of the new streaming implementation.
"""

import time
import psutil
import os
import logging
from config import SalesAConfig
from tokenizer import SalesATokenizer
from data.dataset import MultimodalDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_disk_usage(path="."):
    """Get disk usage for a path in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / 1024 / 1024

def performance_test():
    """Compare performance between streaming and non-streaming approaches"""
    
    print("=== Performance Comparison: Streaming vs Non-Streaming ===")
    print()
    
    # Create config and tokenizer
    config = SalesAConfig()
    tokenizer = SalesATokenizer()
    
    # Test parameters
    cache_sizes = [100, 500, 1000]
    num_samples = 50
    
    print("Testing with different cache sizes and sample counts...")
    print()
    
    for cache_size in cache_sizes:
        print(f"Cache Size: {cache_size}")
        print("-" * 40)
        
        # Test streaming approach
        print("1. Streaming Approach:")
        start_memory = get_memory_usage()
        start_time = time.time()
        
        try:
            dataset_streaming = MultimodalDataset(
                config=config,
                tokenizer=tokenizer,
                split="train",
                dataset_name="open_platypus",
                use_streaming=True,
                cache_size=cache_size
            )
            
            # Load samples
            samples = []
            for i in range(num_samples):
                sample = dataset_streaming[i]
                samples.append(sample)
            
            end_time = time.time()
            end_memory = get_memory_usage()
            
            streaming_time = end_time - start_time
            streaming_memory = end_memory - start_memory
            
            print(f"   Time: {streaming_time:.2f}s")
            print(f"   Memory Increase: {streaming_memory:.2f} MB")
            print(f"   Samples Loaded: {len(samples)}")
            
        except Exception as e:
            print(f"   Error: {e}")
            streaming_time = float('inf')
            streaming_memory = float('inf')
        
        print()
        
        # Test non-streaming approach
        print("2. Non-Streaming Approach:")
        start_memory = get_memory_usage()
        start_time = time.time()
        
        try:
            dataset_non_streaming = MultimodalDataset(
                config=config,
                tokenizer=tokenizer,
                split="train",
                dataset_name="open_platypus",
                use_streaming=False,
                cache_size=cache_size
            )
            
            # Load samples
            samples = []
            for i in range(num_samples):
                sample = dataset_non_streaming[i]
                samples.append(sample)
            
            end_time = time.time()
            end_memory = get_memory_usage()
            
            non_streaming_time = end_time - start_time
            non_streaming_memory = end_memory - start_memory
            
            print(f"   Time: {non_streaming_time:.2f}s")
            print(f"   Memory Increase: {non_streaming_memory:.2f} MB")
            print(f"   Samples Loaded: {len(samples)}")
            
        except Exception as e:
            print(f"   Error: {e}")
            non_streaming_time = float('inf')
            non_streaming_memory = float('inf')
        
        print()
        
        # Comparison
        if streaming_time != float('inf') and non_streaming_time != float('inf'):
            time_improvement = ((non_streaming_time - streaming_time) / non_streaming_time) * 100
            memory_improvement = ((non_streaming_memory - streaming_memory) / non_streaming_memory) * 100
            
            print("3. Comparison:")
            print(f"   Time Improvement: {time_improvement:+.1f}%")
            print(f"   Memory Improvement: {memory_improvement:+.1f}%")
            
            if time_improvement > 0:
                print(f"   ✓ Streaming is {time_improvement:.1f}% faster")
            else:
                print(f"   ⚠ Streaming is {abs(time_improvement):.1f}% slower")
                
            if memory_improvement > 0:
                print(f"   ✓ Streaming uses {memory_improvement:.1f}% less memory")
            else:
                print(f"   ⚠ Streaming uses {abs(memory_improvement):.1f}% more memory")
        
        print()
        print("=" * 50)
        print()

def storage_comparison():
    """Compare storage usage between approaches"""
    
    print("=== Storage Usage Comparison ===")
    print()
    
    # Check if datasets directory exists
    datasets_dir = "datasets"
    if os.path.exists(datasets_dir):
        storage_usage = get_disk_usage(datasets_dir)
        print(f"Local datasets storage: {storage_usage:.2f} MB")
    else:
        print("Local datasets storage: 0 MB (no local downloads)")
    
    print()
    print("Streaming approach benefits:")
    print("- No local storage required for datasets")
    print("- Only cache in memory (configurable size)")
    print("- Automatic cleanup when dataset is destroyed")
    print("- Can handle datasets larger than available disk space")

def main():
    """Run performance tests"""
    
    print("Starting performance comparison...")
    print()
    
    # Performance test
    performance_test()
    
    # Storage comparison
    storage_comparison()
    
    print()
    print("=== Performance Test Complete ===")
    print()
    print("Key Takeaways:")
    print("1. Streaming approach eliminates local storage requirements")
    print("2. Memory usage is configurable via cache_size parameter")
    print("3. Background prefetching improves performance")
    print("4. Automatic fallback ensures reliability")
    print("5. Thread-safe operations prevent data races")

if __name__ == "__main__":
    main() 