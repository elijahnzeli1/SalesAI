#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

print("🔍 Testing imports...")

try:
    # Test basic imports
    import torch
    print("✅ PyTorch imported successfully")
    
    import numpy as np
    print("✅ NumPy imported successfully")
    
    from datasets import load_dataset
    print("✅ Datasets imported successfully")
    
    # Test optional imports
    try:
        import torchvision
        print("✅ TorchVision imported successfully")
    except ImportError as e:
        print(f"⚠️  TorchVision not available: {e}")
    
    try:
        import torchaudio
        print("✅ TorchAudio imported successfully")
    except ImportError as e:
        print(f"⚠️  TorchAudio not available: {e}")
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"⚠️  PIL not available: {e}")
    
    # Test our custom modules
    from config import SalesAConfig
    print("✅ SalesAConfig imported successfully")
    
    from tokenizer import SalesATokenizer
    print("✅ SalesATokenizer imported successfully")
    
    from data.dataset import MultimodalDataset
    print("✅ MultimodalDataset imported successfully")
    
    print("\n🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc() 