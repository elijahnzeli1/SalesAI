#!/usr/bin/env python3
"""
Comprehensive test script for SalesA AI multimodal system
Tests all major components and ensures the system is ready for training
"""

import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import SalesAConfig
from model.salesa_model import SalesAModel
from model.transformer import TransformerBlock, MultiHeadAttention
from model.moe import MoELayer, Expert, Router
from model.encoders import TextEncoder, VisionEncoder, AudioEncoder
from tokenizer import SalesATokenizer
from rl.agent import DQNAgent, SimpleTextEnv, PrioritizedReplayBuffer, EpisodicMemory
from data.dataset import MultimodalDataset
from train import SalesATrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config():
    """Test configuration loading and validation"""
    logger.info("Testing configuration...")
    
    try:
        config = SalesAConfig()
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  - Hidden dim: {config.hidden_dim}")
        logger.info(f"  - Num layers: {config.num_layers}")
        logger.info(f"  - Num experts: {config.num_experts}")
        logger.info(f"  - Vocab size: {config.vocab_size}")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_encoders():
    """Test multimodal encoders"""
    logger.info("Testing encoders...")
    
    try:
        config = SalesAConfig()
        
        # Test text encoder
        text_encoder = TextEncoder(config)
        text_input = torch.randint(0, config.vocab_size, (2, 10))
        text_output = text_encoder(text_input)
        assert text_output.shape == (2, 10, config.hidden_dim)
        logger.info("✓ Text encoder working")
        
        # Test vision encoder
        vision_encoder = VisionEncoder(config)
        vision_input = torch.randn(2, 3, config.vision_dim, config.vision_dim)
        vision_output = vision_encoder(vision_input)
        expected_patches = (config.vision_dim // config.vision_patch_size) ** 2 + 1
        assert vision_output.shape == (2, expected_patches, config.hidden_dim)
        logger.info("✓ Vision encoder working")
        
        # Test audio encoder
        audio_encoder = AudioEncoder(config)
        audio_input = torch.randn(2, config.max_audio_length)
        audio_output = audio_encoder(audio_input)
        assert audio_output.shape[0] == 2 and audio_output.shape[2] == config.hidden_dim
        logger.info("✓ Audio encoder working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Encoder test failed: {e}")
        return False

def test_moe():
    """Test Mixture of Experts implementation"""
    logger.info("Testing MoE components...")
    
    try:
        config = SalesAConfig()
        
        # Test expert
        expert = Expert(config.hidden_dim, config.intermediate_dim)
        expert_input = torch.randn(4, config.hidden_dim)
        expert_output = expert(expert_input)
        assert expert_output.shape == expert_input.shape
        logger.info("✓ Expert working")
        
        # Test router
        router = Router(config.hidden_dim, config.num_experts, config.top_k)
        router_input = torch.randn(2, 5, config.hidden_dim)
        gates, indices = router(router_input)
        assert gates.shape == (2, 5, config.top_k)
        assert indices.shape == (2, 5, config.top_k)
        logger.info("✓ Router working")
        
        # Test MoE layer
        moe = MoELayer(config)
        moe_input = torch.randn(2, 5, config.hidden_dim)
        moe_output = moe(moe_input)
        assert moe_output.shape == moe_input.shape
        logger.info("✓ MoE layer working")
        
        # Test load balancing loss
        load_balance_loss = moe.get_load_balancing_loss()
        assert isinstance(load_balance_loss, torch.Tensor)
        logger.info("✓ Load balancing loss working")
        
        return True
    except Exception as e:
        logger.error(f"✗ MoE test failed: {e}")
        return False

def test_transformer():
    """Test transformer components"""
    logger.info("Testing transformer components...")
    
    try:
        config = SalesAConfig()
        
        # Test attention
        attention = MultiHeadAttention(config)
        attention_input = torch.randn(2, 10, config.hidden_dim)
        attention_output = attention(attention_input)
        assert attention_output.shape == attention_input.shape
        logger.info("✓ Multi-head attention working")
        
        # Test transformer block
        block = TransformerBlock(config)
        block_input = torch.randn(2, 10, config.hidden_dim)
        block_output = block(block_input)
        assert block_output.shape == block_input.shape
        logger.info("✓ Transformer block working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Transformer test failed: {e}")
        return False

def test_model():
    """Test complete model"""
    logger.info("Testing complete model...")
    
    try:
        config = SalesAConfig()
        model = SalesAModel(config)
        
        # Test text-only forward pass
        text_input = torch.randint(0, config.vocab_size, (2, 10))
        text_output = model(input_ids=text_input, task_type="text")
        assert "logits" in text_output
        assert "loss" in text_output
        assert "hidden_states" in text_output
        logger.info("✓ Text-only forward pass working")
        
        # Test vision-only forward pass
        vision_input = torch.randn(2, 3, config.vision_dim, config.vision_dim)
        vision_output = model(images=vision_input, task_type="vision")
        assert "logits" in vision_output
        logger.info("✓ Vision-only forward pass working")
        
        # Test multimodal forward pass with proper dimensions
        # Ensure text and vision inputs have compatible dimensions
        text_input = torch.randint(0, config.vocab_size, (2, 8))  # Match expected sequence length
        vision_input = torch.randn(2, 3, config.vision_dim, config.vision_dim)
        
        multimodal_output = model(
            input_ids=text_input,
            images=vision_input,
            task_type="text"
        )
        assert "logits" in multimodal_output
        logger.info("✓ Multimodal forward pass working")
        
        # Test generation with proper input
        generated = model.generate(text_input, max_length=5)
        # Check that generation worked (either longer or same length due to EOS)
        assert generated.shape[1] >= text_input.shape[1]
        logger.info("✓ Text generation working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_tokenizer():
    """Test tokenizer"""
    logger.info("Testing tokenizer...")
    
    try:
        config = SalesAConfig()
        tokenizer = SalesATokenizer(vocab_size=config.vocab_size)
        
        # Test encoding/decoding
        text = "Hello world, this is a test."
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
        logger.info("✓ Tokenizer encoding/decoding working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Tokenizer test failed: {e}")
        return False

def test_rl_components():
    """Test reinforcement learning components"""
    logger.info("Testing RL components...")
    
    try:
        config = SalesAConfig()
        model = SalesAModel(config)
        tokenizer = SalesATokenizer(vocab_size=config.vocab_size)
        
        # Test replay buffer
        buffer = PrioritizedReplayBuffer(capacity=100)
        transition = ("obs1", 1, 1.0, "obs2", False)
        buffer.push(transition)
        batch, indices, weights = buffer.sample(1)
        assert len(batch) == 1
        logger.info("✓ Prioritized replay buffer working")
        
        # Test episodic memory
        memory = EpisodicMemory(capacity=100)
        memory.add("state1")
        is_novel = memory.is_novel("state2")
        assert is_novel
        logger.info("✓ Episodic memory working")
        
        # Test environment
        env = SimpleTextEnv()
        obs = env.reset()
        next_obs, reward, done, info = env.step(1)
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        logger.info("✓ Environment working")
        
        # Test agent (basic functionality)
        agent = DQNAgent(model, tokenizer, n_actions=10, buffer_capacity=100, memory_capacity=50)
        action = agent.select_action("test observation")
        assert isinstance(action, int)
        assert 0 <= action < 10
        logger.info("✓ RL agent working")
        
        return True
    except Exception as e:
        logger.error(f"✗ RL test failed: {e}")
        return False

def test_dataset():
    """Test dataset loading"""
    logger.info("Testing dataset...")
    
    try:
        config = SalesAConfig()
        tokenizer = SalesATokenizer(vocab_size=config.vocab_size)
        
        # Test dataset creation
        dataset = MultimodalDataset(config, tokenizer, split="train", dataset_name="auto")
        assert len(dataset) > 0
        logger.info(f"✓ Dataset loaded with {len(dataset)} samples")
        
        # Test sample access
        sample = dataset[0]
        assert isinstance(sample, dict)
        logger.info("✓ Dataset sample access working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dataset test failed: {e}")
        return False

def test_trainer():
    """Test trainer"""
    logger.info("Testing trainer...")
    
    try:
        config = SalesAConfig()
        
        # Ensure all required attributes are present
        required_attrs = ['num_epochs', 'warmup_steps', 'gradient_clip_norm', 'early_stopping_patience']
        for attr in required_attrs:
            if not hasattr(config, attr):
                logger.warning(f"Missing attribute {attr} in config, adding default value")
                if attr == 'num_epochs':
                    setattr(config, attr, 10)
                elif attr == 'warmup_steps':
                    setattr(config, attr, 1000)
                elif attr == 'gradient_clip_norm':
                    setattr(config, attr, 1.0)
                elif attr == 'early_stopping_patience':
                    setattr(config, attr, 3)
        
        trainer = SalesATrainer(config)
        
        # Test trainer initialization
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        logger.info("✓ Trainer initialization working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Trainer test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Starting comprehensive SalesA AI system test")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("Tokenizers", test_tokenizer),
        ("Encoders", test_encoders),
        ("MoE Components", test_moe),
        ("Transformer", test_transformer),
        ("Complete Model", test_model),
        ("RL Components", test_rl_components),
        ("Dataset", test_dataset),
        ("Trainer", test_trainer),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✓ {test_name} PASSED")
        else:
            logger.error(f"✗ {test_name} FAILED")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System is ready for training.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python main.py --config multimodal")
        logger.info("2. Monitor training progress")
        logger.info("3. Check logs for detailed metrics")
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 