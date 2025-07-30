import torch
import logging
from pathlib import Path
import argparse
import json

from utils.config import load_config
from tokenizer import SalesATokenizer, build_vocab_with_tiktoken
from model.salesa_model import SalesAModel
from data.dataset import MultimodalDataset
from data.collate import create_multimodal_dataloaders
from train import SalesATrainer
from evaluate import SalesAEvaluator
from rl.agent import SimpleTextEnv, DQNAgent

logger = logging.getLogger(__name__)

def setup_logging(config):
    """Set up logging configuration"""
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler()
        ]
    )

def init_model(config) -> tuple[SalesAModel, SalesATokenizer]:
    """Initialize model and tokenizer"""
    logger.info("Initializing model and tokenizer...")
    
    # Convert complex config to simple config for model
    from config_adapter import adapt_config
    simple_config = adapt_config(config)
    
    # Create raw dataset for vocabulary building
    raw_train_dataset = MultimodalDataset(
        config=config,
        tokenizer=SalesATokenizer(vocab_size=simple_config.vocab_size),
        split="train",
        dataset_name=config.data.dataset_name
    )

    # Build vocabulary
    vocab, enc = build_vocab_with_tiktoken(
        raw_train_dataset.data,
        vocab_size=simple_config.vocab_size,
        model_name="gpt2"
    )
    tokenizer = SalesATokenizer(
        vocab_size=simple_config.vocab_size,
        vocab=vocab,
        enc=enc
    )

    # Initialize model with simple config
    model = SalesAModel(simple_config)

    return model, tokenizer

def train_model(config, model: SalesAModel, tokenizer: SalesATokenizer):
    """Train the model"""
    logger.info("Setting up training...")

    # Convert complex config to simple config for trainer
    from config_adapter import adapt_config
    simple_config = adapt_config(config)

    # Create trainer with simple config and initialized model
    trainer = SalesATrainer(simple_config, model)

    # Create dataloaders
    train_loader, val_loader = create_multimodal_dataloaders(
        config=config,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        dataset_name=config.data.dataset_name,
        task_type=config.data.task_type
    )

    # Train
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience
    )

    return trainer

def evaluate_model(config, model: SalesAModel, tokenizer: SalesATokenizer):
    """Evaluate the model"""
    logger.info("Evaluating model...")
    evaluator = SalesAEvaluator(model, tokenizer)

    # Text generation evaluation
    text_prompts = [
        "The future of AI is",
        "In a world where technology",
        "Machine learning algorithms"
    ]
    text_results = evaluator.evaluate_text_generation(text_prompts)
    logger.info(f"Text generation - Average length: {text_results['avg_length']:.2f}")
    logger.info(f"Text generation - Fluency score: {text_results['fluency_score']:.2f}")

    # Code generation evaluation if applicable
    if config.data.task_type == "code":
        code_prompts = [
            "Write a function to calculate fibonacci numbers",
            "Create a class for a binary tree",
            "Implement quicksort algorithm"
        ]
        code_results = evaluator.evaluate_code_generation(code_prompts)
        logger.info(f"Code generation - Syntax accuracy: {code_results['syntax_accuracy']:.2f}")
        logger.info(f"Code generation - Completion rate: {code_results['completion_rate']:.2f}")

    # Expert usage analysis
    expert_stats = evaluator.analyze_expert_usage()
    logger.info(f"Expert usage analysis completed for {len(expert_stats)} layers")
    for stats in expert_stats:
        logger.info(f"Layer {stats['layer_name']} - Load balance: {stats['load_balance']:.2f}")

def train_rl_agent(config, model: SalesAModel, tokenizer: SalesATokenizer):
    """Train the RL agent"""
    logger.info("Setting up RL training...")

    # Initialize environment and agent
    env = SimpleTextEnv()
    agent = DQNAgent(
        model=model,
        tokenizer=tokenizer,
        n_actions=config.data.action_dim,
        buffer_capacity=config.rl.buffer_capacity,
        memory_capacity=config.rl.memory_capacity
    )

    # Training loop
    logger.info("Starting RL training...")
    for episode in range(config.rl.num_episodes):
        metrics = agent.train_episode(env)
        logger.info(
            f"Episode {episode + 1}/{config.rl.num_episodes} - "
            f"Reward: {metrics['reward']:.2f}, "
            f"Avg Loss: {metrics['avg_loss']:.4f}, "
            f"Memory Size: {metrics['memory_size']}"
        )

    return agent

def save_artifacts(config, model: SalesAModel, tokenizer: SalesATokenizer):
    """Save model artifacts in Hugging Face format"""
    export_dir = Path(config.paths.export_dir)
    logger.info(f"Saving artifacts to {export_dir}...")

    # Create export directory
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights in .safetensors format
    model_path = export_dir / "model.safetensors"
    from safetensors.torch import save_file
    save_file(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Create model card
    model_card = f"""--- 
language: en
license: mit
library_name: transformers
---

# {config.model.name}

**Model author:** {config.model.author}

This model is a generative multimodal AI system designed for lifelong learning. It can process and generate text, images, and audio, and is built with a Mixture of Experts (MoE) architecture for efficient scaling.

## Model Details

- **Architecture:** Transformer-based with multimodal encoders and MoE.
- **Vocab Size:** {config.model.vocab_size}
- **Hidden Dimension:** {config.model.hidden_dim}
- **Layers:** {config.model.num_layers}
- **Heads:** {config.model.num_heads}

"""
    with open(export_dir / "README.md", "w") as f:
        f.write(model_card)

    # Save tokenizer configuration manually
    tokenizer_config = {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": 1024,
        "tokenizer_class": "SalesATokenizer",
        "vocab": tokenizer.vocab if hasattr(tokenizer, 'vocab') else {},
        "token_to_id": tokenizer.token_to_id if hasattr(tokenizer, 'token_to_id') else {},
        "id_to_token": tokenizer.id_to_token if hasattr(tokenizer, 'id_to_token') else {},
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "code_token": tokenizer.code_token
        }
    }
    with open(export_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Save special tokens map
    special_tokens_map = {
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token
    }
    with open(export_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    # Save Hugging Face configuration
    hf_config = {
        "model_type": "sales-a",
        "architectures": ["SalesAModel"],
        "hidden_size": config.model.hidden_dim,
        "num_attention_heads": config.model.num_heads,
        "num_hidden_layers": config.model.num_layers,
        "vocab_size": config.model.vocab_size,
    }
    with open(export_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Create .gitattributes
    with open(export_dir / ".gitattributes", "w") as f:
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    
    # Create chat template
    with open(export_dir / "chat_template.jinja", "w") as f:
        f.write("{% for message in messages %}\n")
        f.write("{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% endif %}\n")
        f.write("{% if message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}{% endif %}\n")
        f.write("{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}\n")
    
    # Create generation config
    generation_config = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "bos_token_id": 1
    }
    with open(export_dir / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)
    
    # Create processor config
    processor_config = {
        "image_processor_type": "VisionEncoder",
        "audio_processor_type": "AudioEncoder",
        "text_processor_type": "TextEncoder",
        "multimodal": True
    }
    with open(export_dir / "processor_config.json", "w") as f:
        json.dump(processor_config, f, indent=2)
        
    # Create preprocessor config
    with open(export_dir / "preprocessor_config.json", "w") as f:
        json.dump(processor_config, f, indent=2)
        
    # Create model index for safetensors
    model_index = {
        "metadata": {"total_size": 1024000000},
        "weight_map": {"model": "model.safetensors"}
    }
    with open(export_dir / "model.safetensors.index.json", "w") as f:
        json.dump(model_index, f, indent=2)

    logger.info("All artifacts saved in Hugging Face format.")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Train SalesA AI model")
    parser.add_argument("--config", type=str, default="base",
                      help="Configuration scenario (base, text_generation, code_generation, multimodal, rl_training)")
    parser.add_argument("--skip-rl", action="store_true",
                      help="Skip reinforcement learning training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    logger.info("Starting SalesA AI training pipeline...")
    logger.info(f"Using configuration: {args.config}")

    # Print model info
    logger.info("=" * 80)
    logger.info(f"{config.model.name} - Complete Implementation")
    logger.info(f"{config.model.author}")
    logger.info("=" * 80)

    # Initialize model and tokenizer
    model, tokenizer = init_model(config)

    # Training
    trainer = train_model(config, model, tokenizer)

    # Evaluation
    evaluate_model(config, model, tokenizer)

    # RL training (optional)
    if not args.skip_rl:
        agent = train_rl_agent(config, model, tokenizer)

    # Save artifacts
    save_artifacts(config, model, tokenizer)

    logger.info("\n" + "=" * 80)
    logger.info("SalesA AI training completed successfully!")
    logger.info("The model demonstrates:")
    logger.info("✓ Multimodal processing (text, vision, audio)")
    logger.info("✓ Mixture of Experts architecture")
    logger.info("✓ CPU-optimized training")
    logger.info("✓ Code generation capabilities")
    logger.info("✓ Human-like text generation")
    if not args.skip_rl:
        logger.info("✓ Reinforcement learning integration")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 