import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            raise

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries"""
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

@dataclass
class ModelConfig:
    name: str = "SalesA AI"
    author: str = "Created by N.E.N (Nthuku Elijah Nzeli) and SalesA Team"
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_dim: int = 1024
    max_seq_len: int = 2048
    num_experts: int = 4
    expert_capacity: int = 2
    top_k: int = 2
    vision_dim: int = 224
    audio_dim: int = 80
    vision_patch_size: int = 16
    audio_patch_size: int = 4

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    num_epochs: int = 10
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = True
    warmup_steps: int = 0
    cosine_schedule: bool = False
    label_smoothing: float = 0.0

@dataclass
class DataConfig:
    max_text_length: int = 1024
    max_audio_length: int = 16000
    action_dim: int = 10
    num_workers: int = 0
    pin_memory: bool = True
    dataset_name: str = "auto"
    task_type: str = "text"
    vision_augment: bool = False
    audio_augment: bool = False
    text_augment: bool = False
    text_dropout: float = 0.0
    vision_dropout: float = 0.0
    audio_dropout: float = 0.0

@dataclass
class RLConfig:
    num_episodes: int = 10
    buffer_capacity: int = 500
    memory_capacity: int = 100
    gamma: float = 0.99
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    epsilon: float = 0.1
    curiosity_bonus: float = 0.1
    update_target_every: int = 5
    ppo_epochs: int = 4
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    double_dqn: bool = False
    dueling_dqn: bool = False
    prioritized_replay: bool = False
    n_step_returns: int = 1
    env_type: str = "text"
    max_steps_per_episode: int = 100
    reward_scale: float = 1.0
    use_intrinsic_reward: bool = True

@dataclass
class PathConfig:
    export_dir: str = "./SalesA"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

@dataclass
class SalesAConfig:
    """Main configuration class for SalesA AI"""
    model_name: str = "SalesA AI"  # <-- Add this line
    model_author: str = "Created by N.E.N (Nthuku Elijah Nzeli) and SalesA Team"
    vocab_size: int = 32000
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, config_path: str, base_path: Optional[str] = None) -> 'SalesAConfig':
        """Load configuration from YAML file with optional base configuration"""
        config_dir = os.path.dirname(config_path)
        config = load_yaml(config_path)
        
        # Handle base configuration
        if 'extends' in config:
            base_file = config.pop('extends')
            if base_path:
                base_file = os.path.join(base_path, base_file)
            else:
                base_file = os.path.join(config_dir, base_file)
            base_config = load_yaml(base_file)
            config = merge_configs(base_config, config)
        
        # Create nested dataclass instances
        model_config = ModelConfig(**config.get('model', {}))
        training_config = TrainingConfig(**config.get('training', {}))
        data_config = DataConfig(**config.get('data', {}))
        rl_config = RLConfig(**config.get('rl', {}))
        paths_config = PathConfig(**config.get('paths', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            rl=rl_config,
            paths=paths_config
        )

    def save(self, file_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'rl': self.rl.__dict__,
            'paths': self.paths.__dict__
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

def load_config(scenario: str = "base") -> SalesAConfig:
    """Load configuration for a specific training scenario"""
    config_dir = Path(__file__).parent.parent / "configs"
    config_file = config_dir / f"{scenario}.yaml"
    
    if not config_file.exists():
        logger.warning(f"Configuration file {config_file} not found, using base config")
        config_file = config_dir / "base.yaml"
        
    return SalesAConfig.from_yaml(str(config_file))