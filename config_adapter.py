# Config Adapter for SalesA AI
# This module converts the complex nested config to the simple config format

from utils.config import SalesAConfig as ComplexSalesAConfig
from config import SalesAConfig as SimpleSalesAConfig

def adapt_config(complex_config: ComplexSalesAConfig) -> SimpleSalesAConfig:
    """
    Convert complex nested config to simple config format expected by SalesAModel
    
    Args:
        complex_config: The complex nested config from utils.config
        
    Returns:
        SimpleSalesAConfig: The simple config format expected by the model
    """
    
    # Extract values from the complex config
    model_config = complex_config.model
    training_config = complex_config.training
    data_config = complex_config.data
    
    # Create simple config with extracted values
    simple_config = SimpleSalesAConfig(
        # Model architecture
        vocab_size=model_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        intermediate_dim=model_config.intermediate_dim,
        max_seq_len=model_config.max_seq_len,
        
        # MoE configuration
        num_experts=model_config.num_experts,
        expert_capacity=model_config.expert_capacity,
        top_k=model_config.top_k,
        
        # Multimodal dimensions
        vision_dim=model_config.vision_dim,
        audio_dim=model_config.audio_dim,
        vision_patch_size=model_config.vision_patch_size,
        audio_patch_size=model_config.audio_patch_size,
        
        # Training parameters
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        dropout_rate=training_config.dropout_rate,
        num_epochs=training_config.num_epochs,
        early_stopping_patience=training_config.early_stopping_patience,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        gradient_clip_norm=training_config.gradient_clip_norm,
        warmup_steps=training_config.warmup_steps,
        
        # Optimization
        use_mixed_precision=training_config.use_mixed_precision,
        gradient_checkpointing=training_config.gradient_checkpointing,
        
        # Data parameters
        max_text_length=data_config.max_text_length,
        max_audio_length=data_config.max_audio_length,
        action_dim=data_config.action_dim,
        
        # Model metadata
        model_name=model_config.name,
        model_author=model_config.author
    )
    
    return simple_config

def create_simple_config_from_yaml(scenario: str = "base") -> SimpleSalesAConfig:
    """
    Load complex config from YAML and convert to simple config
    
    Args:
        scenario: Configuration scenario name
        
    Returns:
        SimpleSalesAConfig: Simple config ready for model initialization
    """
    from utils.config import load_config
    
    # Load complex config
    complex_config = load_config(scenario)
    
    # Convert to simple config
    simple_config = adapt_config(complex_config)
    
    return simple_config