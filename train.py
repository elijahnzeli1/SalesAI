import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional
from tqdm import tqdm
import math

from config import SalesAConfig
from model.salesa_model import SalesAModel
from tokenizer import SalesATokenizer

logger = logging.getLogger(__name__)

class CosineAnnealingWarmupRestarts:
    """Cosine annealing scheduler with warmup and restarts"""
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=10., min_lr=0.001, warmup_steps=0, gamma=1., last_epoch=-1):
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.last_epoch = last_epoch
        self.step_count = 0  # Track actual step count
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        if epoch is None:
            self.step_count += 1
            epoch = self.step_count
        else:
            self.step_count = epoch
        self.last_epoch = epoch

        for param_group in self.optimizer.param_groups:
            if epoch < self.warmup_steps:
                # Ensure we start with a small but non-zero learning rate
                if epoch == 0:
                    lr = self.max_lr * 0.01  # Start with 1% of max LR
                else:
                    lr = self.max_lr * (epoch / self.warmup_steps)
            else:
                effective_epoch = epoch - self.warmup_steps
                effective_cycle_steps = self.cur_cycle_steps - self.warmup_steps
                lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * effective_epoch / effective_cycle_steps)) / 2
            param_group['lr'] = lr

        if epoch >= self.cur_cycle_steps:
            self.cycle += 1
            self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps

class SalesATrainer:
    """Enhanced trainer class for SalesA AI model with advanced features"""
    def __init__(self, config: SalesAConfig, model: Optional[SalesAModel] = None):
        self.config = config
        self.model = model if model is not None else SalesAModel(config)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with proper initialization
        self.scheduler = None  # Will be initialized when training starts
        self.total_steps = 0  # Will be set during training
        
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with enhanced features"""
        total_loss = 0.0
        total_load_balance_loss = 0.0
        self.model.train()
        batch_loss = 0.0
        batch_load_balance_loss = 0.0
        valid_samples_in_batch = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch, return_loss=True)
            loss = outputs["loss"]
            load_balance_loss = outputs.get("load_balance_loss", 0.0)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            if load_balance_loss != 0:
                load_balance_loss = load_balance_loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            batch_loss += loss.item() * self.gradient_accumulation_steps
            if load_balance_loss != 0:
                batch_load_balance_loss += load_balance_loss.item() * self.gradient_accumulation_steps

            # Update on gradient accumulation steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config.get('gradient_clip_norm', 1.0))
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update learning rate
                self.scheduler.step()
                self.global_step += 1

                # Update metrics
                total_loss += batch_loss
                total_load_balance_loss += batch_load_balance_loss
                valid_samples_in_batch += 1

                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': batch_loss / self.gradient_accumulation_steps,
                    'load_balance': batch_load_balance_loss / self.gradient_accumulation_steps,
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })

                # Reset batch metrics
                batch_loss = 0.0
                batch_load_balance_loss = 0.0

        # Calculate average losses
        avg_loss = total_loss / valid_samples_in_batch if valid_samples_in_batch > 0 else float('inf')
        avg_load_balance = total_load_balance_loss / valid_samples_in_batch if valid_samples_in_batch > 0 else 0.0

        logger.info(f"Epoch complete - Avg loss: {avg_loss:.4f}, Avg load balance: {avg_load_balance:.4f}")
        return avg_loss

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model with enhanced metrics"""
        self.model.eval()
        total_loss = 0.0
        total_load_balance_loss = 0.0
        valid_samples = 0
        
        # Additional metrics
        total_accuracy = 0.0
        total_perplexity = 0.0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch, return_loss=True)
                loss = outputs["loss"]
                load_balance_loss = outputs.get("load_balance_loss", 0.0)
                logits = outputs["logits"]

                # Calculate additional metrics
                if "labels" in batch and batch["labels"] is not None:
                    labels = batch["labels"]
                    if labels.dim() == 1:  # Classification task
                        pred = torch.argmax(logits, dim=-1)
                        accuracy = (pred == labels).float().mean().item()
                        total_accuracy += accuracy
                    else:  # Language modeling task
                        # Calculate perplexity
                        log_probs = torch.log_softmax(logits, dim=-1)
                        target_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
                        perplexity = torch.exp(-target_log_probs.mean()).item()
                        total_perplexity += perplexity

                # Update metrics
                total_loss += loss.item()
                total_load_balance_loss += load_balance_loss if load_balance_loss != 0 else 0.0
                valid_samples += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': loss.item(),
                    'load_balance': load_balance_loss if load_balance_loss != 0 else 0.0
                })

        # Calculate average metrics
        avg_loss = total_loss / valid_samples if valid_samples > 0 else float('inf')
        avg_load_balance = total_load_balance_loss / valid_samples if valid_samples > 0 else 0.0
        avg_accuracy = total_accuracy / valid_samples if valid_samples > 0 else 0.0
        avg_perplexity = total_perplexity / valid_samples if valid_samples > 0 else float('inf')

        metrics = {
            "val_loss": avg_loss,
            "val_load_balance_loss": avg_load_balance,
            "val_accuracy": avg_accuracy,
            "val_perplexity": avg_perplexity
        }

        logger.info(f"Validation - Loss: {avg_loss:.4f}, Load balance: {avg_load_balance:.4f}, "
                   f"Accuracy: {avg_accuracy:.4f}, Perplexity: {avg_perplexity:.4f}")
        return metrics

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
             num_epochs: int = 10, early_stopping_patience: int = 3):
        """Full training loop with validation and early stopping"""
        # Initialize scheduler with proper total steps
        if self.scheduler is None:
            self.total_steps = num_epochs * len(train_loader)
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=self.total_steps // 4,
                cycle_mult=1.0,
                max_lr=self.config.learning_rate,
                min_lr=self.config.learning_rate * 0.01,
                warmup_steps=self.config.get('warmup_steps', 1000),
                gamma=0.5
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping_patience', early_stopping_patience)

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Training
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training loss: {train_loss:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["val_loss"]
                logger.info(f"Validation loss: {val_loss:.4f}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model("best_model.pt")
                    logger.info("New best model saved!")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}.pt")

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

    def save_model(self, path: str):
        """Save model checkpoint with enhanced metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint with enhanced metadata"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Model loaded from {path}")
        logger.info(f"Resuming from step {self.global_step} with best val loss {self.best_val_loss}") 