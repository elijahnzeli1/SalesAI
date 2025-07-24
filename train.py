import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional
from tqdm import tqdm

from config import SalesAConfig
from model.salesa_model import SalesAModel
from tokenizer import SalesATokenizer

logger = logging.getLogger(__name__)

class SalesATrainer:
    """Trainer class for SalesA AI model"""
    def __init__(self, config: SalesAConfig):
        self.config = config
        self.model = SalesAModel(config)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.gradient_accumulation_steps = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update metrics
                total_loss += batch_loss
                total_load_balance_loss += batch_load_balance_loss
                valid_samples_in_batch += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': batch_loss / self.gradient_accumulation_steps,
                    'load_balance': batch_load_balance_loss / self.gradient_accumulation_steps
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
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        total_load_balance_loss = 0.0
        valid_samples = 0

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

                # Update metrics
                total_loss += loss.item()
                total_load_balance_loss += load_balance_loss if load_balance_loss != 0 else 0.0
                valid_samples += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': loss.item(),
                    'load_balance': load_balance_loss if load_balance_loss != 0 else 0.0
                })

        # Calculate average losses
        avg_loss = total_loss / valid_samples if valid_samples > 0 else float('inf')
        avg_load_balance = total_load_balance_loss / valid_samples if valid_samples > 0 else 0.0

        metrics = {
            "val_loss": avg_loss,
            "val_load_balance_loss": avg_load_balance
        }

        logger.info(f"Validation - Loss: {avg_loss:.4f}, Load balance: {avg_load_balance:.4f}")
        return metrics

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
             num_epochs: int = 10, early_stopping_patience: int = 3):
        """Full training loop with validation and early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

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
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

        logger.info("Training completed!")

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}") 