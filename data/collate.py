import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union
import logging

from config import SalesAConfig
from tokenizer import SalesATokenizer
# Remove the direct import to avoid circular import
# from data.dataset import MultimodalDataset

logger = logging.getLogger(__name__)

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for multimodal batches"""
    # Initialize batch tensors
    batch_dict = {
        "text": [],
        "image": [],
        "audio": [],
        "labels": [],
        "task_type": [],
        "attention_mask": [],
        "action_labels": []
    }

    # Get max sequence length in batch
    max_len = max(
        [len(item["text"]) if item.get("text") is not None and hasattr(item["text"], '__len__') else (len(str(item["text"])) if item.get("text") is not None else 0) for item in batch]
    )
    max_len = max(max_len, 1)  # Ensure at least length 1

    for item in batch:
        # Process text and create attention mask
        if item.get("text") is not None:
            text = item["text"]
            
            # Handle string text data (tokenize if needed)
            if isinstance(text, str):
                # Tokenize string text (simple tokenization)
                text_tokens = [ord(c) % 256 for c in text[:max_len]]
                text = torch.tensor(text_tokens, dtype=torch.long)
            elif not isinstance(text, torch.Tensor):
                # Convert to tensor if not already
                text = torch.tensor(text, dtype=torch.long)
            
            # Store original length for attention mask
            original_len = len(text)
            
            if len(text) < max_len:
                # Pad text
                padding = torch.zeros(max_len - len(text), dtype=text.dtype)
                text = torch.cat([text, padding])
            elif len(text) > max_len:
                # Truncate if somehow longer
                text = text[:max_len]
                original_len = max_len
            
            # Create attention mask based on actual content length
            attention_mask = torch.zeros(max_len)
            attention_mask[:original_len] = 1
            batch_dict["text"].append(text)
            batch_dict["attention_mask"].append(attention_mask)
        else:
            # Add zero tensor for missing text
            batch_dict["text"].append(torch.zeros(max_len, dtype=torch.long))
            batch_dict["attention_mask"].append(torch.zeros(max_len))

        # Process image
        if item.get("image") is not None:
            batch_dict["image"].append(item["image"])

        # Process audio
        if item.get("audio") is not None:
            batch_dict["audio"].append(item["audio"])

        # Process labels
        if item.get("labels") is not None:
            # Ensure labels are tensors and handle different types
            if isinstance(item["labels"], str):
                # For string labels, we'll create a sequence of token IDs
                label_tokens = [ord(c) % 256 for c in item["labels"][:max_len]]
                labels = torch.tensor(label_tokens, dtype=torch.long)
            elif not isinstance(item["labels"], torch.Tensor):
                labels = torch.tensor(item["labels"], dtype=torch.long)
            else:
                labels = item["labels"]
            
            # Pad or truncate labels to match max_len
            if len(labels) < max_len:
                padding = torch.full((max_len - len(labels),), -100, dtype=torch.long)
                labels = torch.cat([labels, padding])
            elif len(labels) > max_len:
                labels = labels[:max_len]
            
            batch_dict["labels"].append(labels)
        else:
            batch_dict["labels"].append(torch.full((max_len,), -100, dtype=torch.long))

        # Process action labels
        if item.get("action_labels") is not None:
            batch_dict["action_labels"].append(item["action_labels"])

        # Add task type
        batch_dict["task_type"].append(item.get("task_type", "text"))

    # Stack tensors
    if batch_dict["text"]:
        # Ensure all tensors in the batch have the same size
        max_text_len = max([t.shape[0] for t in batch_dict["text"]])
        padded_text = []
        for t in batch_dict["text"]:
            if t.shape[0] < max_text_len:
                padding = torch.zeros(max_text_len - t.shape[0], dtype=t.dtype)
                padded_text.append(torch.cat([t, padding]))
            else:
                padded_text.append(t)
        batch_dict["text"] = torch.stack(padded_text)
        batch_dict["attention_mask"] = torch.stack(batch_dict["attention_mask"])
    else:
        del batch_dict["text"]
        del batch_dict["attention_mask"]

    if batch_dict["image"]:
        batch_dict["image"] = torch.stack(batch_dict["image"])
    else:
        del batch_dict["image"]

    if batch_dict["audio"]:
        batch_dict["audio"] = torch.stack(batch_dict["audio"])
    else:
        del batch_dict["audio"]

    if batch_dict["labels"]:
        try:
            batch_dict["labels"] = torch.stack(batch_dict["labels"])
        except:
            # Handle case where labels are different shapes
            max_label_len = max(label.shape[0] for label in batch_dict["labels"])
            padded_labels = []
            for label in batch_dict["labels"]:
                if len(label) < max_label_len:
                    padding = torch.zeros(max_label_len - len(label), dtype=label.dtype)
                    padded_labels.append(torch.cat([label, padding]))
                else:
                    padded_labels.append(label[:max_label_len])
            batch_dict["labels"] = torch.stack(padded_labels)
    else:
        del batch_dict["labels"]

    if batch_dict["action_labels"]:
        batch_dict["action_labels"] = torch.stack(batch_dict["action_labels"])
    else:
        del batch_dict["action_labels"]

    return batch_dict

def create_multimodal_dataloaders(
    config: SalesAConfig,
    tokenizer: SalesATokenizer,
    batch_size: int = 8,
    dataset_name: Union[str, List[str]] = "auto",
    task_type: Optional[str] = None
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Lazy import to avoid circular import
    from data.dataset import MultimodalDataset

    # Create datasets
    train_dataset = MultimodalDataset(
        config=config,
        tokenizer=tokenizer,
        split="train",
        dataset_name=dataset_name,
        task_type=task_type
    )

    val_dataset = MultimodalDataset(
        config=config,
        tokenizer=tokenizer,
        split="validation",
        dataset_name=dataset_name,
        task_type=task_type
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Safer default
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Safer default
        pin_memory=True
    )

    return train_loader, val_loader 