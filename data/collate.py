import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union
import logging

from config import SalesAConfig
from tokenizer import SalesATokenizer
from data.dataset import MultimodalDataset

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
        [len(item["text"]) if item.get("text") is not None else 0 for item in batch]
    )

    for item in batch:
        # Process text and create attention mask
        if item.get("text") is not None:
            text = item["text"]
            attention_mask = torch.ones(max_len)
            if len(text) < max_len:
                # Pad text
                padding = torch.zeros(max_len - len(text), dtype=text.dtype)
                text = torch.cat([text, padding])
                # Update attention mask
                attention_mask[len(item["text"]):] = 0
            elif len(text) > max_len:
                # Truncate if somehow longer
                text = text[:max_len]
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
            if isinstance(item["labels"], torch.Tensor):
                if len(item["labels"].shape) == 0:
                    # Single label
                    batch_dict["labels"].append(item["labels"].unsqueeze(0))
                else:
                    # Sequence labels
                    labels = item["labels"]
                    if len(labels) < max_len:
                        padding = torch.zeros(max_len - len(labels), dtype=labels.dtype)
                        labels = torch.cat([labels, padding])
                    elif len(labels) > max_len:
                        labels = labels[:max_len]
                    batch_dict["labels"].append(labels)
            else:
                # Convert to tensor if not already
                batch_dict["labels"].append(torch.tensor(item["labels"]))

        # Process action labels
        if item.get("action_labels") is not None:
            batch_dict["action_labels"].append(item["action_labels"])

        # Add task type
        batch_dict["task_type"].append(item.get("task_type", "text"))

    # Stack tensors
    if batch_dict["text"]:
        batch_dict["text"] = torch.stack(batch_dict["text"])
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