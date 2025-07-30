from typing import Optional, List, Union
import tiktoken
import torch
import json
import os
from pathlib import Path

# Domain-specific tokens
EXTRA_TOKENS = [
    "<STOCK>", "<FINANCE>", "<MARKET>", "<SALES>", "<EARNINGS>", "<REVENUE>", "<PRICE>", "<VOLUME>",
    "<ACTION>", "<AUDIO>", "<VISION>", "<TTS>", "<ROBOT>", "<INSTRUCTION>", "<RESPONSE>", "<USER>", "<ASSISTANT>"
]

def build_vocab_with_tiktoken(dataset, vocab_size=32000, model_name="gpt2"):
    """
    Build a vocabulary from the dataset using Tiktoken's BPE model.
    Returns (vocab, enc)
    """
    enc = tiktoken.get_encoding(model_name)
    counter = {}
    for sample in dataset:
        text = sample.get("text")
        if text is not None:
            if isinstance(text, torch.Tensor):
                text_str = " ".join([str(t.item()) for t in text])
            elif isinstance(text, str):
                text_str = text
            else:
                continue
            tokens = enc.encode(text_str)
            for token in tokens:
                counter[token] = counter.get(token, 0) + 1
    # Get most common tokens up to vocab_size minus extra tokens
    most_common = sorted(counter.items(), key=lambda x: -x[1])[:max(0, vocab_size - len(EXTRA_TOKENS))]
    vocab = EXTRA_TOKENS + [token for token, _ in most_common]
    return vocab[:vocab_size], enc

class SalesATokenizer:
    """Tokenizer for SalesA AI using Tiktoken and extra domain tokens"""
    def __init__(self, vocab_size: int = 32000, vocab: Optional[Union[list, dict]] = None, enc=None, model_name: str = "gpt2"):
        self.vocab_size = vocab_size
        self.model_name = model_name
        if enc is not None:
            self.enc = enc
        else:
            self.enc = tiktoken.get_encoding(model_name)
        if vocab is not None:
            # Handle both list and dict vocab formats
            if isinstance(vocab, dict):
                # Convert dict to list format
                vocab_list = list(vocab.keys())
                self.vocab = vocab_list[:vocab_size]
            elif isinstance(vocab, list):
                # Handle list format
                self.vocab = vocab[:vocab_size]
            else:
                # Handle other formats by converting to list
                try:
                    vocab_list = list(vocab)
                    self.vocab = vocab_list[:vocab_size]
                except Exception:
                    # Fallback to default vocab
                    self.vocab = list(self.enc._mergeable_ranks.keys())[:vocab_size]
        else:
            self.vocab = list(self.enc._mergeable_ranks.keys())[:vocab_size]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        # Special tokens
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.code_token = "<|code|>"
        # Add extra tokens to token_to_id if not present
        for token in EXTRA_TOKENS:
            token_bytes = token.encode("utf-8") if isinstance(token, str) else token
            if token_bytes not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token_bytes] = idx
                self.id_to_token[idx] = token_bytes
        # Set special token IDs safely
        try:
            self.pad_token_id = self.enc.encode(self.pad_token)[0]
        except Exception:
            self.pad_token_id = 0
        try:
            self.unk_token_id = self.enc.encode(self.unk_token)[0]
        except Exception:
            self.unk_token_id = 1
        try:
            self.bos_token_id = self.enc.encode(self.bos_token)[0]
        except Exception:
            self.bos_token_id = 2
        try:
            self.eos_token_id = self.enc.encode(self.eos_token)[0]
        except Exception:
            self.eos_token_id = 3
        try:
            self.code_token_id = self.enc.encode(self.code_token)[0]
        except Exception:
            self.code_token_id = 4
        # Add extra token IDs
        for token in EXTRA_TOKENS:
            token_bytes = token.encode("utf-8") if isinstance(token, str) else token
            setattr(self, f"{token.strip('<>').lower()}_id", self.token_to_id[token_bytes])

    def encode(self, text: str) -> list:
        """Use Tiktoken for base encoding, but map extra tokens to their IDs"""
        tokens = []
        for word in text.split():
            word_bytes = word.encode("utf-8")
            if word_bytes in self.token_to_id:
                tokens.append(self.token_to_id[word_bytes])
            else:
                tokens.extend(self.enc.encode(word))
        return tokens

    def decode(self, token_ids: list) -> str:
        """Map extra token IDs back to their string, otherwise use Tiktoken"""
        words = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                # Handle both bytes and string tokens
                if isinstance(token, bytes):
                    token_str = token.decode("utf-8", errors='replace')
                else:
                    token_str = str(token)
                
                # Check if it's an extra token
                if token_str in EXTRA_TOKENS:
                    words.append(token_str)
                else:
                    words.append(token_str)
            else:
                try:
                    words.append(self.enc.decode([tid]))
                except Exception:
                    words.append("<UNK>")
        return " ".join(words)

    def save_vocab_files(self, save_directory: str):
        """Save vocabulary and merge files in Hugging Face format"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save vocab.json (token to id mapping)
        vocab_dict = {}
        for token in self.vocab:
            if isinstance(token, bytes):
                token_str = token.decode('utf-8', errors='replace')
            else:
                token_str = str(token)
            vocab_dict[token_str] = self.token_to_id[token]
        
        with open(save_directory / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        # Save merges.txt (BPE merge rules)
        merges = []
        if hasattr(self.enc, '_mergeable_ranks'):
            # Sort merges by rank (lower rank = higher priority)
            sorted_merges = sorted(self.enc._mergeable_ranks.items(), key=lambda x: x[1])
            for merge_pair, rank in sorted_merges:
                if isinstance(merge_pair, bytes):
                    # Split the merge pair into two parts
                    # This is a simplified approach - actual BPE merges are more complex
                    merge_str = merge_pair.decode('utf-8', errors='replace')
                    if len(merge_str) >= 2:
                        merges.append(f"{merge_str[0]} {merge_str[1:]}")
                    else:
                        merges.append(merge_str)
        
        with open(save_directory / "merges.txt", "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for merge in merges:
                f.write(f"{merge}\n")
        
        # Save tokenizer.json (complete tokenizer configuration)
        tokenizer_config = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": self.pad_token_id, "special": True, "content": self.pad_token},
                {"id": self.unk_token_id, "special": True, "content": self.unk_token},
                {"id": self.bos_token_id, "special": True, "content": self.bos_token},
                {"id": self.eos_token_id, "special": True, "content": self.eos_token},
                {"id": self.code_token_id, "special": True, "content": self.code_token}
            ],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "vocab": vocab_dict,
                "merges": merges,
                "cache_capacity": 1000,
                "dropout": None,
                "unk_token": self.unk_token,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False
            }
        }
        
        with open(save_directory / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        # Save special tokens map
        special_tokens_map = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "additional_special_tokens": [self.code_token] + EXTRA_TOKENS
        }
        
        with open(save_directory / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
        
        # Save tokenizer configuration
        tokenizer_config_simple = {
            "vocab_size": self.vocab_size,
            "model_max_length": 2048,
            "tokenizer_class": "SalesATokenizer",
            "model_name": self.model_name,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "code_token": self.code_token,
            "extra_tokens": EXTRA_TOKENS
        }
        
        with open(save_directory / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config_simple, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary files saved to {save_directory}")
        print(f"Files created: vocab.json, merges.txt, tokenizer.json, special_tokens_map.json, tokenizer_config.json")

    def save_pretrained(self, save_directory: str):
        """Save the tokenizer in Hugging Face format (alias for save_vocab_files)"""
        self.save_vocab_files(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        """Load tokenizer from saved files"""
        save_directory = Path(save_directory)
        
        # Load vocab.json
        with open(save_directory / "vocab.json", "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        
        # Load tokenizer config
        with open(save_directory / "tokenizer_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Convert vocab dict to list format
        vocab_list = [None] * len(vocab_dict)
        for token, token_id in vocab_dict.items():
            vocab_list[token_id] = token.encode('utf-8') if isinstance(token, str) else token
        
        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=config.get("vocab_size", 32000),
            vocab=vocab_list,
            model_name=config.get("model_name", "gpt2"),
            **kwargs
        )
        
        return tokenizer 