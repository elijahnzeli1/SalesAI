from typing import Optional, List, Union
import tiktoken
import torch

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
            if tid in self.id_to_token and self.id_to_token[tid].decode("utf-8") in EXTRA_TOKENS:
                words.append(self.id_to_token[tid].decode("utf-8"))
            else:
                try:
                    words.append(self.enc.decode([tid]))
                except Exception:
                    words.append("<UNK>")
        return " ".join(words) 