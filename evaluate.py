import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
import logging
import re
from collections import defaultdict

from model.salesa_model import SalesAModel
from tokenizer import SalesATokenizer

logger = logging.getLogger(__name__)

class SalesAEvaluator:
    """Evaluator class for SalesA AI model"""
    def __init__(self, model: SalesAModel, tokenizer: SalesATokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def evaluate_text_generation(self, prompts: List[str], max_length: int = 100,
                               temperature: float = 0.7, top_k: int = 50) -> Dict:
        """Evaluate text generation capabilities"""
        self.model.eval()
        results = {
            "generations": [],
            "avg_length": 0.0,
            "fluency_score": 0.0
        }

        total_length = 0
        total_fluency = 0.0

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
                attention_mask = torch.ones_like(input_ids)

                # Generate
                generated = []
                for _ in range(max_length):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_type="text"
                    )
                    next_token_logits = outputs["logits"][0, -1, :] / temperature
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    generated.append(next_token.item())
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.device)], dim=1)

                # Decode and compute metrics
                generated_text = self.tokenizer.decode(generated)
                results["generations"].append(generated_text)
                total_length += len(generated)
                
                # Simple fluency score based on repetition and coherence
                fluency = self._compute_fluency_score(generated_text)
                total_fluency += fluency

        # Compute averages
        num_samples = len(prompts)
        results["avg_length"] = total_length / num_samples if num_samples > 0 else 0
        results["fluency_score"] = total_fluency / num_samples if num_samples > 0 else 0

        return results

    def evaluate_code_generation(self, prompts: List[str], max_length: int = 200) -> Dict:
        """Evaluate code generation capabilities"""
        self.model.eval()
        results = {
            "generations": [],
            "syntax_accuracy": 0.0,
            "completion_rate": 0.0
        }

        total_syntax_valid = 0
        total_complete = 0

        with torch.no_grad():
            for prompt in prompts:
                # Add code generation context
                prompt = f"{self.tokenizer.code_token} {prompt}"
                input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
                attention_mask = torch.ones_like(input_ids)

                # Generate
                generated = []
                for _ in range(max_length):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_type="text"
                    )
                    next_token_logits = outputs["logits"][0, -1, :]
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    generated.append(next_token.item())
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.device)], dim=1)

                # Decode and analyze code
                generated_code = self.tokenizer.decode(generated)
                results["generations"].append(generated_code)

                # Check syntax and completion
                is_valid = self._check_code_syntax(generated_code)
                is_complete = self._check_code_completion(generated_code)

                if is_valid:
                    total_syntax_valid += 1
                if is_complete:
                    total_complete += 1

        # Compute metrics
        num_samples = len(prompts)
        if num_samples > 0:
            results["syntax_accuracy"] = total_syntax_valid / num_samples
            results["completion_rate"] = total_complete / num_samples

        return results

    def analyze_expert_usage(self) -> List[Dict]:
        """Analyze MoE expert usage patterns"""
        expert_stats = []
        
        # Collect stats for each MoE layer
        for name, module in self.model.named_modules():
            if hasattr(module, 'router') and hasattr(module, 'experts'):
                layer_stats = {
                    "layer_name": name,
                    "num_experts": len(module.experts),
                    "usage_counts": defaultdict(int),
                    "load_balance": 0.0
                }

                # Get router probabilities for a dummy batch
                dummy_input = torch.randn(4, 10, module.hidden_dim).to(self.device)
                with torch.no_grad():
                    router_probs, _ = module.router(dummy_input)

                # Analyze routing decisions
                top_k_probs, top_k_indices = torch.topk(router_probs, k=module.top_k, dim=-1)
                for i in range(module.top_k):
                    for expert_idx in top_k_indices[:, :, i].flatten():
                        layer_stats["usage_counts"][expert_idx.item()] += 1

                # Compute load balancing metric
                total_calls = sum(layer_stats["usage_counts"].values())
                expected_calls = total_calls / module.num_experts
                max_deviation = max(abs(count - expected_calls) 
                                 for count in layer_stats["usage_counts"].values())
                layer_stats["load_balance"] = 1.0 - (max_deviation / total_calls)

                expert_stats.append(layer_stats)

        return expert_stats

    def _compute_fluency_score(self, text: str) -> float:
        """Compute a simple fluency score based on repetition and coherence"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Penalize repetitive phrases
        words = text.split()
        if not words:
            return 0.0

        # Check for repeated phrases (bigrams and trigrams)
        bigrams = list(zip(words[:-1], words[1:]))
        trigrams = list(zip(words[:-2], words[1:-1], words[2:])) if len(words) > 2 else []
        
        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        unique_trigrams = len(set(trigrams))
        total_trigrams = len(trigrams)
        
        # Compute diversity ratios
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
        trigram_diversity = unique_trigrams / total_trigrams if total_trigrams > 0 else 0
        
        # Simple sentence structure check
        sentences = re.split('[.!?]+', text)
        valid_sentences = sum(1 for s in sentences if len(s.split()) > 2)
        sentence_ratio = valid_sentences / len(sentences) if sentences else 0
        
        # Combine metrics
        fluency_score = (0.4 * bigram_diversity + 
                        0.4 * trigram_diversity + 
                        0.2 * sentence_ratio)
        
        return fluency_score

    def _check_code_syntax(self, code: str) -> bool:
        """Check if generated code has valid syntax"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _check_code_completion(self, code: str) -> bool:
        """Check if generated code appears complete"""
        # Simple heuristics for code completion
        if not code.strip():
            return False
        
        # Check balanced brackets/parentheses
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return False
        
        # Stack should be empty if all brackets are balanced
        if stack:
            return False
            
        # Check if code ends with common closing patterns
        common_endings = ['}', ')', ']', 'return', 'break', 'continue', 'pass']
        return any(code.strip().endswith(ending) for ending in common_endings) 