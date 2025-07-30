"""
Enhanced Reinforcement Learning Agent for SalesAI
Intelligent multimodal training with sophisticated learning mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class MultimodalEnvironment(ABC):
    """Abstract multimodal environment for intelligent training"""
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take action and return (observation, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Get current task description"""
        pass

class TextGenerationEnv(MultimodalEnvironment):
    """Intelligent text generation environment"""
    
    def __init__(self, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.current_text = ""
        self.target_text = ""
        self.step_count = 0
        self.max_steps = 50
        
        # Task templates for diverse learning (simplified to avoid KeyError)
        self.task_templates = [
            "Write a creative story about {topic}",
            "Explain {topic} in simple terms",
            "Generate code for {topic}",
            "Write a persuasive argument about {topic}",
            "Create a dialogue about {topic}",
            "Summarize the key points about {topic}",
            "Write a poem about {topic}",
            "Generate a business plan for {topic}"
        ]
        
        self.topics = [
            "artificial intelligence", "space exploration", "climate change",
            "human creativity", "technology ethics", "future of work",
            "scientific discovery", "human relationships", "innovation",
            "sustainability", "education", "healthcare"
        ]
        
        # Add missing attributes
        self.characters = ["AI assistant", "human user", "scientist", "student"]
        self.ideas = ["renewable energy", "digital transformation", "smart cities", "healthcare innovation"]
    
    def reset(self) -> Dict[str, Any]:
        """Reset with new task"""
        self.step_count = 0
        self.current_text = ""
        
        # Generate random task with safe formatting
        template = random.choice(self.task_templates)
        topic = random.choice(self.topics)
        
        # Safe formatting to avoid KeyError
        try:
            self.target_text = template.format(topic=topic)
        except KeyError:
            # Fallback if formatting fails
            self.target_text = f"Write about {topic}"
        
        return {
            "text": self.current_text,
            "target": self.target_text,
            "task_type": "text_generation",
            "step": self.step_count
        }
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take text generation action"""
        self.step_count += 1
        
        # Add action to current text
        if self.current_text:
            self.current_text += " " + action
        else:
            self.current_text = action
        
        # Calculate reward based on multiple factors
        reward = self._calculate_reward(action)
        
        # Check if done
        done = (self.step_count >= self.max_steps or 
                len(self.current_text.split()) >= self.max_length or
                self._is_task_complete())
        
        info = {
            "text_length": len(self.current_text.split()),
            "target_similarity": self._calculate_similarity(),
            "creativity_score": self._calculate_creativity(),
            "coherence_score": self._calculate_coherence()
        }
        
        return {
            "text": self.current_text,
            "target": self.target_text,
            "task_type": "text_generation",
            "step": self.step_count
        }, reward, done, info
    
    def _calculate_reward(self, action: str) -> float:
        """Calculate sophisticated reward"""
        reward = 0.0
        
        # Relevance to target
        similarity = self._calculate_similarity()
        reward += similarity * 2.0
        
        # Creativity bonus
        creativity = self._calculate_creativity()
        reward += creativity * 1.5
        
        # Coherence bonus
        coherence = self._calculate_coherence()
        reward += coherence * 1.0
        
        # Length penalty
        if len(self.current_text.split()) > self.max_length:
            reward -= 0.5
        
        # Repetition penalty
        if self._has_repetition():
            reward -= 0.3
        
        return reward
    
    def _calculate_similarity(self) -> float:
        """Calculate similarity to target"""
        if not self.target_text or not self.current_text:
            return 0.0
        
        # Simple word overlap
        target_words = set(self.target_text.lower().split())
        current_words = set(self.current_text.lower().split())
        
        if not target_words:
            return 0.0
        
        overlap = len(target_words.intersection(current_words))
        return overlap / len(target_words)
    
    def _calculate_creativity(self) -> float:
        """Calculate creativity score"""
        if not self.current_text:
            return 0.0
        
        words = self.current_text.split()
        if len(words) < 3:
            return 0.0
        
        # Diversity of vocabulary
        unique_words = len(set(words))
        total_words = len(words)
        
        # Novelty (words not in common vocabulary)
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        novel_words = len([w for w in words if w.lower() not in common_words])
        
        creativity = (unique_words / total_words) * 0.6 + (novel_words / total_words) * 0.4
        return min(creativity, 1.0)
    
    def _calculate_coherence(self) -> float:
        """Calculate text coherence"""
        if not self.current_text:
            return 0.0
        
        sentences = self.current_text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence based on sentence length consistency
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        # Variance in sentence lengths (lower is better)
        variance = np.var(lengths)
        coherence = max(0, 1 - variance / 100)
        return coherence
    
    def _has_repetition(self) -> bool:
        """Check for repetitive patterns"""
        words = self.current_text.split()
        if len(words) < 6:
            return False
        
        # Check for repeated 3-word sequences
        for i in range(len(words) - 5):
            seq1 = " ".join(words[i:i+3])
            seq2 = " ".join(words[i+3:i+6])
            if seq1 == seq2:
                return True
        
        return False
    
    def _is_task_complete(self) -> bool:
        """Check if task is complete"""
        return len(self.current_text.split()) >= 20 and self._calculate_similarity() > 0.3
    
    def get_task_description(self) -> str:
        return f"Generate text for: {self.target_text}"

class CodeGenerationEnv(MultimodalEnvironment):
    """Intelligent code generation environment"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.current_code = ""
        self.target_functionality = ""
        self.step_count = 0
        self.max_steps = 100
        
        self.code_tasks = [
            "sort a list of numbers",
            "find the maximum value in an array",
            "check if a string is a palindrome",
            "calculate fibonacci numbers",
            "reverse a string",
            "find prime numbers",
            "merge two sorted arrays",
            "implement a simple calculator",
            "validate email format",
            "count word frequency"
        ]
    
    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.current_code = ""
        self.target_functionality = random.choice(self.code_tasks)
        
        return {
            "code": self.current_code,
            "target": self.target_functionality,
            "task_type": "code_generation",
            "step": self.step_count
        }
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # Add code action
        if self.current_code:
            self.current_code += "\n" + action
        else:
            self.current_code = action
        
        # Calculate reward
        reward = self._calculate_code_reward(action)
        
        # Check if done
        done = (self.step_count >= self.max_steps or 
                len(self.current_code.split('\n')) >= 50 or
                self._is_code_complete())
        
        info = {
            "code_length": len(self.current_code.split('\n')),
            "syntax_score": self._check_syntax(),
            "functionality_score": self._check_functionality(),
            "readability_score": self._check_readability()
        }
        
        return {
            "code": self.current_code,
            "target": self.target_functionality,
            "task_type": "code_generation",
            "step": self.step_count
        }, reward, done, info
    
    def _calculate_code_reward(self, action: str) -> float:
        reward = 0.0
        
        # Syntax correctness
        syntax_score = self._check_syntax()
        reward += syntax_score * 2.0
        
        # Functionality relevance
        func_score = self._check_functionality()
        reward += func_score * 3.0
        
        # Readability
        readability = self._check_readability()
        reward += readability * 1.0
        
        # Length penalty
        if len(self.current_code.split('\n')) > 50:
            reward -= 0.5
        
        return reward
    
    def _check_syntax(self) -> float:
        """Check basic syntax (simplified)"""
        if not self.current_code:
            return 0.0
        
        # Simple syntax checks
        lines = self.current_code.split('\n')
        score = 0.0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                score += 0.1
            elif 'def ' in line or 'if ' in line or 'for ' in line or 'while ' in line:
                score += 0.2
            elif '=' in line or 'return ' in line:
                score += 0.15
            else:
                score += 0.05
        
        return min(score / len(lines), 1.0) if lines else 0.0
    
    def _check_functionality(self) -> float:
        """Check if code matches target functionality"""
        if not self.current_code or not self.target_functionality:
            return 0.0
        
        # Simple keyword matching
        keywords = {
            "sort": ["sort", "sorted", "order"],
            "maximum": ["max", "maximum", "largest"],
            "palindrome": ["palindrome", "reverse", "mirror"],
            "fibonacci": ["fibonacci", "fib", "sequence"],
            "reverse": ["reverse", "backward", "flip"],
            "prime": ["prime", "is_prime", "prime_number"],
            "merge": ["merge", "combine", "join"],
            "calculator": ["calc", "calculator", "compute"],
            "email": ["email", "validate", "check_email"],
            "frequency": ["count", "frequency", "occurrence"]
        }
        
        target_keywords = keywords.get(self.target_functionality.split()[0], [])
        code_lower = self.current_code.lower()
        
        matches = sum(1 for keyword in target_keywords if keyword in code_lower)
        return min(matches / len(target_keywords), 1.0) if target_keywords else 0.0
    
    def _check_readability(self) -> float:
        """Check code readability"""
        if not self.current_code:
            return 0.0
        
        lines = self.current_code.split('\n')
        score = 0.0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Indentation
            if line.startswith('    ') or line.startswith('\t'):
                score += 0.1
            
            # Comments
            if '#' in line:
                score += 0.1
            
            # Variable names
            if 'def ' in line and '_' in line:
                score += 0.1
            
            # Line length
            if len(line) <= 80:
                score += 0.05
        
        return min(score / len(lines), 1.0) if lines else 0.0
    
    def _is_code_complete(self) -> bool:
        """Check if code is complete"""
        return (len(self.current_code.split('\n')) >= 10 and 
                self._check_syntax() > 0.5 and 
                self._check_functionality() > 0.3)
    
    def get_task_description(self) -> str:
        return f"Generate code for: {self.target_functionality}"

class EnhancedDQNAgent:
    """Enhanced DQN agent with intelligent learning mechanisms"""
    
    def __init__(self, model, tokenizer, n_actions=100, buffer_capacity=10000, 
                 memory_capacity=1000, learning_rate=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.n_actions = n_actions
        self.device = next(model.parameters()).device
        
        # Experience replay
        self.replay_buffer = deque(maxlen=buffer_capacity)
        self.priorities = deque(maxlen=buffer_capacity)
        
        # Episodic memory for meta-learning
        self.episodic_memory = deque(maxlen=memory_capacity)
        self.task_embeddings = {}
        
        # Networks
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_every = 100
        self.step_count = 0
        
        # Meta-learning parameters
        self.meta_learning_rate = 0.1
        self.task_adaptation_steps = 5
        
        # Curiosity-driven exploration
        self.curiosity_bonus = 0.1
        self.novelty_threshold = 0.1
        
        # Multi-task learning
        self.task_weights = defaultdict(lambda: 1.0)
        self.task_performance = defaultdict(lambda: 0.0)
    
    def _build_q_network(self) -> nn.Module:
        """Build Q-network for action selection"""
        return nn.Sequential(
            nn.Linear(512, 256),  # Input from model embeddings
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.n_actions)
        ).to(self.device)
    
    def get_action(self, state: Dict[str, Any], task_type: str) -> int:
        """Get action using epsilon-greedy with curiosity"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Get state embedding from model
        state_embedding = self._get_state_embedding(state)
        
        # Add curiosity bonus for novel states
        novelty = self._calculate_novelty(state_embedding)
        if novelty > self.novelty_threshold:
            # Explore more in novel states
            if random.random() < novelty:
                return random.randint(0, self.n_actions - 1)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_embedding.unsqueeze(0))
            
            # Apply task-specific weights
            task_weight = self.task_weights[task_type]
            q_values *= task_weight
            
            return q_values.argmax().item()
    
    def _get_state_embedding(self, state: Dict[str, Any]) -> torch.Tensor:
        """Get state embedding from the multimodal model"""
        # Use the model to encode the current state
        if state.get("task_type") == "text_generation":
            text = state.get("text", "")
            if text:
                # Encode text using the model
                tokens = self.tokenizer.encode(text)
                if len(tokens) > 0:
                    tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embeddings = self.model.text_encoder(tokens)
                        return embeddings.mean(dim=1).squeeze()
        
        # Fallback to random embedding
        return torch.randn(512).to(self.device)
    
    def _calculate_novelty(self, state_embedding: torch.Tensor) -> float:
        """Calculate novelty of current state"""
        if len(self.episodic_memory) == 0:
            return 1.0
        
        # Calculate similarity to stored embeddings
        similarities = []
        for memory_embedding in self.episodic_memory:
            similarity = F.cosine_similarity(
                state_embedding.unsqueeze(0), 
                memory_embedding.unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        return 1.0 - max_similarity
    
    def store_experience(self, state: Dict[str, Any], action: int, reward: float, 
                        next_state: Dict[str, Any], done: bool, task_type: str):
        """Store experience with priority"""
        # Calculate priority based on reward and novelty
        priority = abs(reward) + self._calculate_novelty(self._get_state_embedding(state))
        
        self.replay_buffer.append((state, action, reward, next_state, done, task_type))
        self.priorities.append(priority)
        
        # Store in episodic memory
        state_embedding = self._get_state_embedding(state)
        self.episodic_memory.append(state_embedding)
    
    def train_step(self) -> Dict[str, float]:
        """Train the agent on a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "avg_reward": 0.0}
        
        # Sample batch with priority
        batch_indices = self._sample_prioritized_batch()
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        states, actions, rewards, next_states, dones, task_types = zip(*batch)
        
        # Convert to tensors
        state_embeddings = torch.stack([self._get_state_embedding(s) for s in states])
        next_state_embeddings = torch.stack([self._get_state_embedding(s) for s in next_states])
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(state_embeddings)
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_embeddings)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Calculate loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Add task-specific loss
        task_loss = 0.0
        for i, task_type in enumerate(task_types):
            task_weight = self.task_weights[task_type]
            task_loss += task_weight * loss
        
        total_loss = loss + 0.1 * task_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            "loss": loss.item(),
            "task_loss": task_loss.item(),
            "avg_reward": rewards.mean().item(),
            "epsilon": self.epsilon
        }
    
    def _sample_prioritized_batch(self) -> List[int]:
        """Sample batch using priorities"""
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        return np.random.choice(len(self.replay_buffer), self.batch_size, p=probabilities)
    
    def adapt_to_task(self, task_type: str, task_description: str):
        """Meta-learning: adapt to new task quickly"""
        # Store task embedding
        task_embedding = self._encode_task_description(task_description)
        self.task_embeddings[task_type] = task_embedding
        
        # Quick adaptation using episodic memory
        if len(self.episodic_memory) > 0:
            # Find similar past experiences
            similar_experiences = self._find_similar_experiences(task_embedding)
            
            # Update task weights based on similarity
            if similar_experiences:
                self.task_weights[task_type] = 1.5  # Boost similar tasks
            else:
                self.task_weights[task_type] = 0.8  # Reduce for novel tasks
    
    def _encode_task_description(self, description: str) -> torch.Tensor:
        """Encode task description"""
        tokens = self.tokenizer.encode(description)
        if len(tokens) > 0:
            tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.model.text_encoder(tokens)
                return embeddings.mean(dim=1).squeeze()
        return torch.randn(512).to(self.device)
    
    def _find_similar_experiences(self, task_embedding: torch.Tensor) -> List[torch.Tensor]:
        """Find similar experiences in episodic memory"""
        similarities = []
        for memory_embedding in self.episodic_memory:
            similarity = F.cosine_similarity(
                task_embedding.unsqueeze(0), 
                memory_embedding.unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        # Return experiences with high similarity
        threshold = 0.7
        similar_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
        return [self.episodic_memory[i] for i in similar_indices]
    
    def update_task_performance(self, task_type: str, performance: float):
        """Update task performance for adaptive learning"""
        self.task_performance[task_type] = performance
        
        # Adjust task weights based on performance
        if performance > 0.8:
            self.task_weights[task_type] *= 1.1  # Boost high-performing tasks
        elif performance < 0.3:
            self.task_weights[task_type] *= 0.9  # Reduce low-performing tasks 