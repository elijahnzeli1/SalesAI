import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import logging

from model.salesa_model import SalesAModel
from tokenizer import SalesATokenizer

logger = logging.getLogger(__name__)

class AGIEnvironment:
    """Abstract environment for AGI seed (text, audio, robotics, etc.)"""
    def reset(self):
        """Reset environment to initial state"""
        raise NotImplementedError

    def step(self, action):
        """Take action and return (next_observation, reward, done, info)"""
        raise NotImplementedError

    def get_observation(self):
        """Get current observation"""
        raise NotImplementedError

class SimpleTextEnv(AGIEnvironment):
    """Simple text-based environment for demonstration"""
    def __init__(self):
        self.state = 0
        self.max_steps = 10
        self.steps = 0
        self.goal_state = 5

    def reset(self):
        self.state = 0
        self.steps = 0
        return self.get_observation()

    def step(self, action):
        # Reward is +1 if action == state+1, else 0
        # Bonus reward for reaching goal state
        reward = 1.0 if action == self.state + 1 else 0.0
        if self.state == self.goal_state:
            reward += 5.0  # Bonus for reaching goal
        
        self.state += 1
        self.steps += 1
        done = self.steps >= self.max_steps or self.state >= self.goal_state
        
        return self.get_observation(), reward, done, {"state": self.state}

    def get_observation(self):
        # Return a simple text prompt as observation
        return f"Current state: {self.state}, Goal: {self.goal_state}"

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for better sample efficiency"""
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, transition: Tuple, priority: float = None):
        """Save a transition with priority"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, List[int], torch.Tensor]:
        """Sample a batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], [], torch.tensor([])
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)

class EpisodicMemory:
    """Enhanced episodic memory for tracking novel states and experiences"""
    def __init__(self, capacity: int = 1000, novelty_threshold: float = 0.1):
        self.memory = deque(maxlen=capacity)
        self.novelty_threshold = novelty_threshold
        self.state_embeddings = []

    def add(self, state, embedding=None):
        """Add state to memory with optional embedding"""
        self.memory.append(state)
        if embedding is not None:
            self.state_embeddings.append(embedding)

    def is_novel(self, state, embedding=None) -> bool:
        """Check if state is novel using embedding similarity"""
        if embedding is None:
            return state not in self.memory
        
        if len(self.state_embeddings) == 0:
            return True
        
        # Calculate similarity with existing embeddings
        similarities = [F.cosine_similarity(embedding.unsqueeze(0), 
                                          existing_emb.unsqueeze(0), dim=1).item() 
                       for existing_emb in self.state_embeddings]
        
        max_similarity = max(similarities)
        return max_similarity < self.novelty_threshold

    def get_similar_experiences(self, embedding, k: int = 5):
        """Retrieve similar experiences for meta-learning"""
        if len(self.state_embeddings) == 0:
            return []
        
        similarities = [(i, F.cosine_similarity(embedding.unsqueeze(0), 
                                              emb.unsqueeze(0), dim=1).item()) 
                       for i, emb in enumerate(self.state_embeddings)]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.memory[i] for i, _ in similarities[:k]]

    def __len__(self) -> int:
        return len(self.memory)

class MetaLearner:
    """Meta-learning component for rapid adaptation to new tasks"""
    def __init__(self, model: SalesAModel, learning_rate: float = 1e-4):
        self.model = model
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.task_memory = []

    def adapt_to_task(self, task_examples: List[Tuple], adaptation_steps: int = 5):
        """Quick adaptation to a new task using few-shot learning"""
        if not task_examples:
            return
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Quick adaptation
        for _ in range(adaptation_steps):
            for obs, action, reward, next_obs, done in task_examples:
                # Forward pass
                input_ids = torch.tensor([self.model.tokenizer.encode(obs)], dtype=torch.long)
                outputs = self.model(input_ids, task_type="action")
                loss = F.cross_entropy(outputs["logits"], torch.tensor([action]))
                
                # Update
                self.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_optimizer.step()
        
        # Store task information
        self.task_memory.append({
            "examples": task_examples,
            "adapted_params": {name: param.clone() for name, param in self.model.named_parameters()}
        })

    def get_task_similarity(self, current_task: List[Tuple]) -> float:
        """Calculate similarity with previous tasks"""
        if not self.task_memory:
            return 0.0
        
        # Simple similarity based on action patterns
        current_actions = [action for _, action, _, _, _ in current_task]
        
        similarities = []
        for task_info in self.task_memory:
            task_actions = [action for _, action, _, _, _ in task_info["examples"]]
            # Calculate action pattern similarity
            similarity = len(set(current_actions) & set(task_actions)) / len(set(current_actions) | set(task_actions))
            similarities.append(similarity)
        
        return max(similarities) if similarities else 0.0

class QNetwork(nn.Module):
    """Enhanced Q-Network using SalesA model backbone with dueling architecture"""
    def __init__(self, model: SalesAModel, n_actions: int, dueling: bool = True):
        super().__init__()
        self.model = model
        self.n_actions = n_actions
        self.dueling = dueling
        
        if dueling:
            # Dueling DQN: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(model.config.hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(model.config.hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )
        else:
            self.q_head = nn.Linear(model.config.hidden_dim, n_actions)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network"""
        # Use text encoder and transformer backbone
        x = self.model.text_encoder(input_ids)
        for block in self.model.transformer_blocks:
            x = block(x)
        
        # Use last token's hidden state for Q-value prediction
        last_hidden = x[:, -1, :]
        
        if self.dueling:
            value = self.value_stream(last_hidden)
            advantage = self.advantage_stream(last_hidden)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(last_hidden)
        
        return q_values

class DQNAgent:
    """Enhanced DQN agent with SalesA model backbone and advanced features"""
    def __init__(self, model: SalesAModel, tokenizer: SalesATokenizer, n_actions: int,
                 buffer_capacity: int = 10000, memory_capacity: int = 1000,
                 double_dqn: bool = True, dueling: bool = True):
        self.device = next(model.parameters()).device
        self.tokenizer = tokenizer
        
        # Networks
        self.q_net = QNetwork(model, n_actions, dueling).to(self.device)
        self.target_q_net = QNetwork(model, n_actions, dueling).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        
        # Memory
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        self.episodic_memory = EpisodicMemory(capacity=memory_capacity)
        
        # Meta-learning component
        self.meta_learner = MetaLearner(model)
        
        # Parameters
        self.n_actions = n_actions
        self.gamma = 0.99
        self.epsilon = 0.1
        self.curiosity_bonus = 0.05
        self.update_target_every = 10
        self.episode_count = 0
        
        # Advanced features
        self.double_dqn = double_dqn
        self.n_step_returns = 3
        self.n_step_buffer = deque(maxlen=self.n_step_returns)

    def obs_to_tensor(self, obs: str) -> torch.Tensor:
        """Convert observation to tensor"""
        return torch.tensor([self.tokenizer.encode(obs)], dtype=torch.long).to(self.device)

    def select_action(self, obs: str, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy with exploration bonus"""
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            input_ids = self.obs_to_tensor(obs)
            q_values = self.q_net(input_ids)
            
            # Add exploration bonus for novel states
            if self.episodic_memory.is_novel(obs):
                q_values += self.curiosity_bonus
            
            return int(torch.argmax(q_values).item())

    def compute_intrinsic_reward(self, obs: str) -> float:
        """Compute intrinsic reward based on novelty and task similarity"""
        novelty_reward = self.curiosity_bonus if self.episodic_memory.is_novel(obs) else 0.0
        
        # Task similarity reward (encourage exploration of similar tasks)
        task_similarity = self.meta_learner.get_task_similarity([])  # Current task context
        similarity_reward = task_similarity * 0.1
        
        return novelty_reward + similarity_reward

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step with prioritized replay"""
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch with priorities
        batch, indices, weights = self.replay_buffer.sample(batch_size)
        weights = weights.to(self.device)
        
        obs_batch = torch.cat([self.obs_to_tensor(b[0]) for b in batch])
        action_batch = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
        reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_obs_batch = torch.cat([self.obs_to_tensor(b[3]) for b in batch])
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)

        # Compute Q values
        q_values = self.q_net(obs_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network for action selection, target for evaluation
                next_q_values = self.q_net(next_obs_batch)
                next_actions = next_q_values.argmax(1)
                next_q_values_target = self.target_q_net(next_obs_batch)
                max_next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_q_net(next_obs_batch)
                max_next_q = next_q_values.max(1)[0]
            
            target = reward_batch + self.gamma * max_next_q * (1 - done_batch)

        # Compute loss with importance sampling weights
        loss = F.mse_loss(q_value, target, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Update priorities
        priorities = loss.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

        # Update network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return weighted_loss.item()

    def update_target_network(self):
        """Update target network weights"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train_episode(self, env: AGIEnvironment) -> Dict[str, float]:
        """Train for one episode with enhanced features"""
        obs = env.reset()
        total_reward = 0.0
        losses = []
        episode_transitions = []

        while True:
            # Select and take action
            action = self.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            # Compute intrinsic reward and update memory
            intrinsic = self.compute_intrinsic_reward(next_obs)
            self.episodic_memory.add(next_obs)
            total_r = reward + intrinsic

            # Store transition
            transition = (obs, action, total_r, next_obs, done)
            self.replay_buffer.push(transition)
            episode_transitions.append(transition)

            # Train
            if loss := self.train_step():
                losses.append(loss)

            total_reward += total_r
            obs = next_obs

            if done:
                break

        # Meta-learning: adapt to this episode's task
        if len(episode_transitions) > 5:  # Only adapt if enough transitions
            self.meta_learner.adapt_to_task(episode_transitions)

        # Update target network periodically
        self.episode_count += 1
        if self.episode_count % self.update_target_every == 0:
            self.update_target_network()

        return {
            "reward": total_reward,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "memory_size": len(self.episodic_memory),
            "buffer_size": len(self.replay_buffer),
            "episode_length": len(episode_transitions)
        }

    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'replay_buffer': self.replay_buffer,
            'episodic_memory': self.episodic_memory
        }, path)

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        if 'replay_buffer' in checkpoint:
            self.replay_buffer = checkpoint['replay_buffer']
        if 'episodic_memory' in checkpoint:
            self.episodic_memory = checkpoint['episodic_memory'] 