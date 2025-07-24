import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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
        self.max_steps = 5
        self.steps = 0

    def reset(self):
        self.state = 0
        self.steps = 0
        return self.get_observation()

    def step(self, action):
        # Reward is +1 if action == state+1, else 0
        reward = 1.0 if action == self.state + 1 else 0.0
        self.state += 1
        self.steps += 1
        done = self.steps >= self.max_steps
        return self.get_observation(), reward, done, {}

    def get_observation(self):
        # Return a simple text prompt as observation
        return f"Current state: {self.state}"

class ReplayBuffer:
    """Experience replay buffer for RL training"""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition: Tuple):
        """Save a transition"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

class EpisodicMemory:
    """Episodic memory for tracking novel states"""
    def __init__(self, capacity: int = 100):
        self.memory = deque(maxlen=capacity)

    def add(self, state):
        """Add state to memory"""
        self.memory.append(state)

    def is_novel(self, state) -> bool:
        """Check if state is novel"""
        return state not in self.memory

    def __len__(self) -> int:
        return len(self.memory)

class QNetwork(nn.Module):
    """Q-Network using SalesA model backbone"""
    def __init__(self, model: SalesAModel, n_actions: int):
        super().__init__()
        self.model = model
        self.n_actions = n_actions
        self.q_head = nn.Linear(model.config.hidden_dim, n_actions)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network"""
        # Use text encoder and transformer backbone
        x = self.model.text_encoder(input_ids)
        for block in self.model.transformer_blocks:
            x = block(x)
        # Use last token's hidden state for Q-value prediction
        q_values = self.q_head(x[:, -1, :])
        return q_values

class DQNAgent:
    """DQN agent with SalesA model backbone"""
    def __init__(self, model: SalesAModel, tokenizer: SalesATokenizer, n_actions: int,
                 buffer_capacity: int = 10000, memory_capacity: int = 100):
        self.device = next(model.parameters()).device
        self.tokenizer = tokenizer
        
        # Networks
        self.q_net = QNetwork(model, n_actions).to(self.device)
        self.target_q_net = QNetwork(model, n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        
        # Memory
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.episodic_memory = EpisodicMemory(capacity=memory_capacity)
        
        # Parameters
        self.n_actions = n_actions
        self.gamma = 0.99
        self.epsilon = 0.2
        self.curiosity_bonus = 0.1
        self.update_target_every = 5
        self.episode_count = 0

    def obs_to_tensor(self, obs: str) -> torch.Tensor:
        """Convert observation to tensor"""
        return torch.tensor([self.tokenizer.encode(obs)], dtype=torch.long).to(self.device)

    def select_action(self, obs: str, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            input_ids = self.obs_to_tensor(obs)
            q_values = self.q_net(input_ids)
            return int(torch.argmax(q_values).item())

    def compute_intrinsic_reward(self, obs: str) -> float:
        """Compute intrinsic reward based on novelty"""
        return self.curiosity_bonus if self.episodic_memory.is_novel(obs) else 0.0

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
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
            next_q_values = self.target_q_net(next_obs_batch)
            max_next_q = next_q_values.max(1)[0]
            target = reward_batch + self.gamma * max_next_q * (1 - done_batch)

        # Compute loss and update
        loss = F.mse_loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network weights"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train_episode(self, env: AGIEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        obs = env.reset()
        total_reward = 0.0
        losses = []

        while True:
            # Select and take action
            action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)

            # Compute intrinsic reward and update memory
            intrinsic = self.compute_intrinsic_reward(next_obs)
            self.episodic_memory.add(next_obs)
            total_r = reward + intrinsic

            # Store transition
            self.replay_buffer.push((obs, action, total_r, next_obs, done))

            # Train
            if loss := self.train_step():
                losses.append(loss)

            total_reward += total_r
            obs = next_obs

            if done:
                break

        # Update target network periodically
        self.episode_count += 1
        if self.episode_count % self.update_target_every == 0:
            self.update_target_network()

        return {
            "reward": total_reward,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "memory_size": len(self.episodic_memory),
            "buffer_size": len(self.replay_buffer)
        }

    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count
        }, path)

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count'] 