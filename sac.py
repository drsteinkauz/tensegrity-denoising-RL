import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from collections.abc import Iterable
from itertools import zip_longest
from typing import List

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Define the neural networks for actor, critic, and target networks
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)

        self.apply(weights_init_)
    
    def forward(self, state, action):
        q1_value = self.q1(torch.cat([state, action], dim=-1))
        q2_value = self.q2(torch.cat([state, action], dim=-1))
        return q1_value, q2_value


class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # self.mu = MLP(observation_dim, action_dim)
        # self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.p = MLP(observation_dim, 2*action_dim)

        self.apply(weights_init_)
    
    def forward(self, observation):
        # mean = self.mu(observation)
        # log_std = torch.clamp(self.log_std, min=-20, max=2)
        # std = torch.exp(log_std)
        mu_sigma = self.p(observation)
        mean, log_std = mu_sigma.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, observation):
        epsilon_tanh = 1e-6
        mean, std = self.forward(observation)
        dist = torch.distributions.Normal(mean, std)
        action_unbounded = dist.rsample()
        action_bounded = torch.tanh(action_unbounded) * (1 - epsilon_tanh)
        action_log_prob = dist.log_prob(action_unbounded)
        action_log_prob -= torch.log(1 - action_bounded.pow(2) + epsilon_tanh)
        action_log_prob = action_log_prob.sum(1, keepdim=True)
        mean_bounded = torch.tanh(mean)
        return action_bounded, action_log_prob, std
    
    def predict(self, observation):
        epsilon_tanh = 1e-6
        mean, std = self.forward(observation)
        dist = torch.distributions.Normal(mean, std)
        action_unbounded = dist.rsample()
        action_bounded = torch.tanh(action_unbounded) * (1 - epsilon_tanh)
        return action_bounded, std
    
    def to(self, device):
        self.device = device
        return super(PolicyNetwork, self).to(device)

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, observation, action, reward, next_state, next_observation, done):
        self.buffer.append((state, observation, action, reward, next_state, next_observation, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)


# SAC Agent class
class SACAgent:
    def __init__(self, state_dim, observation_dim, action_dim, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # Device
        self.device = device

        # Hyperparameters
        self.gamma = 0.99       # Discount factor
        self.tau = 0.005        # Soft target update factor
        self.lr = 3e-4          # Learning rate
        self.batch_size = 256    # Batch size
        self.buffer_size = 1000000 # Replay buffer size
        self.updates_per_step = 1
        self.warmup_steps = 256

        # self.alpha = 1          # Entropy coefficient # 0.2
        self.log_ent_coef = torch.zeros(1).to(self.device).requires_grad_(True)
        self._target_entropy = -action_dim
        
        # Networks
        # self.actor = PolicyNetwork(state_dim, action_dim).to(self.device)
        # self.critic = QNetwork(state_dim, action_dim).to(self.device)
        # self.target_critic = QNetwork(state_dim, action_dim).to(self.device)
        self.actor = torch.nn.DataParallel(PolicyNetwork(observation_dim, action_dim)).to(self.device)
        self.critic = torch.nn.DataParallel(QNetwork(state_dim, action_dim)).to(self.device)
        self.target_critic = torch.nn.DataParallel(QNetwork(state_dim, action_dim)).to(self.device)
        
        # Target value network is the same as value network but with soft target updates
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
    
    def select_action(self, observation):
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.actor.module.sample(observation)
        return action.squeeze(0).cpu().numpy()
    
    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, observation_batch, action_batch, reward_batch, next_state_batch, next_observation_batch, done_batch = zip(*batch)
        
        with torch.no_grad():
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            observation_batch = torch.FloatTensor(observation_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            next_observation_batch = torch.FloatTensor(next_observation_batch).to(self.device)
            done_batch = torch.FloatTensor(done_batch).to(self.device)

        sampled_action, action_log_prob, std = self.actor.module.sample(observation_batch)
        
        self.alpha = torch.exp(self.log_ent_coef).detach().item()
        # self.alpha = 0.0001

        # entropy coefficient update
        ent_coef_loss = -(self.log_ent_coef * (action_log_prob + self._target_entropy).detach()).mean()

        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        # Critic update
        with torch.no_grad():
            sampled_action_next, action_log_prob_next, _ = self.actor.module.sample(next_observation_batch)
            q1_target_next_pi, q2_target_next_pi = self.target_critic(next_state_batch, sampled_action_next)
            q_target_next_pi = torch.min(q1_target_next_pi, q2_target_next_pi)
            next_q_value = reward_batch.view(-1, 1) + self.gamma * (1 - done_batch.view(-1, 1)) * (q_target_next_pi - self.alpha * action_log_prob_next)
        q1_value, q2_value = self.critic(state_batch, action_batch)
        critic1_loss = F.mse_loss(q1_value, next_q_value)
        critic2_loss = F.mse_loss(q2_value, next_q_value)
        critic_loss = (critic1_loss + critic2_loss) / 2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        q1_pi, q2_pi = self.critic(state_batch, sampled_action)
        q_value_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * action_log_prob - q_value_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft target update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        info = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'ent_coef_loss': ent_coef_loss.item(),
            'ent_coef': self.alpha,
            'log_pi': action_log_prob.mean().item(),
            'pi_std': std.mean().item(),
        }
        return info
