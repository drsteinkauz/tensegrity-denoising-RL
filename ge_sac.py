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
    
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim=256):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.apply(weights_init_)

    def encode(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, hidden_n = self.gru(x)
        last_out = out[:, -1, :]
        feature = self.encoder(last_out)
        return feature

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        feature = self.encode(x)
        return feature

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity, state_dim, obs_dim, action_dim, obs_act_seq_len=64, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.obs_act_seq = torch.zeros((capacity, obs_act_seq_len, obs_dim+action_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs_act_seq = torch.zeros((capacity, obs_act_seq_len, obs_dim+action_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
    
    def push(self, state, observation, obs_act_seq, action, reward, next_state, next_observation, next_obs_act_seq, done):
        i = self.ptr

        with torch.no_grad():
            self.states[i] = torch.tensor(state, dtype=torch.float32, device=self.device)
            self.observations[i] = torch.tensor(observation, dtype=torch.float32, device=self.device)
            self.obs_act_seq[i] = torch.tensor(obs_act_seq, dtype=torch.float32, device=self.device)
            self.actions[i] = torch.tensor(action, dtype=torch.float32, device=self.device)
            self.rewards[i] = torch.tensor([reward], dtype=torch.float32, device=self.device)
            self.next_states[i] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            self.next_observations[i] = torch.tensor(next_observation, dtype=torch.float32, device=self.device)
            self.next_obs_act_seq[i] = torch.tensor(next_obs_act_seq, dtype=torch.float32, device=self.device)
            self.dones[i] = torch.tensor([done], dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        # batch = random.sample(self.buffer, batch_size)
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        batch = (
            self.states[indices],
            self.observations[indices],
            self.obs_act_seq[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.next_observations[indices],
            self.next_obs_act_seq[indices],
            self.dones[indices],
        )
        return batch


# SAC Agent class
class SACAgent:
    def __init__(self, state_dim, observation_dim, action_dim, inheparam_dim, inheparam_range, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # Device
        self.device = device

        # Hyperparameters
        self.gamma = 0.99       # Discount factor
        self.tau = 0.005        # Soft target update factor
        self.lr = 3e-4          # Learning rate
        self.lr_GE = 1e-3       # Learning rate for GRUAutoEncoder
        self.batch_size = 256    # Batch size
        self.buffer_size = 1000000 # Replay buffer size
        self.updates_per_step = 1
        self.warmup_steps = 256
        self.lambda_for_GAE = 0.1

        self.inheparam_dim = inheparam_dim
        with torch.no_grad():
            self.inheparam_mean = torch.tensor((inheparam_range[:,0] + inheparam_range[:,1]) / 2, dtype=torch.float32, device=self.device).view(1, -1).detach()
            self.inheparam_std = torch.tensor((inheparam_range[:,1] - inheparam_range[:,0]) / 2, dtype=torch.float32, device=self.device).view(1, -1).detach()
            # shape: (1, inheparam_dim)

        # self.alpha = 1          # Entropy coefficient # 0.2
        self.log_ent_coef = torch.zeros(1).to(self.device).requires_grad_(True)
        self._target_entropy = -action_dim
        
        # Networks
        self.actor = PolicyNetwork(observation_dim+inheparam_dim, action_dim).to(self.device)
        self.critic = QNetwork(state_dim, action_dim).to(self.device)
        self.target_critic = QNetwork(state_dim, action_dim).to(self.device)
        self.gruencoder = GRUEncoder(observation_dim+action_dim, inheparam_dim).to(self.device)
        
        # Target value network is the same as value network but with soft target updates
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.gruencoder_optimizer = optim.Adam(self.gruencoder.parameters(), lr=self.lr_GE)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size, state_dim=state_dim, obs_dim=observation_dim, action_dim=action_dim, obs_act_seq_len=64, device=self.device)
    
    def select_action(self, obs_act_seq, observation, state):
        obs_act_seq = torch.FloatTensor(obs_act_seq).to(self.device).unsqueeze(0)
        predicted_inheparam = self.gruencoder.encode(obs_act_seq)
        gt_inheparam = torch.FloatTensor(state[-self.inheparam_dim:]).to(self.device).unsqueeze(0)
        with torch.no_grad():
            # feature = torch.cat([torch.FloatTensor(observation).to(self.device).unsqueeze(0), predicted_inheparam], dim=-1)
            feature = torch.cat([torch.FloatTensor(observation).to(self.device).unsqueeze(0), gt_inheparam], dim=-1)
            action, _, _ = self.actor.sample(feature)
        return action.squeeze(0).cpu().numpy()
    
    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return
        
        # batch = self.replay_buffer.sample(self.batch_size)
        # state_batch, observation_batch, obs_act_seq_batch, action_batch, reward_batch, next_state_batch, next_observation_batch, next_obs_act_seq_batch, done_batch = zip(*batch)

        # with torch.no_grad():
        #     state_batch = torch.FloatTensor(state_batch).to(self.device)
        #     observation_batch = torch.FloatTensor(observation_batch).to(self.device)
        #     obs_act_seq_batch = torch.FloatTensor(obs_act_seq_batch).to(self.device)
        #     action_batch = torch.FloatTensor(action_batch).to(self.device)
        #     reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        #     next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        #     next_observation_batch = torch.FloatTensor(next_observation_batch).to(self.device)
        #     next_obs_act_seq_batch = torch.FloatTensor(next_obs_act_seq_batch).to(self.device)
        #     done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        state_batch, observation_batch, obs_act_seq_batch, action_batch, reward_batch, next_state_batch, next_observation_batch, next_obs_act_seq_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        predicted_inheparam_batch = self.gruencoder.encode(obs_act_seq_batch)
        gt_inheparam_batch = state_batch[:, -self.inheparam_dim:].detach()
        # feature_batch = torch.cat([observation_batch, predicted_inheparam_batch], dim=-1)
        feature_batch = torch.cat([observation_batch, gt_inheparam_batch], dim=-1)
        sampled_action, action_log_prob, std = self.actor.sample(feature_batch.detach())

        # GRUAutoEncoder update
        predicted_inheparam_batch_normalized = (predicted_inheparam_batch - self.inheparam_mean) / self.inheparam_std
        gt_inheparam_batch_normalized = (gt_inheparam_batch - self.inheparam_mean) / self.inheparam_std
        predict_error = F.mse_loss(predicted_inheparam_batch_normalized, gt_inheparam_batch_normalized)
        predict_loss = 0.5*predict_error

        self.gruencoder_optimizer.zero_grad()
        predict_loss.backward()
        self.gruencoder_optimizer.step()
        
        # entropy coefficient update
        self.alpha = torch.exp(self.log_ent_coef).detach().item()

        ent_coef_loss = -(self.log_ent_coef * (action_log_prob + self._target_entropy).detach()).mean()

        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        # Critic update
        with torch.no_grad():
            next_predicted_inheparam_batch = self.gruencoder.encode(next_obs_act_seq_batch)
            next_gt_inheparam_batch = next_state_batch[:, -self.inheparam_dim:].detach()
            # next_feature_batch = torch.cat([next_observation_batch, next_predicted_inheparam_batch], dim=-1)
            next_feature_batch = torch.cat([next_observation_batch, next_gt_inheparam_batch], dim=-1)
            sampled_action_next, action_log_prob_next, _ = self.actor.sample(next_feature_batch)
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
            'predict_loss': predict_loss.item(),
            'predict_error': predict_error.item(),
            'log_pi': action_log_prob.mean().item(),
            'pi_std': std.mean().item(),
        }
        return info
