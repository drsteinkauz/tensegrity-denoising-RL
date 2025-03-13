import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

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


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SoftQNetwork, self).__init__()
        self.q = MLP(state_dim + action_dim, 1)
    
    def forward(self, state, action):
        q_value = self.q(torch.cat([state, action], dim=-1))
        return q_value


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.v = MLP(state_dim, 1)
    
    def forward(self, state):
        return self.v(state)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # self.mu = MLP(state_dim, action_dim)
        # self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.p = MLP(state_dim, 2*action_dim)
    
    def forward(self, state):
        # mean = self.mu(state)
        # log_std = torch.clamp(self.log_std, min=-20, max=2)
        # std = torch.exp(log_std)
        mu_sigma = self.p(state)
        mean, log_std = mu_sigma.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, state):
        epsilon_tanh = 1e-6
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action_unbounded = dist.sample()
        action_bounded = torch.tanh(action_unbounded) * (1 - epsilon_tanh)
        action_log_prob = dist.log_prob(action_unbounded)
        action_log_prob -= torch.log(1 - action_bounded.pow(2) + epsilon_tanh)
        action_log_prob = action_log_prob.sum(1, keepdim=True)
        mean_bounded = torch.tanh(mean)
        return action_bounded, action_log_prob, mean_bounded, mean, std

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)


# SAC Agent class
class SACAgent:
    def __init__(self, state_dim, action_dim):
        # Hyperparameters
        self.gamma = 0.99       # Discount factor
        self.tau = 0.005        # Soft target update factor
        self.lr = 3e-4          # Learning rate
        self.batch_size = 64    # Batch size
        self.buffer_size = 1000000 # Replay buffer size
        self.updates_per_step = 1

        # self.alpha = 1          # Entropy coefficient # 0.2
        self.log_ent_coef = torch.zeros(1).cuda().requires_grad_(True)
        self._target_entropy = -action_dim
        
        # Networks
        self.actor = PolicyNetwork(state_dim, action_dim).cuda()
        self.critic1 = SoftQNetwork(state_dim, action_dim).cuda()
        self.critic2 = SoftQNetwork(state_dim, action_dim).cuda()
        self.target_value = ValueNetwork(state_dim).cuda()
        self.value = ValueNetwork(state_dim).cuda()
        
        # Target value network is the same as value network but with soft target updates
        self.target_value.load_state_dict(self.value.state_dict())
        
        # Optimizers
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
    
    def select_action(self, state):
        epsilon_tanh = 1e-6
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = torch.tanh(dist.sample()) * (1 - epsilon_tanh)
        return action.squeeze(0).cpu().numpy()
    
    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(state_batch).cuda()
        action_batch = torch.FloatTensor(action_batch).cuda()
        reward_batch = torch.FloatTensor(reward_batch).cuda()
        next_state_batch = torch.FloatTensor(next_state_batch).cuda()
        done_batch = torch.FloatTensor(done_batch).cuda()

        self.alpha = torch.exp(self.log_ent_coef).detach().item()
        # self.alpha = 0.0001
        
        # Critic update
        with torch.no_grad():
            target_value = self.target_value(next_state_batch)
            target_q_value = reward_batch.view(-1, 1) + self.gamma * (1 - done_batch.view(-1, 1)) * target_value
        
        q1_value = self.critic1(state_batch, action_batch)
        critic1_loss = F.mse_loss(q1_value, target_q_value)
        
        q2_value = self.critic2(state_batch, action_batch)
        critic2_loss = F.mse_loss(q2_value, target_q_value)
        
        self.critic_loss = (critic1_loss + critic2_loss) / 2    
        
        # Actor update
        sampled_action, action_log_prob, _, mean, std = self.actor.sample(state_batch)
        
        q1_pi = self.critic1(state_batch, sampled_action)
        q2_pi = self.critic2(state_batch, sampled_action)
        q_value_pi = torch.min(q1_pi, q1_pi)
        
        self.actor_loss = (self.alpha * action_log_prob - q_value_pi).mean()

        # Value update
        value_pred = self.value(state_batch)
        # with torch.no_grad():
        #     sampled_action_v, action_log_prob_v, _, _, _ = self.actor.sample(state_batch)
        #     q1_pi_v = self.critic1(state_batch, sampled_action_v)
        #     q2_pi_v = self.critic2(state_batch, sampled_action_v)
        #     q_value_pi_v = torch.min(q1_pi_v, q1_pi_v)
        # self.value_loss = 0.5 * F.mse_loss(value_pred, (q_value_pi_v - self.alpha * action_log_prob_v))
        self.value_loss = 0.5 * F.mse_loss(value_pred, (q_value_pi - self.alpha * action_log_prob).detach())


        # Entropy Coefficient update
        self.ent_coef_loss = -(self.log_ent_coef * (action_log_prob + self._target_entropy).detach()).mean()
        
        # Optimizer step
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        
        # self.critic1_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic1_optimizer.step()
        
        # self.critic2_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic2_optimizer.step()
        
        # self.value_optimizer.zero_grad()
        # value_loss.backward()
        # self.value_optimizer.step()

        total_loss = self.critic_loss + 0.1*self.actor_loss + self.value_loss + self.ent_coef_loss
        
        # gradient = torch.autograd.grad(outputs=self.ent_coef_loss, inputs=self.log_ent_coef, create_graph=True)
        # print("ent_grad", gradient)
        # print("action_log_prob", action_log_prob.mean())
        # print("action_batch", action_batch.mean(dim=0))
        # print("action_batch_atanh", torch.atanh(action_batch).mean(dim=0))
        # print("mean", mean.mean(dim=0))
        # print("std", std.mean(dim=0))
        # print("grad", -((action_log_prob + self._target_entropy).detach()).mean())
        # gradient = torch.autograd.grad(outputs=actor_loss, inputs=self.actor.parameters(), create_graph=True)
        # print("act_grad", gradient)
        # gradient = torch.autograd.grad(outputs=critic_loss, inputs=self.critic1.parameters(), create_graph=True)
        # print("crit1_grad", gradient)
        # gradient = torch.autograd.grad(outputs=critic_loss, inputs=self.critic2.parameters(), create_graph=True)
        # print("crit2_grad", gradient)
        # gradient = torch.autograd.grad(outputs=value_loss, inputs=self.value.parameters(), create_graph=True)
        # print("val_grad", gradient)
        # gradient = torch.autograd.grad(outputs=self.actor_loss, inputs=self.critic1.parameters(), create_graph=True)
        # print("actor-q grad", gradient)

        # print("critic loss", self.critic_loss)
        # print("actor loss", self.actor_loss)
        # print("value loss", self.value_loss)
        
        self.actor_optimizer.zero_grad()
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.ent_coef_optimizer.zero_grad()

        total_loss.backward()

        self.actor_optimizer.step()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        self.value_optimizer.step()
        self.ent_coef_optimizer.step()
        
        # Soft target update
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)