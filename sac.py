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
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # self.mu = MLP(state_dim, action_dim)
        # self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.p = MLP(state_dim, 2*action_dim)

        self.apply(weights_init_)
    
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
        return action_bounded, action_log_prob, std

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.buffer = deque(maxlen=capacity)

        self.running_mean_state = np.zeros(state_dim)
        self.running_var_state = np.ones(state_dim)
        self.running_mean_return = 0
        self.running_var_return = 1
        self.normalize_eps = 1e-6
        self.returns = 0.0
        self.gamma = 0.99

        self.clip_state = 10
        self.clip_reward = 10
    
    def push(self, state, action, reward, next_state, done):
        self.returns = self.gamma * self.returns + reward * (1 - done)
        if self.size() == 0:
            self.update_state_stats(state)
        self.buffer.append((state, action, reward, next_state, done))
        self.update_return_stats(self.returns)
        self.update_state_stats(next_state)
    
    def sample(self, batch_size):
        sampled_batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*sampled_batch)
        
        state_normalized = np.clip((state - self.running_mean_state) / np.sqrt(self.running_var_state + self.normalize_eps), -self.clip_state, self.clip_state)
        reward_normalized = np.clip(reward / np.sqrt(self.running_var_return + self.normalize_eps), -self.clip_reward, self.clip_reward)
        next_state_normalized = np.clip((next_state - self.running_mean_state) / np.sqrt(self.running_var_state + self.normalize_eps), -self.clip_state, self.clip_state)

        sampled_batch_normalized = list(zip(state_normalized, action, reward_normalized, next_state_normalized, done))
        return sampled_batch_normalized
    
    def size(self):
        return len(self.buffer)
    
    def update_state_stats(self, state):
        delta_state = state - self.running_mean_state
        self.running_mean_state += delta_state / (self.size()+1)
        self.running_var_state += delta_state * (state - self.running_mean_state)

    def update_return_stats(self, returns):
        delta_return = returns - self.running_mean_return
        self.running_mean_return += delta_return / self.size()
        self.running_var_return += delta_return * (returns - self.running_mean_return) 


# SAC Agent class
class SACAgent:
    def __init__(self, state_dim, action_dim):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = 0.99       # Discount factor
        self.tau = 0.005        # Soft target update factor
        self.lr = 3e-4          # Learning rate
        self.batch_size = 256    # Batch size
        self.buffer_size = 1000000 # Replay buffer size
        self.updates_per_step = 1
        self.warmup_steps = 256

        # self.alpha = 1          # Entropy coefficient # 0.2
        self.log_ent_coef = torch.zeros(1).cuda().requires_grad_(True)
        self._target_entropy = -action_dim
        
        # Networks
        self.actor = PolicyNetwork(state_dim, action_dim).cuda()
        self.critic = QNetwork(state_dim, action_dim).cuda()
        self.target_critic = QNetwork(state_dim, action_dim).cuda()

        self.batch_norm_stats = self._get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = self._get_parameters_by_name(self.target_critic, ["running_"])
        
        # Target value network is the same as value network but with soft target updates
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, state_dim)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).cuda().unsqueeze(0)
        action, _, _ = self.actor.sample(state)
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

        sampled_action, action_log_prob, std = self.actor.sample(state_batch)
        
        self.alpha = torch.exp(self.log_ent_coef).detach().item()
        # self.alpha = 0.0001

        # entropy coefficient update
        ent_coef_loss = -(self.log_ent_coef * (action_log_prob + self._target_entropy).detach()).mean()

        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        # Critic update
        with torch.no_grad():
            sampled_action_next, action_log_prob_next, _ = self.actor.sample(next_state_batch)
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
        self._polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        # for target_param, param in zip(self.batch_norm_stats_target, self.batch_norm_stats):
        #     target_param.data.copy_(param.data)


        info = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'ent_coef_loss': ent_coef_loss.item(),
            'ent_coef': self.alpha,
            'log_pi': action_log_prob.mean().item(),
            'pi_std': std.mean().item(),
            'running_state_mean': self.replay_buffer.running_mean_state,
            'running_state_var': self.replay_buffer.running_var_state,
            'running_return_mean': self.replay_buffer.running_mean_return,
            'running_return_var': self.replay_buffer.running_var_return,
        }
        return info
    
    def _get_parameters_by_name(self, model, included_names):
        """
        Extract parameters from the state dict of ``model``
        if the name contains one of the strings in ``included_names``.

        :param model: the model where the parameters come from.
        :param included_names: substrings of names to include.
        :return: List of parameters values (Pytorch tensors)
            that matches the queried names.
        """
        return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]
    
    def _polyak_update(
        self,
        params,
        target_params,
        tau,
    ) -> None:
        """
        Perform a Polyak average update on ``target_params`` using ``params``:
        target parameters are slowly updated towards the main parameters.
        ``tau``, the soft update coefficient controls the interpolation:
        ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
        The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
        or a computation graph, reducing memory cost and improving performance.  We scale the target params
        by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
        params (in place).
        See https://github.com/DLR-RM/stable-baselines3/issues/93

        :param params: parameters to use to update the target params
        :param target_params: parameters to update
        :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
        """
        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for param, target_param in self._zip_strict(params, target_params):
                target_param.data.mul_(1 - tau)
                torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
    
    def _zip_strict(self, *iterables):
        r"""
        ``zip()`` function but enforces that iterables are of equal length.
        Raises ``ValueError`` if iterables not of equal length.
        Code inspired by Stackoverflow answer for question #32954486.

        :param \*iterables: iterables to ``zip()``
        """
        # As in Stackoverflow #32954486, use
        # new object for "empty" in case we have
        # Nones in iterable.
        sentinel = object()
        for combo in zip_longest(*iterables, fillvalue=sentinel):
            if sentinel in combo:
                raise ValueError("Iterables have different lengths")
            yield combo