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

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp_update = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.apply(weights_init_)

    def forward(self, node_attr, edge_index, edge_attr):
        # node_attr: [batch, node_num, node_dim]
        # edge_index: [2, edge_num]
        # edge_attr: [batch, edge_num, edge_dim]
        B, N, D = node_attr.shape
        E = edge_index.shape[1]

        idx_received, idx_sent = edge_index  # [edge_num]
        node_received = node_attr[:, idx_received, :]   # [batch, edge_num, node_dim]
        node_sent = node_attr[:, idx_sent, :]   # [batch, edge_num, node_dim]

        msg_input = torch.cat([node_received, node_sent, edge_attr], dim=-1)  # [batch, edge_num, 2D+edge_dim]
        msg = self.mlp_msg(msg_input)  # [batch, edge_num, hidden_dim]

        # aggregation (sum over incoming messages per node)
        agg_msg = torch.zeros(B, N, msg.shape[-1], 
                             device=node_attr.device, 
                             dtype=msg.dtype)  
        
        # vectorized contennation
        idx_expanded = idx_received.unsqueeze(0).unsqueeze(-1).expand(B, -1, msg.size(-1))
        agg_msg.scatter_add_(1, idx_expanded, msg)

        update_input = torch.cat([node_attr, agg_msg], dim=-1)  # [batch, node_num, node_dim+out_dim]
        out = self.mlp_update(update_input)  # [batch, node_num, out_dim]
        return out

class ActorHead(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.apply(weights_init_)

    def forward(self, h, edge_index, edge_attr, edge_type_mask):
        # h: [batch, node_num, hidden_dim]
        # edge_index: [2, edge_num]
        # edge_attr: [batch, edge_num, edge_dim]
        idx_received, idx_sent = edge_index
        h_received = h[:, idx_received, :] # [batch, edge_num, hidden_dim]
        h_sent = h[:, idx_sent, :] # [batch, edge_num, hidden_dim]

        h_received_a = h_received[:, edge_type_mask, :]  # [batch, active_edge_num, hidden_dim]
        h_sent_a = h_sent[:, edge_type_mask, :]  # [batch, active_edge_num, hidden_dim]
        edge_attr_a = edge_attr[:, edge_type_mask, :]  # [batch, active_edge_num, edge_dim]

        actor_input = torch.cat([h_received_a, h_sent_a, edge_attr_a], dim=-1) # [batch, active_edge_num, 2*hidden_dim + edge_dim]
        actor_output = self.edge_mlp(actor_input)  # [batch, active_edge_num, 2]

        actor_output_avg = actor_output.view(actor_output.size(0), -1, 2, actor_output.size(2))
        actor_output_avg = actor_output_avg.mean(dim=2)
        return actor_output_avg

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


class GNNPolicyNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super(GNNPolicyNetwork, self).__init__()
        self.gnn1 = GNNEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.gnn2 = GNNEncoder(node_dim=hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.actor_head = ActorHead(hidden_dim=hidden_dim, edge_dim=edge_dim)

        self.apply(weights_init_)
    
    def forward(self, nodes, edge_index, edge_attr, edge_type_mask):
        # nodes: [batch, node_num, node_dim]
        # edge_index: [2, edge_num]
        # edge_attr: [batch, edge_num, edge_dim]
        h = self.gnn1(nodes, edge_index, edge_attr)  # [batch, node_num, hidden_dim]
        h = self.gnn2(h, edge_index, edge_attr)  # [batch, node_num, hidden_dim]
        action_logits = self.actor_head(h, edge_index, edge_attr, edge_type_mask)  # [batch, active_edge_num, 2]
        
        mean = action_logits[:, :, 0]  # [batch, active_edge_num]
        log_std = action_logits[:, :, 1]  # [batch, active_edge_num]
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, nodes, edge_index, edge_attr, edge_type_mask):
        epsilon_tanh = 1e-6
        mean, std = self.forward(nodes, edge_index, edge_attr, edge_type_mask)
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
        return super(GNNPolicyNetwork, self).to(device)

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
    
    def push(self, observation, action, reward, next_observation, done):
        i = self.ptr

        self.observations[i] = torch.tensor(observation, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[i] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_observations[i] = torch.tensor(next_observation, dtype=torch.float32, device=self.device)
        self.dones[i] = torch.tensor([done], dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
        # batch = random.sample(self.buffer, batch_size)

    def sample(self, batch_size):
        # torch.randperm replaces torch.randint
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )


# SAC Agent class
class SACAgent:
    def __init__(self, state_dim, observation_dim, action_dim, global_obs_dim, graph_type="j", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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

        # Initialize the graph structure
        self.node_obs_dim = 6 # cap position*3, cap velocity*3
        self.global_obs_dim = global_obs_dim
        self.node_dim = self.node_obs_dim + self.global_obs_dim
        self.edge_dim = 2 # edge type, distance
        self.hidden_dim = 128

        self.node_num = 6 # 6 caps
        self.edge_type = torch.zeros((24, 1), dtype=torch.float32).to(self.device)
        self.edge_type[0:12, 0] = 1 # edge type: active tendon * 6
        self.edge_type[12:18, 0] = 2 # edge type: passive tendon * 3
        self.edge_type[18:24, 0] = 0 # edge type: bar * 3
        self.edge_type_mask = (self.edge_type[:, 0] == 1)  # mask for active edges

        if graph_type == "j":
            self.edge_index = torch.tensor([[4, 0, 0, 2, 2, 4, 5, 1, 1, 3, 3, 5, 1, 4, 0, 3, 2, 5, 0, 1, 2, 3, 4, 5],
                                            [0, 4, 2, 0, 4, 2, 1, 5, 3, 1, 5, 3, 4, 1, 3, 0, 5, 2, 1, 0, 3, 2, 5, 4]], dtype=torch.long).to(self.device)
        elif graph_type == "w":
            raise NotImplementedError("Graph type 'w' is not implemented yet.")
        else:
            raise ValueError("Unsupported graph type. Use 'j' for jonathan's configration or 'w' for will's configration.")
        
        # Networks
        self.gnn_actor = GNNPolicyNetwork(node_dim=self.node_dim, edge_dim=self.edge_dim).to(self.device)
        self.critic = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_critic = QNetwork(observation_dim, action_dim).to(self.device)
        
        # Target value network is the same as value network but with soft target updates
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr)
        self.gnn_actor_optimizer = optim.Adam(self.gnn_actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size, obs_dim=observation_dim, action_dim=action_dim, device=self.device)
        self.scaler = torch.amp.GradScaler( )
        self.precomputed_indices = {
            "active_edge_mask": (self.edge_type[:, 0] == 1),
            "edge_index": self.edge_index,
            "idx_received": self.edge_index[0],
            "idx_sent": self.edge_index[1]
        }
    
    def select_action(self, observation):
        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        nodes, edge_attr = self._obs_to_graph_input(observation)
        with torch.no_grad():
            action, _, _ = self.gnn_actor.sample(nodes, self.edge_index, edge_attr, self.edge_type_mask)
        return action.squeeze(0).cpu().numpy()
    
    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return
        
        observation_batch, action_batch, reward_batch, next_observation_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            nodes_batch, edge_attr_batch = self._obs_to_graph_input(observation_batch)
            nodes_next_batch, edge_attr_next_batch = self._obs_to_graph_input(next_observation_batch)
            
            sampled_action, action_log_prob, std = self.gnn_actor.sample(
                nodes_batch, self.edge_index, edge_attr_batch, self.edge_type_mask
            )
            
            self.alpha = torch.exp(self.log_ent_coef).detach().item()
            
            ent_coef_loss = -(self.log_ent_coef * (action_log_prob + self._target_entropy).detach()).mean()
            
            # Critic update
            with torch.no_grad():
                sampled_action_next, action_log_prob_next, _ = self.gnn_actor.sample(
                    nodes_next_batch, self.edge_index, edge_attr_next_batch, self.edge_type_mask
                )
                
                q1_target_next_pi, q2_target_next_pi = self.target_critic(next_observation_batch, sampled_action_next)
                q_target_next_pi = torch.min(q1_target_next_pi, q2_target_next_pi)
                
                next_q_value = reward_batch.view(-1, 1) + self.gamma * (1 - done_batch.view(-1, 1)) * (q_target_next_pi - self.alpha * action_log_prob_next)
            
            q1_value, q2_value = self.critic(observation_batch, action_batch)
            critic1_loss = F.mse_loss(q1_value, next_q_value)
            critic2_loss = F.mse_loss(q2_value, next_q_value)
            critic_loss = (critic1_loss + critic2_loss) / 2
            
            # Actor update
            q1_pi, q2_pi = self.critic(observation_batch, sampled_action)
            q_value_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * action_log_prob - q_value_pi).mean()
        
        self.ent_coef_optimizer.zero_grad()
        self.scaler.scale(ent_coef_loss).backward()
        self.scaler.step(self.ent_coef_optimizer)
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.scaler.step(self.critic_optimizer)
        
        self.gnn_actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.gnn_actor.parameters(), max_norm=1.0)
        self.scaler.step(self.gnn_actor_optimizer)
        
        self.scaler.update()
        
        with torch.no_grad():
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

    def _obs_to_graph_input(self, observation):
        B = observation.size(0)
        nodes = torch.empty(B, self.node_num, self.node_dim, 
                            device=observation.device)
        
        cap_pos = observation[:, :18].view(B, 6, 3)
        cap_vel = observation[:, 18:36].view(B, 6, 3)
        global_obs = observation[:, 36:].view(B, 1, -1).expand(-1, self.node_num, -1)
        
        nodes[:, :, :3] = cap_pos
        nodes[:, :, 3:6] = cap_vel
        nodes[:, :, 6:] = global_obs
        
        idx_received = self.precomputed_indices["idx_received"]
        idx_sent = self.precomputed_indices["idx_sent"]
        
        src_nodes = nodes[:, idx_received, :3]
        dst_nodes = nodes[:, idx_sent, :3]
        
        distances = torch.norm(src_nodes - dst_nodes, dim=-1, keepdim=True)
        
        edge_attr = torch.empty(B, self.edge_index.size(1), self.edge_dim, 
                            device=observation.device)
        
        edge_attr[..., 0] = self.edge_type[:, 0].view(1, -1)
        edge_attr[..., 1] = distances.squeeze(-1)  
        
        return nodes, edge_attr
    
    def save(self, path):
        
        state = {
            'gnn_actor_state_dict': self.gnn_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'ent_coef_optimizer_state_dict': self.ent_coef_optimizer.state_dict(),
            'gnn_actor_optimizer_state_dict': self.gnn_actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_ent_coef': self.log_ent_coef,
            'scaler_state_dict': self.scaler.state_dict(),
            'replay_buffer': self.replay_buffer.get_state() if hasattr(self.replay_buffer, 'get_state') else None,
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'edge_type_mask': self.edge_type_mask,
            'precomputed_indices': self.precomputed_indices
        }
        
        torch.save(state, path)
        print(f"Agent saved to {path}")
    
    def load(self, path, load_replay_buffer=False):
        
        state = torch.load(path, map_location=self.device)
        
        self.gnn_actor.load_state_dict(state['gnn_actor_state_dict'])
        self.critic.load_state_dict(state['critic_state_dict'])
        self.target_critic.load_state_dict(state['target_critic_state_dict'])
        
        self.ent_coef_optimizer.load_state_dict(state['ent_coef_optimizer_state_dict'])
        self.gnn_actor_optimizer.load_state_dict(state['gnn_actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer_state_dict'])
        
        self.log_ent_coef = state['log_ent_coef'].to(self.device)
        self.scaler.load_state_dict(state['scaler_state_dict'])
        
        self.edge_index = state['edge_index'].to(self.device)
        self.edge_type = state['edge_type'].to(self.device)
        self.edge_type_mask = state['edge_type_mask'].to(self.device)
        self.precomputed_indices = state['precomputed_indices']
        
        for key in self.precomputed_indices:
            if torch.is_tensor(self.precomputed_indices[key]):
                self.precomputed_indices[key] = self.precomputed_indices[key].to(self.device)
        
        if load_replay_buffer and state['replay_buffer'] is not None:
            self.replay_buffer.set_state(state['replay_buffer'])
            print("Replay buffer loaded successfully")
        
        print(f"Agent loaded from {path}")
        return self

    