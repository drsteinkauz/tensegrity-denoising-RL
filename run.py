import gnn_sac
import tr_env_gym

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import time
import json
from datetime import datetime

def train(env, log_dir, model_dir, lr, save_actor_only, gpu_idx=None, tb_step_recorder="False", starting_point=None):
    if gpu_idx is not None:
        device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.state_shape
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if env._desired_action == "tracking":
        global_obs_dim = 2
    else:
        global_obs_dim = 0
    agent = gnn_sac.SACAgent(state_dim=state_dim, observation_dim=observation_dim, action_dim=action_dim, global_obs_dim=global_obs_dim, graph_type=env._robot_type, device=device)

    agent.lr = lr
    
    if starting_point is not None:
        agent.load(starting_point,load_replay_buffer=True)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    TIMESTEPS = 100000
    step_num = 0
    eps_num = 0

    if tb_step_recorder == "True":
        writer = None
    else:
        writer = SummaryWriter(log_dir)

    while True:
        observation = env.reset()[0]
        episode_reward = 0
        episode_len = 0
        episode_forward_reward = 0
        episode_ctrl_reward = 0

        if tb_step_recorder == "True":
            writer = SummaryWriter(log_dir)
        else:
            actor_losses = []
            critic_losses = []
            ent_coef_losses = []
            ent_coefs = []

        start_time = time.time()

        while True:
            if step_num < agent.warmup_steps:
                action_scaled = np.random.uniform(-1, 1, size=(6,))
            else:
                action_scaled = agent.select_action(observation)
            next_observation, reward, done, _, info_env = env.step(action_scaled)
            agent.replay_buffer.push(observation, action_scaled, reward, next_observation, done)
            info_agent = agent.update()
            
            observation = next_observation
            episode_reward += reward
            episode_forward_reward += info_env["reward_forward"]
            episode_ctrl_reward += info_env["reward_ctrl"]

            step_num += 1
            episode_len += 1

            if info_agent is not None:
                if tb_step_recorder == "True":
                    writer.add_scalar("loss/actor_loss", info_agent["actor_loss"], step_num)
                    writer.add_scalar("loss/critic_loss", info_agent["critic_loss"], step_num)
                    writer.add_scalar("loss/ent_coef_loss", info_agent["ent_coef_loss"], step_num)
                    writer.add_scalar("loss/ent_coef", info_agent["ent_coef"], step_num)
                    # writer.add_scalar("loss/log_pi", info_agent["log_pi"], step_num)
                    # writer.add_scalar("loss/pi_std", info_agent["pi_std"], step_num)
                else:
                    actor_losses.append(info_agent["actor_loss"])
                    critic_losses.append(info_agent["critic_loss"])
                    ent_coef_losses.append(info_agent["ent_coef_loss"])
                    ent_coefs.append(info_agent["ent_coef"])

            if step_num % TIMESTEPS == 0:
                if save_actor_only:
                    torch.save(agent.gnn_actor.state_dict(), os.path.join(model_dir, f"actor_{step_num}.pth"))
                else:
                    agent.save(os.path.join(model_dir, f"agent_{step_num}.pth"))

            if done or episode_len >= 5000:
                break

        end_time = time.time()
        train_speed = episode_len / (end_time - start_time)
        
        eps_num += 1
        writer.add_scalar("ep/ep_rew", episode_reward, eps_num)
        writer.add_scalar("ep/ep_len", episode_len, eps_num)
        writer.add_scalar("ep/learning_rate", agent.lr, eps_num)
        writer.add_scalar("ep/train_speed", train_speed, step_num)
        if tb_step_recorder == "False":
            writer.add_scalar("loss/actor_loss", np.array(actor_losses).mean(), step_num)
            writer.add_scalar("loss/critic_loss", np.array(critic_losses).mean(), step_num)
            writer.add_scalar("loss/ent_coef_loss", np.array(ent_coef_losses).mean(), step_num)
            writer.add_scalar("loss/ent_coef", np.array(ent_coefs).mean(), step_num)
        writer.flush()
        if tb_step_recorder == "True":
            writer.close()

        print("--------------------------------")
        print(f"Episode {eps_num}")
        print(f"ep_rew: {episode_reward}")
        print(f"ep_fw_rew: {episode_forward_reward}")
        print(f"ep_ctrl_rew: {episode_ctrl_reward}")
        print(f"ep_len: {episode_len}")
        print(f"step_num: {step_num}")
        print(f"learning_rate: {agent.lr}")
        print(f"train_speed: {train_speed}")
        print("--------------------------------")
    
    writer.close()

def test(env, path_to_model, saved_data_dir, simulation_seconds, read_actor_only):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the graph structure
    node_obs_dim = 6 # cap position*3, cap velocity*3
    if env._desired_action == "tracking":
        global_obs_dim = 2
    else:
        global_obs_dim = 0
    node_dim = node_obs_dim + global_obs_dim
    edge_dim = 2 # edge type, distance
    hidden_dim = 128

    node_num = 6 # 6 caps
    edge_type = torch.zeros((24, 1), dtype=torch.float32).to(device)
    edge_type[0:12, 0] = 1 # edge type: active tendon * 6
    edge_type[12:18, 0] = 2 # edge type: passive tendon * 3
    edge_type[18:24, 0] = 0 # edge type: bar * 3
    edge_type_mask = (edge_type[:, 0] == 1)  # mask for active edges

    if env._robot_type == "j":
        edge_index = torch.tensor([[4, 0, 0, 2, 2, 4, 5, 1, 1, 3, 3, 5, 1, 4, 0, 3, 2, 5, 0, 1, 2, 3, 4, 5],
                                        [0, 4, 2, 0, 4, 2, 1, 5, 3, 1, 5, 3, 4, 1, 3, 0, 5, 2, 1, 0, 3, 2, 5, 4]], dtype=torch.long).to(device)
    elif env._robot_type == "w":
        raise NotImplementedError("Graph type 'w' is not implemented yet.")
    else:
        raise ValueError("Unsupported graph type. Use 'j' for jonathan's configration or 'w' for will's configration.")

    actor = gnn_sac.GNNPolicyNetwork(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    state_dict = torch.load(path_to_model, map_location=torch.device(device=device))
    if not read_actor_only:
        state_dict = state_dict['gnn_actor_state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    actor.load_state_dict(state_dict)
    os.makedirs(saved_data_dir, exist_ok=True)

    obs = env.reset()[0]
    done = False
    extra_steps = 500

    dt = env.dt
    actions_list = []
    tendon_length_list = []
    cap_posi_list = []
    reward_forward_list = []
    reward_ctrl_list = []
    waypt_list = []
    x_pos_list = []
    y_pos_list = []
    iter = int(simulation_seconds/dt)
    for i in range(iter):
        # Convert observation to graph input
        obs_graph_nodes, obs_graph_edge_attr = _obs_to_graph_input(torch.from_numpy(obs).float().unsqueeze(0).to(device), device, node_num, node_dim, edge_dim, edge_index, edge_type)
        # action_scaled, _ = actor.predict(torch.from_numpy(obs).float())
        action_scaled, _ = actor.forward(obs_graph_nodes, edge_index, obs_graph_edge_attr, edge_type_mask)
        action_scaled = torch.tanh(action_scaled)
        # action_unscaled = action_scaled.detach() * 0.3 - 0.15
        # action_unscaled = action_scaled.detach() * 0.05
        obs, _, done, _, info = env.step(action_scaled.detach().numpy())



        actions_list.append(action_scaled.detach().numpy().squeeze())
        #the tendon lengths are the last 9 observations
        # tendon_length_list.append(obs[-9:])
        tendon_length_list.append(info["tendon_length"])
        cap_posi_list.append(info["real_observation"][:18])
        reward_forward_list.append(info["reward_forward"])
        reward_ctrl_list.append(info["reward_ctrl"])
        waypt_list.append(info["waypt"])
        x_pos_list.append(info["x_position"])
        y_pos_list.append(info["y_position"])

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break

    action_array = np.array(actions_list)
    tendon_length_array = np.array(tendon_length_list)
    cap_posi_array = np.array(cap_posi_list)
    reward_forward_array = np.array(reward_forward_list)
    reward_ctrl_array = np.array(reward_ctrl_list)
    x_pos_array = np.array(x_pos_list)
    y_pos_array = np.array(y_pos_list)
    oript_array = np.array(env._oripoint)
    iniyaw_array = np.array([env._reset_psi])
    waypt_array = np.array(env._waypt)
    np.save(os.path.join(saved_data_dir, "action_data.npy"),action_array)
    np.save(os.path.join(saved_data_dir, "tendon_data.npy"),tendon_length_array)
    np.save(os.path.join(saved_data_dir, "cap_posi_data.npy"),cap_posi_array)
    np.save(os.path.join(saved_data_dir, "reward_forward_data.npy"),reward_forward_array)
    np.save(os.path.join(saved_data_dir, "reward_ctrl_data.npy"),reward_ctrl_array)
    np.save(os.path.join(saved_data_dir, "x_pos_data.npy"),x_pos_array)
    np.save(os.path.join(saved_data_dir, "y_pos_data.npy"),y_pos_array)
    np.save(os.path.join(saved_data_dir, "oript_data.npy"),oript_array)
    np.save(os.path.join(saved_data_dir, "iniyaw_data.npy"),iniyaw_array)
    if env._desired_action == "tracking":
        np.save(os.path.join(saved_data_dir, "waypt_data.npy"),waypt_array)

def group_test(env, path_to_model, saved_data_dir, simulation_seconds, read_actor_only, group_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the graph structure
    node_obs_dim = 6 # cap position*3, cap velocity*3
    if env._desired_action == "tracking":
        global_obs_dim = 2
    else:
        global_obs_dim = 0
    node_dim = node_obs_dim + global_obs_dim
    edge_dim = 2 # edge type, distance
    hidden_dim = 128

    node_num = 6 # 6 caps
    edge_type = torch.zeros((24, 1), dtype=torch.float32).to(device)
    edge_type[0:12, 0] = 1 # edge type: active tendon * 6
    edge_type[12:18, 0] = 2 # edge type: passive tendon * 3
    edge_type[18:24, 0] = 0 # edge type: bar * 3
    edge_type_mask = (edge_type[:, 0] == 1)  # mask for active edges

    if env._robot_type == "j":
        edge_index = torch.tensor([[4, 0, 0, 2, 2, 4, 5, 1, 1, 3, 3, 5, 1, 4, 0, 3, 2, 5, 0, 1, 2, 3, 4, 5],
                                        [0, 4, 2, 0, 4, 2, 1, 5, 3, 1, 5, 3, 4, 1, 3, 0, 5, 2, 1, 0, 3, 2, 5, 4]], dtype=torch.long).to(device)
    elif env._robot_type == "w":
        raise NotImplementedError("Graph type 'w' is not implemented yet.")
    else:
        raise ValueError("Unsupported graph type. Use 'j' for jonathan's configration or 'w' for will's configration.")

    actor = gnn_sac.GNNPolicyNetwork(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    state_dict = torch.load(path_to_model, map_location=torch.device(device=device))
    if not read_actor_only:
        state_dict = state_dict['gnn_actor_state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    actor.load_state_dict(state_dict)
    os.makedirs(saved_data_dir, exist_ok=True)

    oript_list = []
    xy_pos_list = []
    iniyaw_list = []
    if env._desired_action == "tracking":
        waypt_list = []
    for i in range(group_num):
        obs = env.reset()[0]
        done = False
        extra_steps = 500

        dt = env.dt
        iter = int(simulation_seconds/dt)
        for j in range(iter):
            # Convert observation to graph input
            obs_graph_nodes, obs_graph_edge_attr = _obs_to_graph_input(torch.from_numpy(obs).float().unsqueeze(0).to(device), device, node_num, node_dim, edge_dim, edge_index, edge_type)
            # action_scaled, _ = actor.predict(torch.from_numpy(obs).float())
            action_scaled, _ = actor.forward(obs_graph_nodes, edge_index, obs_graph_edge_attr, edge_type_mask)
            action_scaled = torch.tanh(action_scaled)
            obs, _, done, _, info = env.step(action_scaled.detach().numpy())

            if done:
                extra_steps -= 1
                if extra_steps < 0:
                    break
        oript_list.append(np.array(env._oripoint))
        xy_pos_list.append(np.array([info["x_position"], info["y_position"]]))
        iniyaw_list.append(np.array([env._reset_psi]))
        if env._desired_action == "tracking":
            waypt_list.append(np.array(env._waypt))
    oript_array = np.array(oript_list)
    xy_pos_array = np.array(xy_pos_list) - oript_array
    iniyaw_array = np.array(iniyaw_list)
    if env._desired_action == "tracking":
        waypt_array = np.array(waypt_list) - oript_array

    for i in range(group_num):
        iniyaw_ang = iniyaw_list[i][0]
        rot_mat = np.array([[np.cos(iniyaw_ang), np.sin(iniyaw_ang)],[-np.sin(iniyaw_ang), np.cos(iniyaw_ang)]])
        xy_pos_array[i] = np.dot(rot_mat, xy_pos_array[i].T).T
        if env._desired_action == "tracking":
            waypt_array[i] = np.dot(rot_mat, waypt_array[i].T).T
    np.save(os.path.join(saved_data_dir, "group_xy_pos_data.npy"),xy_pos_array)
    if env._desired_action == "tracking":
        np.save(os.path.join(saved_data_dir, "group_waypt_data.npy"),waypt_array)

def traj_test(env, path_to_model_tr, path_to_model_ccw, path_to_model_cw, saved_data_dir, simulation_seconds, read_actor_only):
    way_points = np.array([[0.0, 3.0], [3.0, 0.0], [6.0, 3.0], [6.0, 0.0]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the graph structure
    node_obs_dim = 6 # cap position*3, cap velocity*3
    tr_global_obs_dim = 2
    ccw_global_obs_dim = 0
    cw_global_obs_dim = 0
    node_dim_tr = node_obs_dim + tr_global_obs_dim
    node_dim_ccw = node_obs_dim + ccw_global_obs_dim
    node_dim_cw = node_obs_dim + cw_global_obs_dim
    edge_dim = 2 # edge type, distance
    hidden_dim = 128

    node_num = 6 # 6 caps
    edge_type = torch.zeros((24, 1), dtype=torch.float32).to(device)
    edge_type[0:12, 0] = 1 # edge type: active tendon * 6
    edge_type[12:18, 0] = 2 # edge type: passive tendon * 3
    edge_type[18:24, 0] = 0 # edge type: bar * 3
    edge_type_mask = (edge_type[:, 0] == 1)  # mask for active edges

    if env._robot_type == "j":
        edge_index = torch.tensor([[4, 0, 0, 2, 2, 4, 5, 1, 1, 3, 3, 5, 1, 4, 0, 3, 2, 5, 0, 1, 2, 3, 4, 5],
                                   [0, 4, 2, 0, 4, 2, 1, 5, 3, 1, 5, 3, 4, 1, 3, 0, 5, 2, 1, 0, 3, 2, 5, 4]], dtype=torch.long).to(device)
    elif env._robot_type == "w":
        raise NotImplementedError("Graph type 'w' is not implemented yet.")
    else:
        raise ValueError("Unsupported graph type. Use 'j' for jonathan's configration or 'w' for will's configration.")
    
    actor_tr = gnn_sac.GNNPolicyNetwork(node_dim=node_dim_tr, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    actor_ccw = gnn_sac.GNNPolicyNetwork(node_dim=node_dim_ccw, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    actor_cw = gnn_sac.GNNPolicyNetwork(node_dim=node_dim_cw, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    state_dict_tr = torch.load(path_to_model_tr, map_location=torch.device(device=device))
    state_dict_ccw = torch.load(path_to_model_ccw, map_location=torch.device(device=device))
    state_dict_cw = torch.load(path_to_model_cw, map_location=torch.device(device=device))
    if not read_actor_only:
        state_dict_tr = state_dict_tr['gnn_actor_state_dict']
        state_dict_ccw = state_dict_ccw['gnn_actor_state_dict']
        state_dict_cw = state_dict_cw['gnn_actor_state_dict']
    state_dict_tr = {k.replace("module.", ""): v for k, v in state_dict_tr.items()}
    state_dict_ccw = {k.replace("module.", ""): v for k, v in state_dict_ccw.items()}
    state_dict_cw = {k.replace("module.", ""): v for k, v in state_dict_cw.items()}
    actor_tr.load_state_dict(state_dict_tr)
    actor_ccw.load_state_dict(state_dict_ccw)
    actor_cw.load_state_dict(state_dict_cw)
    os.makedirs(saved_data_dir, exist_ok=True)

    obs = env.reset()[0]
    crt_yaw = env._reset_psi
    crt_x = env._oripoint[0]
    crt_y = env._oripoint[1]
    waypt_idx = 0
    straight_stage = False
    waypt_thresh = 0.25
    yaw_thresh = np.array([10.0, 15.0]) * np.pi / 180.0
    yaw_thresh_radius = (yaw_thresh[1] - yaw_thresh[0]) / 2.0
    yaw_thresh_center = yaw_thresh[0] + yaw_thresh_radius
    done = False
    extra_steps = 500

    op = env._oripoint
    op = op.reshape(1, 2)
    way_points = way_points + op.repeat(way_points.shape[0], axis=0)

    dt = env.dt
    actions_list = []
    tendon_length_list = []
    cap_posi_list = []
    reward_forward_list = []
    reward_ctrl_list = []
    x_pos_list = []
    y_pos_list = []
    iter = int(simulation_seconds/dt)

    for i in range(iter):
        if np.linalg.norm(way_points[waypt_idx] - np.array([crt_x, crt_y])) < waypt_thresh:
            print(f"Reached waypoint {waypt_idx} at ({crt_x}, {crt_y})")
            waypt_idx += 1
            if waypt_idx >= way_points.shape[0]:
                print("Reached the final waypoint, stopping.")
                break
            straight_stage = False
        tgt_yaw = np.arctan2(way_points[waypt_idx, 1] - crt_y, way_points[waypt_idx, 0] - crt_x)
        tgt_yaw_center = tgt_yaw - yaw_thresh_center
        tgt_yaw_center = np.arctan2(np.sin(tgt_yaw_center), np.cos(tgt_yaw_center))
        yaw_diff = tgt_yaw_center - crt_yaw
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # Normalize the angle difference to [-pi, pi]
        if yaw_diff < -np.pi/2.0 or yaw_diff > np.pi/2.0: # toward forward tracking or backward tracking
            tgt_yaw_center = tgt_yaw_center - np.pi
            tgt_yaw_center = np.arctan2(np.sin(tgt_yaw_center), np.cos(tgt_yaw_center))
            yaw_diff = tgt_yaw_center - crt_yaw
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # Normalize the angle difference to [-pi, pi]
        if yaw_diff > yaw_thresh_radius and straight_stage==False: # turn ccw
        # if yaw_diff > yaw_thresh[1]: # turn ccw
            obs_graph_nodes, obs_graph_edge_attr = _obs_to_graph_input(torch.from_numpy(obs).float().unsqueeze(0).to(device), device, node_num, node_dim_ccw, edge_dim, edge_index, edge_type)
            action_scaled, _ = actor_ccw.forward(obs_graph_nodes, edge_index, obs_graph_edge_attr, edge_type_mask)
            action_scaled = torch.tanh(action_scaled)
            obs, _, done, _, info = env.step(action_scaled.detach().numpy())
        elif yaw_diff < -yaw_thresh_radius and straight_stage==False: # turn cw
        # elif yaw_diff < yaw_thresh[0]: # turn cw
            obs_graph_nodes, obs_graph_edge_attr = _obs_to_graph_input(torch.from_numpy(obs).float().unsqueeze(0).to(device), device, node_num, node_dim_cw, edge_dim, edge_index, edge_type)
            action_scaled, _ = actor_cw.forward(obs_graph_nodes, edge_index, obs_graph_edge_attr, edge_type_mask)
            action_scaled = torch.tanh(action_scaled)
            obs, _, done, _, info = env.step(action_scaled.detach().numpy())
        else: # go straight
            # if not straight_stage:
            #     print("Switching to straight stage")
            straight_stage = True
            tracking_cmd = np.array([way_points[waypt_idx, 0] - crt_x, way_points[waypt_idx, 1] - crt_y])
            if np.linalg.norm(tracking_cmd) > 1.0:
                tracking_cmd = tracking_cmd / np.linalg.norm(tracking_cmd)
            obs[36:38] = tracking_cmd
            obs_graph_nodes, obs_graph_edge_attr = _obs_to_graph_input(torch.from_numpy(obs).float().unsqueeze(0).to(device), device, node_num, node_dim_tr, edge_dim, edge_index, edge_type)
            action_scaled, _ = actor_tr.forward(obs_graph_nodes, edge_index, obs_graph_edge_attr, edge_type_mask)
            action_scaled = torch.tanh(action_scaled)
            obs, _, done, _, info = env.step(action_scaled.detach().numpy())
        
        actions_list.append(action_scaled.detach().numpy().squeeze())
        tendon_length_list.append(info["tendon_length"])
        cap_posi_list.append(info["real_observation"][:18])
        reward_forward_list.append(info["reward_forward"])
        reward_ctrl_list.append(info["reward_ctrl"])
        crt_yaw = info["psi"]
        crt_x = info["x_position"]
        crt_y = info["y_position"]
        x_pos_list.append(info["x_position"])
        y_pos_list.append(info["y_position"])
        
        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break
    
    action_array = np.array(actions_list)
    tendon_length_array = np.array(tendon_length_list)
    cap_posi_array = np.array(cap_posi_list)
    reward_forward_array = np.array(reward_forward_list)
    reward_ctrl_array = np.array(reward_ctrl_list)
    x_pos_array = np.array(x_pos_list)
    y_pos_array = np.array(y_pos_list)
    oript_array = np.array(env._oripoint)
    iniyaw_array = np.array([env._reset_psi])
    np.save(os.path.join(saved_data_dir, "action_data.npy"),action_array)
    np.save(os.path.join(saved_data_dir, "tendon_data.npy"),tendon_length_array)
    np.save(os.path.join(saved_data_dir, "cap_posi_data.npy"),cap_posi_array)
    np.save(os.path.join(saved_data_dir, "reward_forward_data.npy"),reward_forward_array)
    np.save(os.path.join(saved_data_dir, "reward_ctrl_data.npy"),reward_ctrl_array)
    np.save(os.path.join(saved_data_dir, "x_pos_data.npy"),x_pos_array)
    np.save(os.path.join(saved_data_dir, "y_pos_data.npy"),y_pos_array)
    np.save(os.path.join(saved_data_dir, "oript_data.npy"),oript_array)
    np.save(os.path.join(saved_data_dir, "iniyaw_data.npy"),iniyaw_array)
    np.save(os.path.join(saved_data_dir, "waypt_data.npy"),way_points)


def _obs_to_graph_input(observation, device, node_num, node_dim, edge_dim, edge_index, edge_type):
        # observation: [batch, obs_dim]
        if node_dim < 6:
            raise ValueError("node_dim must be at least 6 to accommodate cap position and velocity.")
        nodes = torch.zeros(observation.shape[0], node_num, node_dim, dtype=torch.float32, device=observation.device)
        nodes[:, :, 0:3] = observation[:, :18].view(-1, 6, 3) # cap position
        nodes[:, :, 3:6] = observation[:, 18:36].view(-1, 6, 3) # cap velocity
        if node_dim > 6:
            nodes[:, :, 6:] = observation[:, 36:].unsqueeze(1).expand(-1, node_num, -1)  # global observation

        edge_attr = torch.zeros((observation.shape[0], edge_index.shape[1], edge_dim), dtype=torch.float32, device=observation.device)
        edge_type_expanded = edge_type.expand(observation.shape[0], -1, -1)
        edge_attr[:, :, 0] = edge_type_expanded[:, :, 0]
        edge_attr[:, :, 1] = torch.norm(nodes[:, edge_index[0], :3] - nodes[:, edge_index[1], :3], dim=-1)

        return nodes, edge_attr

def save_args_to_json(args, filename=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"results_{timestamp}"
    
    os.makedirs(folder_name, exist_ok=True)
    
    if filename is None:
        filename = f"config.json"
    
    filepath = os.path.join(folder_name, filename)
    
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"configs are saved to: {filepath}")
    return timestamp

# Training loop
if __name__ == "__main__":
    # Define the environment here (for example, OpenAI Gym)
    # env = gym.make("Pendulum-v0")
    
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', metavar='path_to_model')
    parser.add_argument('--group_test', metavar='path_to_model')
    parser.add_argument('--traj_test', nargs=3, metavar=('path_to_model_tr', 'path_to_model_ccw', 'path_to_model_cw'),
                        help='path to the trained model for trajectory testing, three models for straight, counterclockwise and clockwise')
    parser.add_argument('--starting_point', metavar='path_to_starting_model')
    parser.add_argument('--env_xml', default="w", type=str, choices=["w", "j", "3tr_will_normal_size.xml", "3prism_jonathan_steady_side.xml"],
                        help="ther name of the xml file for the mujoco environment, should be in same directory as run.py")
    parser.add_argument('--sb3_algo', default="SAC", type=str, choices=["SAC", "TD3", "A2C", "PPO"],
                        help='StableBaseline3 RL algorithm: SAC, TD3, A2C, PPO')
    parser.add_argument('--desired_action', default="straight", type=str, choices=["straight", "turn", "tracking", "arc"],
                        help="either straight or turn, determines what the agent is learning")
    parser.add_argument('--desired_direction', default=1, type=int, choices=[-1, 1], 
                        help="either 1 or -1, 1 means roll forward or turn counterclockwise,-1 means roll backward or turn clockwise")
    parser.add_argument('--delay', default="1", type=int, choices=[1, 10, 100],
                        help="how many steps to take in environment before updating critic\
                        Can be 1, 10, or 100. Default is 1, which worked best when training")
    parser.add_argument('--terminate_when_unhealthy', default="yes", type=str,choices=["yes", "no"],
                         help="Determines if the training is reset when the tensegrity stops moving or not, default is True.\
                            Best results are to set yes when training to move straight and set no when training to turn")
    parser.add_argument('--contact_with_self_penatly', default= 0.0, type=float,
                        help="The penalty multiplied by the total contact between bars, which is then subtracted from the reward.\
                        By default this is 0.0, meaning there is no penalty for contact.")
    # parser.add_argument('--log_dir', default="logs", type=str,
    #                     help="The directory where the training logs will be saved")
    # parser.add_argument('--model_dir', default="models", type=str,
    #                     help="The directory where the trained models will be saved")
    parser.add_argument('--saved_data_dir', default="saved_data", type=str)
    parser.add_argument('--simulation_seconds', default=30, type=int,
                         help="time in seconds to run simulation when testing, default is 30 seconds")
    parser.add_argument('--lr_SAC', default=3e-4, type=float,
                        help="learning rate for SAC, default is 3e-4")
    parser.add_argument('--gpu_idx', default=2, type=int,
                        help="index of the GPU to use, default is 2")
    parser.add_argument("--model_saving_type",default="full", type = str, choices=["full", "actor"])
    args = parser.parse_args()

    if args.terminate_when_unhealthy == "no":
        terminate_when_unhealthy = False
    else:
        terminate_when_unhealthy = True
    
    if args.env_xml == "w":
        args.env_xml = "3tr_will_normal_size.xml"
        robot_type = "w"
    elif args.env_xml == "j":
        args.env_xml = "3prism_jonathan_steady_side.xml"
        robot_type = "j"
    
    if args.model_saving_type == "full":
        save_or_read_actor_only = False
    elif args.model_saving_type == "actor":
        save_or_read_actor_only = True

    if args.train:
        timestamp = save_args_to_json(args)
        args.log_dir = f"results_{timestamp}/logs"
        args.model_dir = f"results_{timestamp}/models"

        gymenv = tr_env_gym.tr_env_gym(render_mode="None",
                                    xml_file=os.path.join(os.getcwd(),args.env_xml),
                                    robot_type=robot_type,
                                    is_test = False,
                                    desired_action = args.desired_action,
                                    desired_direction = args.desired_direction,
                                    terminate_when_unhealthy = terminate_when_unhealthy)
        if args.starting_point and os.path.isfile(args.starting_point):
            train(gymenv, args.log_dir, args.model_dir, lr=args.lr_SAC, save_actor_only=save_or_read_actor_only, gpu_idx=args.gpu_idx, starting_point=args.starting_point)
        else:
            train(gymenv, args.log_dir, args.model_dir, lr=args.lr_SAC, save_actor_only=save_or_read_actor_only, gpu_idx=args.gpu_idx)

    if(args.test):
        if os.path.isfile(args.test):
            gymenv = tr_env_gym.tr_env_gym(render_mode='human',
                                        xml_file=os.path.join(os.getcwd(),args.env_xml),
                                        robot_type=robot_type,
                                        is_test = True,
                                        desired_action = args.desired_action,
                                        desired_direction = args.desired_direction)
            test(gymenv, path_to_model=args.test, saved_data_dir=args.saved_data_dir, simulation_seconds = args.simulation_seconds, read_actor_only=save_or_read_actor_only)
        else:
            print(f'{args.test} not found.')

    if(args.group_test):
        if os.path.isfile(args.group_test):
            gymenv = tr_env_gym.tr_env_gym(render_mode='None',
                                        xml_file=os.path.join(os.getcwd(),args.env_xml),
                                        robot_type=robot_type,
                                        is_test = True,
                                        desired_action = args.desired_action,
                                        desired_direction = args.desired_direction)
            group_test(gymenv, path_to_model=args.group_test, saved_data_dir="group_test_data", simulation_seconds = args.simulation_seconds, read_actor_only = save_or_read_actor_only, group_num=32)
        else:
            print(f'{args.group_test} not found.')

    if(args.traj_test):
        if os.path.isfile(args.traj_test[0]) and os.path.isfile(args.traj_test[1]) and os.path.isfile(args.traj_test[2]):
            gymenv = tr_env_gym.tr_env_gym(render_mode='None',
                                        xml_file=os.path.join(os.getcwd(),args.env_xml),
                                        robot_type=robot_type,
                                        is_test = True,
                                        desired_action = "tracking",
                                        desired_direction = args.desired_direction)
            traj_test(gymenv, path_to_model_tr=args.traj_test[0], path_to_model_ccw=args.traj_test[1], path_to_model_cw=args.traj_test[2], saved_data_dir="traj_test_data", simulation_seconds = args.simulation_seconds, read_actor_only=save_or_read_actor_only)
        else:
            print(f'{args.traj_test} not found.')