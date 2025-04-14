import sac
import tr_env_gym
import transformer

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from timer import Timer
torch.set_printoptions(linewidth=180)

def batched_step(env, action,batch_size,device):
    """
    Takes a batch of actions and steps the environment for each action.
    """
    next_state, next_observation, reward, done, _, info_env = zip(*[
        env[i].step(action[i]) for i in range(len(env))
    ])
    next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device)
    next_observation = torch.tensor(np.array(next_observation), dtype=torch.float32, device=device)
    reward = np.array(reward, dtype=np.float32)
    done = np.array(done, dtype=bool)
    info_env = list(info_env)
    return next_state, next_observation, reward, done, info_env

def train(env, log_dir, model_dir, lr_SAC, lr_Transformer, device, batch_size, feature_type=0):
    
    feature_list = [41,105,82]
    state_dim = env[0].state_shape+5 # 5 is the number for damping and friction
    observation_dim = feature_list[feature_type]
    action_dim = env[0].action_space.shape[0]
    # print("state_dim", state_dim)
    # print("observation_dim", observation_dim)
    # print("action_dim", action_dim)
    agent = sac.SACAgent(state_dim, observation_dim, action_dim, device=device, batch_size=batch_size)
    otf = transformer.OnlineTransformer(input_dim=18, output_dim=41, d_model=128, num_encoder_layers=7, 
                                        num_decoder_layers=7, nhead=8, dropout=0.4, batch_size=batch_size, 
                                        dim_feedforward=128,device=device)

    agent.lr = lr_SAC
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    TIMESTEPS = 2500
    step_num = 0
    eps_num = 0

    writer = SummaryWriter(log_dir, max_queue=10, flush_secs=30)
    timer = Timer()

    state, observation = zip(*(env[i].reset()[0] for i in range(batch_size)))
    #print(observation)
    state = torch.from_numpy(np.array(state)).float()
    observation = torch.from_numpy(np.array(observation)).float()
    episode_reward = np.array([0.0 for _ in range(batch_size)])
    episode_len = np.array([0 for _ in range(batch_size)])
    episode_forward_reward = np.array([0.0 for _ in range(batch_size)])
    episode_ctrl_reward = np.array([0.0 for _ in range(batch_size)])
    info_otf = otf.update(noised_input=observation, previledge=state, epoch=eps_num, learning_rate=lr_Transformer)
    feature = otf.get_feature(type_index=feature_type)
    timer.new_time_group(6)
    while True:
        
        #print(observation.shape)
        try:
            action_scaled = agent.select_action(feature)
            mask = episode_len < agent.warmup_steps
            num_replacements = np.sum(mask)
            if num_replacements > 0:
                action_scaled[mask] = np.random.uniform(-1, 1, size=(num_replacements, 6))
        except:
            print("select failed")
            action_scaled = np.random.uniform(-1, 1, size=(batch_size,6))

        timer.clip_group()
        # action_unscaled = action_scaled * 0.3 - 0.15
        action_unscaled = action_scaled * 0.05
        next_state, next_observation, reward, done, info_env = batched_step(env, action_unscaled, batch_size, device=agent.device)
        #print("next_observation",next_observation.shape,"next_state[...,:18]",next_state[...,:18].shape)
        #print(isinstance(done, np.ndarray),isinstance(next_observation, np.ndarray),isinstance(next_state, np.ndarray))
        
        timer.clip_group()

        if done.any() or torch.isnan(next_observation).any() or (episode_len >= 5000).any():
            mask = torch.isnan(next_observation).any(dim=1).cpu().numpy() | done | (episode_len >= 5000)
            print("isnan(next_observation).any(),",np.where(mask)[0]," of env restarted")
            mask_state, mask_observation = zip(*(env[i].reset()[0] for i in np.where(mask)[0]))
            mask_state = torch.from_numpy(np.array(mask_state)).float()
            mask_observation = torch.from_numpy(np.array(mask_observation)).float()
            next_state[mask,:],next_observation[mask,:] = mask_state.to(device), mask_observation.to(device)
            episode_reward[mask], episode_forward_reward[mask], episode_ctrl_reward[mask], episode_len[mask] = \
                            np.zeros(mask.sum()), np.zeros(mask.sum()), np.zeros(mask.sum()), np.zeros(mask.sum())
        
        timer.clip_group()

        info_otf = otf.update(noised_input=next_observation, previledge=next_state, epoch=eps_num, learning_rate=lr_Transformer)
        next_feature = otf.get_feature(type_index=0)
        if torch.isnan(next_feature).any():
            print("isnan(next_feature).any(), a group of env restarted")
            break

        timer.clip_group()

        agent.replay_buffer.push(state, feature, action_scaled, reward, next_state, next_feature, done)
        timer.clip_group()
        info_agent = agent.update()
        
        state = next_state
        observation = next_observation
        feature = next_feature
        for idx in range(batch_size):
            episode_reward[idx] += reward[idx]
            episode_forward_reward[idx] += info_env[idx]["reward_forward"]
            episode_ctrl_reward[idx] += info_env[idx]["reward_ctrl"]

        step_num += 1
        episode_len += 1

        if info_agent is not None:
            writer.add_scalar("loss/actor_loss", info_agent["actor_loss"], step_num)
            writer.add_scalar("loss/critic_loss", info_agent["critic_loss"], step_num)
            writer.add_scalar("loss/ent_coef_loss", info_agent["ent_coef_loss"], step_num)
            writer.add_scalar("loss/ent_coef", info_agent["ent_coef"], step_num)
            writer.add_scalar("loss/log_pi", info_agent["log_pi"], step_num)
            writer.add_scalar("loss/pi_std", info_agent["pi_std"], step_num)

        if step_num % TIMESTEPS == 0:
            torch.save(agent.actor.state_dict(), os.path.join(model_dir, f"actor_{step_num}.pth"))
            torch.save(otf.state_dict(), os.path.join(model_dir, f"transformer_{step_num}.pth"))
        
        
        eps_num += 1
        writer.add_scalar("ep/ep_rew", episode_reward.mean().item(), eps_num)
        for i, length in enumerate(episode_len):
            writer.add_scalar("ep/ep_len", length, eps_num)
        writer.add_scalar("ep/ep_f_rew", episode_forward_reward.mean().item(), eps_num)
        writer.add_scalar("ep/ep_c_rew", episode_ctrl_reward.mean().item(), eps_num)
        writer.add_scalar("ep/learning_rate", agent.lr, eps_num)
        
        timer.clip_group()

        if eps_num % 100 == 0:
            print("--------------------------------")
            print(f"Episode {eps_num}")
            print(f"avg_ep_rew: {episode_reward.mean()}")
            print(f"avg_ep_fw_rew: {episode_forward_reward.mean()}")
            print(f"avg_ep_ctrl_rew: {episode_ctrl_reward.mean()}")
            print(f"avg_ep_len: {episode_len.mean()}")
            print(f"step_num: {step_num}")
            print(f"learning_rate: {agent.lr}")
            timer.group_print(["Action_Sample", "Batched_step", "Reset", "Update_Transformer", "Replay_buffer_Push", "Renew"])
            timer.new_time_group(6)
            time100 = timer.clip_time()
            print(f"{time100:.3g}s for 100 steps; {time100*2.5/9:.3g}h for 100k steps")
            print("--------------------------------")

        

    writer.close()

    
def test(env, path_to_model, saved_data_dir, simulation_seconds):
    """
    Test is to be modified.
    """
    pass
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     actor = sac.PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
#     state_dict = torch.load(path_to_model, map_location=torch.device(device=device))
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     actor.load_state_dict(state_dict)
#     os.makedirs(saved_data_dir, exist_ok=True)

#     _, obs = env.reset()[0]
#     done = False
#     extra_steps = 500

#     dt = env.dt
#     actions_list = []
#     tendon_length_list = []
#     cap_posi_list = []
#     reward_forward_list = []
#     reward_ctrl_list = []
#     waypt_list = []
#     x_pos_list = []
#     y_pos_list = []
#     iter = int(simulation_seconds/dt)
#     for i in range(iter):
#         action_scaled, _ = actor.predict(torch.from_numpy(obs).float())
#         # action_unscaled = action_scaled.detach() * 0.3 - 0.15
#         action_unscaled = action_scaled.detach() * 0.05
#         _, obs, _, done, _, info = env.step(action_unscaled.numpy())



#         actions_list.append(action_unscaled.numpy())
#         #the tendon lengths are the last 9 observations
#         # tendon_length_list.append(obs[-9:])
#         tendon_length_list.append(info["tendon_length"])
#         cap_posi_list.append(info["real_observation"][:18])
#         reward_forward_list.append(info["reward_forward"])
#         reward_ctrl_list.append(info["reward_ctrl"])
#         waypt_list.append(info["waypt"])
#         x_pos_list.append(info["x_position"])
#         y_pos_list.append(info["y_position"])

#         if done:
#             extra_steps -= 1

#             if extra_steps < 0:
#                 break

#     action_array = np.array(actions_list)
#     tendon_length_array = np.array(tendon_length_list)
#     cap_posi_array = np.array(cap_posi_list)
#     reward_forward_array = np.array(reward_forward_list)
#     reward_ctrl_array = np.array(reward_ctrl_list)
#     waypt_array = np.array(waypt_list)
#     x_pos_array = np.array(x_pos_list)
#     y_pos_array = np.array(y_pos_list)
#     np.save(os.path.join(saved_data_dir, "action_data.npy"),action_array)
#     np.save(os.path.join(saved_data_dir, "tendon_data.npy"),tendon_length_array)
#     np.save(os.path.join(saved_data_dir, "cap_posi_data.npy"),cap_posi_array)
#     np.save(os.path.join(saved_data_dir, "reward_forward_data.npy"),reward_forward_array)
#     np.save(os.path.join(saved_data_dir, "reward_ctrl_data.npy"),reward_ctrl_array)
#     np.save(os.path.join(saved_data_dir, "waypt_data.npy"),waypt_array)
#     np.save(os.path.join(saved_data_dir, "x_pos_data.npy"),x_pos_array)
#     np.save(os.path.join(saved_data_dir, "y_pos_data.npy"),y_pos_array)


# Training loop
if __name__ == "__main__":
    # Define the environment here (for example, OpenAI Gym)
    # env = gym.make("Pendulum-v0")
    
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', metavar='path_to_model')
    parser.add_argument('--test3', metavar='path_to_model', nargs=3)
    parser.add_argument('--tracking_test', metavar='path_to_model')
    parser.add_argument('--starting_point', metavar='path_to_starting_model')
    parser.add_argument('--env_xml', default="3tr_will_normal_size.xml", type=str,
                        help="ther name of the xml file for the mujoco environment, should be in same directory as run.py")
    parser.add_argument('--sb3_algo', default="SAC", type=str, choices=["SAC", "TD3", "A2C", "PPO"],
                        help='StableBaseline3 RL algorithm: SAC, TD3, A2C, PPO')
    parser.add_argument('--desired_action', default="straight", type=str, choices=["straight", "turn", "tracking", "aiming", "vel_track"],
                        help="either straight or turn, determines what the agent is learning")
    parser.add_argument('--desired_direction', default=1, type=int, choices=[-1, 1], 
                        help="either 1 or -1, 1 means roll forward or turn counterclockwise,-1 means roll backward or turn clockwise")
    parser.add_argument('--delay', default="1", type=int, choices=[1, 10, 100],
                        help="how many steps to take in environment before updating critic\
                        Can be 1, 10, or 100. Default is 1, which worked best when training")
    parser.add_argument('--terminate_when_unhealthy', default="yes", type=str,choices=["yes", "no"],
                         help="Determines if the training is reset when the tensegrity stops moving or not, default is True.\
                            Best results are to set yes when training to move straight and set no when training to turn")
    parser.add_argument('--contact_with_self_penalty', default= 0.0, type=float,
                        help="The penalty multiplied by the total contact between bars, which is then subtracted from the reward.\
                        By default this is 0.0, meaning there is no penalty for contact.")
    parser.add_argument('--log_dir', default="logs", type=str,
                        help="The directory where the training logs will be saved")
    parser.add_argument('--model_dir', default="models", type=str,
                        help="The directory where the trained models will be saved")
    parser.add_argument('--saved_data_dir', default="saved_data", type=str)
    parser.add_argument('--simulation_seconds', default=30, type=int,
                         help="time in seconds to run simulation when testing, default is 30 seconds")
    parser.add_argument('--lr_SAC', default=3e-4, type=float,
                        help="learning rate for SAC, default is 3e-4")
    parser.add_argument('--lr_Transformer', default=1e-4, type=float,
                        help="learning rate for Transformer, default is 1e-4")
    parser.add_argument('--gpu_idx', default=2, type=int,
                        help="index of the GPU to use, default is 2")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="batch size for training, default is 32")
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=int,
                        help="device working on, default is torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")")
    args = parser.parse_args()

    if args.terminate_when_unhealthy == "no":
        terminate_when_unhealthy = False
    else:
        terminate_when_unhealthy = True

    if args.train:
        gymenv = [tr_env_gym.tr_env_gym(render_mode="None",
                                    xml_file=os.path.join(os.getcwd(),args.env_xml),
                                    is_test = False,
                                    desired_action = args.desired_action,
                                    desired_direction = args.desired_direction,
                                    terminate_when_unhealthy = terminate_when_unhealthy)for _ in range(args.batch_size)]
        
        train(gymenv, args.log_dir, args.model_dir, lr_SAC=args.lr_SAC, lr_Transformer=args.lr_Transformer, device=args.device, batch_size=args.batch_size)

    if(args.test):
        if os.path.isfile(args.test):
            gymenv = tr_env_gym.tr_env_gym(render_mode='human',
                                        xml_file=os.path.join(os.getcwd(),args.env_xml),
                                        is_test = True,
                                        desired_action = args.desired_action,
                                        desired_direction = args.desired_direction)
            test(gymenv, path_to_model=args.test, saved_data_dir=args.saved_data_dir, simulation_seconds = args.simulation_seconds)
        else:
            print(f'{args.test} not found.')