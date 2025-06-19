import sac
import tr_env_gym

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
import gym

def train(env, log_dir, model_dir, lr, gpu_idx=None, tb_step_recorder="False", starting_point=None,algo="A2C"):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import BaseCallback
    
    vec_env = DummyVecEnv([lambda: env])
    
    if gpu_idx is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class TensorboardCallback(BaseCallback):
        def __init__(self, log_dir, tb_step_recorder, algo, verbose=0):
            super().__init__(verbose)
            self.log_dir = log_dir
            self.tb_step_recorder = tb_step_recorder
            self.algo = algo
            self.writer = None
            self.actual_log_dir = None
            self.max_episode_len = 5000  
            
            self.episode_reward = None
            self.episode_len = None
            self.episode_forward_reward = None
            self.episode_ctrl_reward = None
            self.step_num = None
            self.eps_num = None
            self.actor_losses = None
            self.critic_losses = None
            self.ent_coef_losses = None
            self.ent_coefs = None
            self.start_time = None
            self.last_reward = None
            self.last_info = None
            self.force_done = False  
            
        def _on_training_start(self) -> None:
            self.actual_log_dir = self.model.logger.get_dir()
            self.writer = SummaryWriter(self.actual_log_dir)
            
            self.episode_reward = 0
            self.episode_len = 0
            self.episode_forward_reward = 0
            self.episode_ctrl_reward = 0
            self.step_num = 0
            self.eps_num = 0
            self.actor_losses = []
            self.critic_losses = []
            self.ent_coef_losses = []
            self.ent_coefs = []
            self.start_time = time.time()
            self.last_reward = 0
            self.last_info = {}
            self.force_done = False
        
        def _on_step(self) -> bool:
            if self.writer is None:
                return True
                
            if self.episode_len >= self.max_episode_len and not self.force_done:
                print(f"Episode length reached maximum ({self.max_episode_len} steps), forcing episode end")
                self.force_done = True
                
                if hasattr(self.locals, 'dones'):
                    self.locals['dones'][0] = True
                elif 'dones' in self.locals:
                    self.locals['dones'][0] = True
                
                self._on_episode_end()
                return True
                
            if self.force_done:
                return True
                
            rewards = self.locals.get('rewards')
            infos = self.locals.get('infos')
            
            if rewards is not None and len(rewards) > 0:
                self.last_reward = rewards[0]
                
            if infos is not None and len(infos) > 0:
                self.last_info = infos[0]
            
            self.episode_reward += self.last_reward
            self.episode_len += 1
            
            if "reward_forward" in self.last_info:
                self.episode_forward_reward += self.last_info["reward_forward"]
            if "reward_ctrl" in self.last_info:
                self.episode_ctrl_reward += self.last_info["reward_ctrl"]
            
            self.step_num += 1
            
            dones = self.locals.get('dones')
            if dones is not None and len(dones) > 0 and dones[0]:
                self._on_episode_end()
                    
            return True
        
        def _on_episode_end(self):
            end_time = time.time()
            train_speed = self.episode_len / (end_time - self.start_time)
            
            self.eps_num += 1
            self.writer.add_scalar("ep/ep_rew", self.episode_reward, self.eps_num)
            self.writer.add_scalar("ep/ep_len", self.episode_len, self.eps_num)
            self.writer.add_scalar("ep/learning_rate", self.model.learning_rate, self.eps_num)
            self.writer.add_scalar("ep/train_speed", train_speed, self.step_num)
            self.writer.add_scalar("ep/ep_fw_rew", self.episode_forward_reward, self.eps_num)
            self.writer.add_scalar("ep/ep_ctrl_rew", self.episode_ctrl_reward, self.eps_num)
            
            if self.tb_step_recorder == "False":
                if self.actor_losses:
                    self.writer.add_scalar("loss/actor_loss", np.array(self.actor_losses).mean(), self.step_num)
                if self.critic_losses:
                    self.writer.add_scalar("loss/critic_loss", np.array(self.critic_losses).mean(), self.step_num)
                if self.ent_coef_losses:
                    self.writer.add_scalar("loss/ent_coef_loss", np.array(self.ent_coef_losses).mean(), self.step_num)
                if self.ent_coefs:
                    self.writer.add_scalar("loss/ent_coef", np.array(self.ent_coefs).mean(), self.step_num)
            
            self.writer.flush()
            
            print("--------------------------------")
            print(f"Episode {self.eps_num} ({self.algo})")
            print(f"ep_rew: {self.episode_reward}")
            print(f"ep_fw_rew: {self.episode_forward_reward}")
            print(f"ep_ctrl_rew: {self.episode_ctrl_reward}")
            print(f"ep_len: {self.episode_len}")
            print(f"step_num: {self.step_num}")
            print(f"learning_rate: {self.model.learning_rate}")
            print(f"train_speed: {train_speed} steps/sec")
            
            if self.force_done:
                print(f"NOTE: Episode was force ended at {self.episode_len} steps (max allowed: {self.max_episode_len})")
            
            print("--------------------------------")
            
            self.episode_reward = 0
            self.episode_len = 0
            self.episode_forward_reward = 0
            self.episode_ctrl_reward = 0
            self.actor_losses = []
            self.critic_losses = []
            self.ent_coef_losses = []
            self.ent_coefs = []
            self.start_time = time.time()
            self.force_done = False 
        
        def _on_rollout_end(self) -> None:
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                logs = self.model.logger.name_to_value
                if "train/actor_loss" in logs:
                    self.actor_losses.append(logs["train/actor_loss"])
                if "train/critic_loss" in logs:
                    self.critic_losses.append(logs["train/critic_loss"])
                if "train/ent_coef_loss" in logs:
                    self.ent_coef_losses.append(logs["train/ent_coef_loss"])
                if "train/ent_coef" in logs:
                    self.ent_coefs.append(logs["train/ent_coef"])
    
    def close(self):
        """关闭写入器"""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
    
    algo_log_dir = os.path.join(log_dir, algo.lower())
    algo_model_dir = os.path.join(model_dir, algo.lower())
    os.makedirs(algo_model_dir, exist_ok=True)
    
    custom_callback = TensorboardCallback(algo_log_dir, tb_step_recorder, algo)
    
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
    
    if algo == "A2C":
        model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_rms_prop=True,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            device=device,
            verbose=0
        )
    elif algo == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            device=device,
            verbose=0
        )
    elif algo == "TD3":
        model = TD3(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=100,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            device=device,
            verbose=0
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    
    model.learn(
            total_timesteps=10_000_000,
            callback=custom_callback,
            tb_log_name=f"{algo.lower()}_training",
            progress_bar=False
    )
    custom_callback.writer.close()
    
    model.save(os.path.join(model_dir, f"{algo.lower()}_final_model"))
    

def test(env, path_to_model, saved_data_dir, simulation_seconds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = sac.PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    state_dict = torch.load(path_to_model, map_location=torch.device(device=device))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    actor.load_state_dict(state_dict)
    os.makedirs(saved_data_dir, exist_ok=True)

    _, obs, _, _ = env.reset()[0]
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
        # action_scaled, _ = actor.predict(torch.from_numpy(obs).float())
        action_scaled, _ = actor.forward(torch.from_numpy(obs).float())
        action_scaled = torch.tanh(action_scaled)
        # action_unscaled = action_scaled.detach() * 0.3 - 0.15
        action_unscaled = action_scaled.detach() * 0.05
        _, obs, _, _, done, _, info = env.step(action_scaled.detach().numpy())



        actions_list.append(action_scaled.detach().numpy())
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

def group_test(env, path_to_model, saved_data_dir, simulation_seconds, group_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = sac.PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    state_dict = torch.load(path_to_model, map_location=torch.device(device=device))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    actor.load_state_dict(state_dict)
    os.makedirs(saved_data_dir, exist_ok=True)

    oript_list = []
    xy_pos_list = []
    iniyaw_list = []
    if env._desired_action == "tracking":
        waypt_list = []
    for i in range(group_num):
        _, obs, _, _ = env.reset()[0]
        done = False
        extra_steps = 500

        dt = env.dt
        iter = int(simulation_seconds/dt)
        for j in range(iter):
            # action_scaled, _ = actor.predict(torch.from_numpy(obs).float())
            action_scaled, _ = actor.forward(torch.from_numpy(obs).float())
            action_scaled = torch.tanh(action_scaled)
            _, obs, _, _, done, _, info = env.step(action_scaled.detach().numpy())

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

# Training loop
if __name__ == "__main__":
    # Define the environment here (for example, OpenAI Gym)
    # env = gym.make("Pendulum-v0")
    
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', metavar='path_to_model')
    parser.add_argument('--group_test', metavar='path_to_model')
    parser.add_argument('--starting_point', metavar='path_to_starting_model')
    parser.add_argument('--env_xml', default="w", type=str, choices=["w", "j", "3tr_will_normal_size.xml", "3prism_jonathan_steady_side.xml"],
                        help="ther name of the xml file for the mujoco environment, should be in same directory as run.py")
    parser.add_argument('--sb3_algo', default="SAC", type=str, choices=["SAC", "TD3", "A2C", "PPO"],
                        help='StableBaseline3 RL algorithm: SAC, TD3, A2C, PPO')
    parser.add_argument('--desired_action', default="straight", type=str, choices=["straight", "turn", "tracking", "vel_track", "arc"],
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
    parser.add_argument('--log_dir', default="logs", type=str,
                        help="The directory where the training logs will be saved")
    parser.add_argument('--model_dir', default="models", type=str,
                        help="The directory where the trained models will be saved")
    parser.add_argument('--saved_data_dir', default="saved_data", type=str)
    parser.add_argument('--simulation_seconds', default=30, type=int,
                         help="time in seconds to run simulation when testing, default is 30 seconds")
    parser.add_argument('--lr_SAC', default=3e-4, type=float,
                        help="learning rate for SAC, default is 3e-4")
    parser.add_argument('--gpu_idx', default=2, type=int,
                        help="index of the GPU to use, default is 2")
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

    if args.train:
        gymenv = tr_env_gym.tr_env_gym(render_mode="None",
                                    xml_file=os.path.join(os.getcwd(),args.env_xml),
                                    robot_type=robot_type,
                                    is_test = False,
                                    desired_action = args.desired_action,
                                    desired_direction = args.desired_direction,
                                    terminate_when_unhealthy = terminate_when_unhealthy)
        if args.starting_point and os.path.isfile(args.starting_point):
            train(gymenv, args.log_dir, args.model_dir, 
                lr=args.lr_SAC, gpu_idx=args.gpu_idx, 
                starting_point=args.starting_point,
                algo=args.sb3_algo)  
        else:
            train(gymenv, args.log_dir, args.model_dir, 
                lr=args.lr_SAC, gpu_idx=args.gpu_idx,
                algo=args.sb3_algo)  

    if(args.test):
        if os.path.isfile(args.test):
            gymenv = tr_env_gym.tr_env_gym(render_mode='human',
                                        xml_file=os.path.join(os.getcwd(),args.env_xml),
                                        robot_type=robot_type,
                                        is_test = True,
                                        desired_action = args.desired_action,
                                        desired_direction = args.desired_direction)
            test(gymenv, path_to_model=args.test, saved_data_dir=args.saved_data_dir, simulation_seconds = args.simulation_seconds)
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
            group_test(gymenv, path_to_model=args.group_test, saved_data_dir="group_test_data", simulation_seconds = args.simulation_seconds, group_num=32)
        else:
            print(f'{args.group_test} not found.')