# state, observation = env.reset()[0]
        state, observation, obs_act_seq = env.reset()[0]
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
            predict_losses = []
            predict_errors = []

        start_time = time.time()
        if env._use_stability_detection:
            stability_cnt = 0
        while True:
            if step_num < agent.warmup_steps:
                action_scaled = np.random.uniform(-1, 1, size=(6,))
            else:
                action_scaled = agent.select_action(obs_act_seq, observation, state)
            next_state, next_observation, reward, done, _, info_env = env.step(action_scaled)
            next_obs_act = np.concatenate((next_observation, action_scaled))
            next_obs_act_seq = np.concatenate((obs_act_seq[1:], next_obs_act.reshape(1, -1)), axis=0)
            agent.replay_buffer.push(state, observation, obs_act_seq, action_scaled, reward, next_state, next_observation, next_obs_act_seq, done)
            info_agent = agent.update()
            
            state = next_state
            observation = next_observation
            obs_act_seq = next_obs_act_seq
            episode_reward += reward
            episode_forward_reward += info_env["reward_forward"]
            episode_ctrl_reward += info_env["reward_ctrl"]
            
            step_num += 1
            episode_len += 1
            if env._use_stability_detection:
                stability_cnt+=state[0]
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
                    predict_losses.append(info_agent["predict_loss"])
                    predict_errors.append(info_agent["predict_error"])

            if step_num % TIMESTEPS == 0:
                torch.save(agent.actor.state_dict(), os.path.join(model_dir, f"actor_{step_num}.pth"))
                torch.save(agent.gruencoder.state_dict(), os.path.join(model_dir, f"gruencoder_{step_num}.pth"))

            if done or episode_len >= 1500:
                break

        end_time = time.time()
        train_speed = episode_len / (end_time - start_time)
        
        eps_num += 1
        writer.add_scalar("ep/ep_rew", episode_reward, eps_num)
        writer.add_scalar("ep/ep_len", episode_len, eps_num)
        writer.add_scalar("ep/RL_learning_rate", agent.lr, eps_num)
        writer.add_scalar("ep/gre_lr",agent.lr_GE,eps_num)
        writer.add_scalar("ep/train_speed", train_speed, step_num)
        if env._use_stability_detection:
            writer.add_scalar("ep/stability_rate",stability_cnt/episode_len,eps_num)
        if tb_step_recorder == "False":
            writer.add_scalar("loss/actor_loss", np.array(actor_losses).mean(), step_num)
            writer.add_scalar("loss/critic_loss", np.array(critic_losses).mean(), step_num)
            writer.add_scalar("loss/ent_coef_loss", np.array(ent_coef_losses).mean(), step_num)
            writer.add_scalar("loss/ent_coef", np.array(ent_coefs).mean(), step_num)
            writer.add_scalar("loss/predict_loss", np.array(predict_losses).mean(), step_num)
            writer.add_scalar("loss/predict_error", np.array(predict_errors).mean(), step_num)
            
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

def test(env, path_to_actor, path_to_ge, saved_data_dir, simulation_seconds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = ge_sac.PolicyNetwork(env.observation_space.shape[0]+env.inheparam_shape, env.action_space.shape[0]).to(device)
    actor_state_dict = torch.load(path_to_actor, map_location=torch.device(device=device))
    actor.load_state_dict(actor_state_dict)
    ge = ge_sac.GRUEncoder(input_dim=env.observation_space.shape[0]+env.action_space.shape[0], feature_dim=env.inheparam_shape).to(device)
    ge_state_dict = torch.load(path_to_ge, map_location=torch.device(device=device))
    ge.load_state_dict(ge_state_dict)