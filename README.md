# tensegrity-denoising-RL
To train a RL model with task "tracking":
```
python3 run.py --train --desired_action tracking --model_dir models_tracking --log_dir logs_tracking --gpu_idx 0 --env_xml w
```
To train a RL model with task "straight"
```
python3 run.py --train --desired_action straight --desired_direction 1 --model_dir models_straight --log_dir logs_straight --gpu_idx 0 --env_xml w
```
To test a trained RL model with task "tracking":
```
python3 run.py --test ./actors/actor_5425000_18nipfa5t.pth --env_xml w --desired_action tracking --simulation_seconds 20
```
To test a trained RL model with task "tracking" for several episodes to see an overall behavior:
```
python3 run.py --group_test ./actors/actor_5425000_18nipfa5t.pth --env_xml w --desired_action tracking --simulation_seconds 20
```