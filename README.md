# tensegrity-RL
A repository for training a tensegrity robot to move using reinforcement learning. The instructions for training a model fron scratch are given below, but for best results you can use the models in the "best_models_pretrained" folder without futher training. To view the results of the pretrained models, use the following commands.

```
python3 run.py --test ./best_models_pretrained/forward/SAC_5500000.zip --simulation_seconds 100
```

```
python3 run.py --test ./best_models_pretrained/backward/SAC_4700000.zip --simulation_seconds 100
```

```
python3 run.py --test ./best_models_pretrained/yaw_CCW/SAC_5000000.zip --simulation_seconds 100
```

```
python3 run.py --test ./best_models_pretrained/yaw_CW/SAC_4000000.zip --simulation_seconds 100
```

```
python3 run.py --test3 ./models_traj/SAC_16525000_track.zip ./models_traj/SAC_2175000_ccw.zip ./models_traj/SAC_1250000_cw.zip --saved_data_dir traj_saved_data --desired_action aiming --simulation_seconds 20
```



Environment: A tensegrity robot moving on either a flat plane or an uneven surface. The tensegrity robot consists of three rigid bars connected together by 6 actuated tendons and 3 unacuated tendons. 

Observations: The angular position of each rigid bar, the angular velocity of each rigid bar, the linear velocity of each rigid bar, and the length of each tendon. 

Actions: 6 actuated tendons. Actuating a tendon causes it to change length. The actuator operates in the range of -0.45 to -0.15.

Reward: (change_in_position_or_heading * desired_direction)/dt. The reward is based on the linear velocity (when the goal is to move in a straight line) or angular velocity (when the goal is to turn in place) of the tensegrity robot. The desired direction is either 1 or -1, indicating if the robot should learn to move forward (1) or backward (-1) when moving straight, or turn counterclockwise (1) or clockwise (-1) when turning in place. dt is the change in time between actions.


## Getting started 

##### Create Conda environment:

It is recommended that you create a conda environment. 

```
conda create -n "tensegrity" python=3.8.10
conda activate tensegrity
```

##### Install MuJoCo:
1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
1. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

If you want to specify a nonstandard location for the package,
use the env variable `MUJOCO_PY_MUJOCO_PATH`.

##### Installing packages:

Navigate to this repo's directory
```
pip install -r requirements.txt
```

##### Installing custom environment:

To install the custom tensegrity environment so that it can be used as a Gym environment,
```
cd tensegrity_env
pip install -e .
cd ..
```

If you make any changes to the tensegrity environment (the tensegrity_env.py or any file in the tensegrity_env directory), you must reinstall the environment.
```
cd tensegrity_env
rm -r tensegrity_env.egg-info
pip install -e .
cd ..

```

## Commands to train and test the tensegrity

The ```run.py ``` is the main Python file that will be run. Below are the following arguements for this file.
Note that "3prism_jonathan_steady_side.xml" is the default xml file for the tensegrity, and "3prism_jonathan_steady_side_uneven_ground.xml" is the same tensegrity xml file but with an uneven ground instead of a flat plane.

Arguement      | Default | Description
------------------------| ------------- | ----------
--train  | no default | Either --train or --test must be specified. --train is for training the RL model and --test is for viewing the results of the model. After --test, the path to the model to be tested must be given.
--test  | no default | Either --train or --test must be specified. --train is for training the RL model and --test is for viewing the results of the model. After --test, the path to the model to be tested must be given.
--starting_point | no default | After --starting_point, the path to a trained model must be specified. Instead of training a model from scratch, a model will be trained using the given model as a starting point
--env_xml | "3prism_jonathan_steady_side.xml" | The name of the xml file for the MuJoCo environment. This xml file must be located in the same directory as ```run.py```. 
--sb3_algo | "SAC" | The Stable Baselines3 RL algorithm. Options are "SAC", "TD3", "A2C", or "PPO".
--desired_action | "straight" | What goal the RL model is trying to accomplish. Options are "straight" or "turn"
--desired_direction | 1 | The direction the RL model is trying to move the tensegrity. Options are 1 (forward or counterclockwise) or -1 (backward or clockwise)
--delay | 1 | How many steps to take in the environment before updating the critic. Options are 1, 10, or 100, but 1 worked best
--terminate_when_unhealthy | "yes | Determines if training is reset when the tensegrity stops moving (yes) or the training continues through to the maximum step (no)
--contact_with_self_penalty | 0.0 | The penalty multiplied by the total contact between bars, which is then subtracted from the reward. |
--log_dir | "logs" | The directory where the training logs will be saved
--model_dir | "models" | The directory where the trained models will be saved
--saved_data_dir | "saved_data" | The directory where the data collected when testing the model will be saved (tendon length, contact with ground, actions)
--simulation_seconds | 30 | How many seconds the simulation should be run when testing a model
--lr_SAC | 3e-4 | learning rate for SAC
--lr_Transformer | 1e-3 | learning rate for Denoising Online Transformer
--batch_size | 16 | batch size for training. At training batch_size of env will be created
--device | torch.device("cuda" if torch.cuda.is_available() else "cpu") | device working on

To train an RL model using SAC that moves the tensegirty forward:
```
python3 run.py --train --desired_action straight --desired_direction 1 --model_dir models_forward --log_dir logs_forward
```

To train an RL model using SAC that moves the tensegirty backward:
```
python3 run.py --train --desired_action straight --desired_direction -1 --model_dir models_backward --log_dir logs_backward
```

To train an RL model using SAC that turns the tensegrity counterclockwise:
```
python3 run.py --train --desired_action turn --desired_direction 1 --terminate_when_unhealthy no --model_dir models_ccw --log_dir logs_ccw
```

To train an RL model using SAC that turns the tensegrity clockwise:
```
python3 run.py --train --desired_action turn --desired_direction -1 --terminate_when_unhealthy no --model_dir models_cw --log_dir logs_cw
```

To test an RL model (substitute in the number of the training step that you want to test the model at)

```
python3 run.py --test ./models_forward/SAC_1875000.zip --simulation_seconds 30
```

## Commands to train from a starting model

Instead of training a model from scratch, a model can be given as a starting point to train from. This is useful especially when attempting to train the tensegrity on a more difficult task. For instance, the tensegrity can be trained to move forward, then using the best forward model, the tensegrity can be trained to move forward while limiting the contact between bars. 

```
python3 run.py --train --starting_point ./models_forward/SAC_1875000.zip --desired_action straight --desired_direction 1 --model_dir models_forward --log_dir logs_forward
```


## Commands to display training data

Data is saved during training that can be viewed using Tensorboard. This includes the mean reward, the mean episode length, the actor loss, the critic loss, etc. This data can be used to determine how well the training is going and when the model is performing the best. To view this data, use the Tensorboard command and specify the log directory. 

```
tensorboard --logdir logs_forward
```


## Commands to display test run data

The actions taken, the tendon lengths, and the total bar contact is saved when a model is tested to the saved_data_dir, which by default is "saved_data". To view this data, use the commands below. The arguement ```--saved_data ``` can be used to change the default directory to look in for the saved run. 

```
python3 plot_actions.py
```

 ```
python3 plot_contact.py 
```

```
python3 plot_tendon_lengths.py
```