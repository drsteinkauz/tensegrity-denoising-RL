 tmux new -s trl1 -d "python3 run.py --train --env_xml j --desired_action tracking --log_dir trl1 --model_dir trm1 --gpu_idx 0"
 sleep 1
 tmux new -s trl2 -d "python3 run.py --train --env_xml j --desired_action tracking --log_dir trl2 --model_dir trm2 --gpu_idx 1"
 sleep 1
 tmux new -s trl3 -d "python3 run.py --train --env_xml j --desired_action tracking --log_dir trl3 --model_dir trm3 --gpu_idx 2"
 sleep 1
tmux new -s trl4 -d "python3 run.py --train --env_xml j --desired_action tracking --log_dir trl4 --model_dir trm4 --gpu_idx 3"

