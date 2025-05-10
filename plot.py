import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = 'logs_forward'  
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

scalar_tags = ea.Tags()['scalars']
print(scalar_tags)
for tag in scalar_tags:
    scalar_data = ea.Scalars(tag)
    steps = [x.step for x in scalar_data]
    values = [x.value for x in scalar_data]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=tag)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f'{tag} over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('img.jpg')