"""
Sanity check to make sure cartpole can train.
We also time this for benchmarking purposes.

Using CPU (8 cores),
- achieved 500 score in 79872 steps (~1 min)
"""

from stable_baselines3 import PPO

import string
import random
rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

model = PPO(
    'MlpPolicy', "CartPole-v1",
    tensorboard_log='results/tboard', verbose=0,
    device='cpu'
).learn(100000, tb_log_name=f'CP_PPO_{rand_id}')