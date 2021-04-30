"""
Using CPU (8 cores),
- achieved >200 score in 268 episodes / 98175 steps (~32 min)
"""
import gym
from afrl.sac import AF_SAC

import string
import random
rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

env = gym.make('Pendulum-v0')

delta = 0.95

model = AF_SAC(
    'MlpPolicy',
    env=gym.make('LunarLanderContinuous-v2'),
    delta=delta,
    forecast_horizon=8,
    dynamics_layers=[32, 32],
    dynamics_lr=1e-4,
    verbose=0,
    learning_starts=1000,
    tensorboard_log='results/tboard',
    device='cpu',
    batch_size=256,
)

model.learn(
    total_timesteps=200_000,
    tb_log_name=f'P_AF_SAC_delta={delta}_{rand_id}'
)
