import time
import gym
from af.sac import AF_SAC
from sac.sac import SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

env = gym.make('Pendulum-v0')

model = AF_SAC(
    'MlpPolicy',
    env=gym.make('Pendulum-v0'),
    delta=3,
    forecast_horizon=8,
    dynamics_layers=[32, 32],
    dynamics_lr=1e-4,
    verbose=2,
    learning_starts=1000,
    tensorboard_log='runs',
    logger_prefix='p'
)

model.learn(
    total_timesteps=60000,
    tb_log_name='P_AF_SAC'
)

