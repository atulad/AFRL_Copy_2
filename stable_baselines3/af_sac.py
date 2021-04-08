import time
import gym
from af.sac import AF_SAC

env = gym.make('LunarLanderContinuous-v2')

model = AF_SAC(
    'MlpPolicy',
    env,
    delta=100,
    forecast_horizon=11,
    dynamics_layers=[32, 32],
    dynamics_lr=1e-4,
    verbose=2,
    batch_size=256,
    learning_starts=1000,
    tensorboard_log='runs'
)

model.learn(
    total_timesteps=int(5e5),
    tb_log_name='AF_SAC'
)