import time
import gym
from afrl.sac import AF_SAC

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
    q_loss_csv_filepath='results/csv/loss_p_sac.csv',
    forecast_csv_filepath='results/csv/forecast_p_sac.csv',
    device='cpu'
)

model.learn(
    total_timesteps=60000,
    tb_log_name='P_AF_SAC'
)

