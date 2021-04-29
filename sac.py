"""
Sanity check to make sure lunarlander can train.
We also time this for benchmarking purposes.

Using CPU (8 cores),
- achieved >200 score in 320 episodes / 96918 steps (~25 min)
- maxed out at ~240 score in in 515 episodes / 156215 steps (~45 min)
"""

hparams = dict(
  policy='MlpPolicy',
  learning_rate=3e-4,
  buffer_size=1000000,
  batch_size=256,
  ent_coef='auto',
  gamma=0.99,
  tau=0.01,
  train_freq=1,
  gradient_steps=1,
  learning_starts=10000,
  policy_kwargs=dict(net_arch=[400, 300]),
)

from stable_baselines3 import SAC
import gym
env = gym.make('LunarLanderContinuous-v2')
agent = SAC(env=env, device='cpu', verbose=2, **hparams)
agent.learn(total_timesteps=int(5e5))
