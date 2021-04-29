import time
import gym
from afrl.dqn import AF_DQN
from stable_baselines3 import DQN

env = gym.make('CartPole-v1')

model = AF_DQN('MlpPolicy',
            env,
            delta=0.1,
            forecast_horizon=11,
            dynamics_layers=[32, 32],
            dynamics_lr=1e-4,
            verbose=2,
            learning_rate=1e-3,
            buffer_size=5000,
            batch_size=32,
            learning_starts=0,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            exploration_initial_eps=0.8,
            tau=0.1,
            gamma=0.9,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10,
            tensorboard_log='runs',
            device='cpu'
)

model.learn(
    total_timesteps=1e5,
    tb_log_name='DQN'
)

from stable_baselines3.common.evaluation import evaluate_policy
eval_env = gym.make('CartPole-v1')
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')