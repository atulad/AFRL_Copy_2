from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn
from torch import FloatTensor as ft
from torch.nn import functional as F
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from stable_baselines3.common.type_aliases import GymEnv, Schedule, RolloutReturn, TrainFreq, MaybeCallback
from .af import DynamicsModel

class AF_DQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        ### AF PARAMS BEGIN
        delta: float,
        forecast_horizon: int,
        dynamics_layers: List[int],
        dynamics_lr: float = 1e-4,
        ### AF PARAMS END
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

        # AF
        self.delta = delta
        self.forecast_horizon = forecast_horizon
        self.dynamics = DynamicsModel(
            env.observation_space.shape[0], 1,
            dynamics_layers, dynamics_lr, self.device)

        self.zero_forecasts = np.zeros(self.forecast_horizon, np.int8)
        self.empty_plan = list

    def train_dynamics_model(self, gradient_steps: int, s: th.Tensor, a: th.Tensor, s2: th.Tensor):
        losses = []
        for _ in range(gradient_steps):
            # Compute Huber loss (less sensitive to outliers)
            loss = ((self.dynamics.predict(s, a) - s2)**2).mean()

            # Optimize the dynamics model
            self.dynamics.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.max_grad_norm)
            self.dynamics.optimizer.step()

            losses.append(loss.cpu().detach().numpy())

        with open('dyna_loss.csv', 'a') as f:
            f.write(f'{self.num_timesteps}, {self._episode_num}, {np.mean(losses)}\n')

    def _q_val(self, state: np.ndarray, action: int):
        with th.no_grad():
            return self.critic(ft(state.reshape(1, -1)).cuda()).cpu().numpy()[0][action]

    def _replan(self, state, plan, forecast):
        new_plan = np.empty(self.forecast_horizon, dtype=np.int8)
        k = 0 # used to keep track of forecast of the actions

        # recycle old plan
        for action in plan[1:]:
            with th.no_grad():
                best_action = int(self.predict(state, deterministic=False)[0])
                replan_q = self._q_val(state, best_action)
                plan_q = self._q_val(state, action)
                if replan_q > self.delta + plan_q:
                    break
                new_plan[k] = action
                state = self.dynamics(state, action)
            forecast[k] = forecast[k+1] + 1 # update the forecast of this action
            k += 1

        # produce new plan
        for i in range(k, self.forecast_horizon):
            action = self.predict(state, deterministic=False)[0].item()
            new_plan[i] = action
            with th.no_grad():
                state = self.dynamics(state, action)
            forecast[i] = 0

        return new_plan, forecast

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        s = replay_data.observations
        a = replay_data.actions
        s2 = replay_data.next_observations
        # use the same param gradient_steps for both training dynamics model and q-net
        self.train_dynamics_model(gradient_steps, s, a, s2)
        super().train(gradient_steps, batch_size)

    def _af_on_done(self):
        self.plan, self.forecasts = self._replan(self._last_obs[0], self.empty_plan(), self.zero_forecasts) # _last_obs auto reset when done
        self.episode_forecasts.append(
            np.mean(self.episode_forecast[self.forecast_horizon:])
            if len(self.episode_forecast) > self.forecast_horizon
            else np.mean(self.episode_forecast))
        self.episode_forecast = []
        with open('forecast.csv', 'a') as f:
            f.write(f'{self.num_timesteps}, {self._episode_num}, {self.episode_forecasts[-1]}\n')

    """
    __MODIFICATIONS__
    collect_rollouts: change action to get from plan and add call _af_on_done when done
    learn: init forecast tracker and plan (3 lines)
    """
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # AF
                action = self.plan[0]
                self.episode_forecast.append(self.forecasts[0])
                buffer_action = action # sb needs this
                # AF END

                # Rescale and perform action
                new_obs, reward, done, infos = env.step([action])

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

                # AF
                self.plan, self.forecasts = self._replan(self._last_obs[0], self.plan, self.forecasts) # self._last_obs was updated in _store_transition

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

                self._af_on_done()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        # AF (could put this in callback.on_training_start)
        self.episode_forecast = []
        self.episode_forecasts = []
        self.plan, self.forecasts = self._replan(self._last_obs[0], self.empty_plan(), self.zero_forecasts) # VecEnv resets automatically
        # AF END

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self
