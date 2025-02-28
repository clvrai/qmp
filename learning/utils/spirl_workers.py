from collections import defaultdict

import torch
import numpy as np

from garage import EpisodeBatch, StepType
from garage.experiment import deterministic
from garage.sampler import _apply_env_update
from garage.sampler.worker import Worker

from garage.sampler.default_worker import DefaultWorker
from garage.sampler.fragment_worker import FragmentWorker

from environments.kitchen.spirl.spirl_skill_decoder import load_skill_decoder
from learning.utils.general import SuppressStdout

import copy

import numpy as np

from garage import EpisodeBatch, StepType
from garage.sampler import _apply_env_update, InProgressEpisode
from garage.sampler.default_worker import DefaultWorker


class SkillDefaultWorker(DefaultWorker):

    N_STEPS_PER_ACTION = 10

    def __init__(
        self,
        *,  # Require passing by keyword, since everything's an int.
        seed,
        max_episode_length,
        worker_number,
        aggregate_infos=None,
        accumulate_reward=True,
    ):

        print("Loading SPiRL skill decoder ...")
        with SuppressStdout():
            self.model = load_skill_decoder()

        self.aggregate_infos = aggregate_infos
        self.accumulate_reward = accumulate_reward

        super().__init__(
            seed=seed,
            max_episode_length=max_episode_length,
            worker_number=worker_number,
        )

    def step_episode(self):
        ### check max_episode_length stuff
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            accumulated_reward = 0
            infos = {}

            z = torch.tensor([a], dtype=torch.float)
            last_obs = self._prev_obs  # ?? or one before?
            for i in range(self.N_STEPS_PER_ACTION):
                with torch.no_grad():
                    obs = torch.tensor(
                        [self.split_observation(last_obs)], dtype=torch.float
                    )
                    env_action = self.model.get_skill_action(obs=obs, skill=z)
                    env_action = env_action.action[0].cpu().detach().numpy()

                es = self.env.step(env_action)

                ## update last obs, accumulate reward, aggregate info, check done?
                last_obs = es.observation

                if self.accumulate_reward:
                    accumulated_reward += es.reward
                else:
                    accumulated_reward = es.reward

                if infos == {}:
                    infos = es.env_info
                else:
                    if self.aggregate_infos:
                        infos = self.aggregate_infos(infos, es.env_info)
                    else:
                        infos = es.env_info

                if es.terminal:
                    break

            es.reward = accumulated_reward
            es.env_info = infos
            es.action = a

            self._observations.append(self._prev_obs)
            ### what is env_steps
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if not es.terminal:
                self._prev_obs = last_obs
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True

    def split_observation(self, observation):
        obs_without_task = (
            observation.copy()
            if isinstance(observation, np.ndarray)
            else observation.clone()
        )

        ### HACK HACK HACK
        if obs_without_task.shape[0] != 60:
            obs_without_task = obs_without_task[..., :60]

        return obs_without_task


class SkillFragmentWorker(FragmentWorker):

    DEFAULT_N_ENVS = 8
    N_STEPS_PER_ACTION = 10

    def __init__(
        self,
        *,
        seed,
        max_episode_length,
        worker_number,
        n_envs=DEFAULT_N_ENVS,
        timesteps_per_call=1,
        aggregate_infos=None,
        accumulate_reward=True,
    ):

        print("Loading SPiRL skill decoder ...")
        with SuppressStdout():
            self.model = load_skill_decoder()

        self.aggregate_infos = aggregate_infos
        self.accumulate_reward = accumulate_reward

        super().__init__(
            seed=seed,
            max_episode_length=max_episode_length,
            worker_number=worker_number,
            n_envs=n_envs,
            timesteps_per_call=timesteps_per_call,
        )

    def step_episode(self):
        prev_obs = np.asarray([frag.last_obs for frag in self._fragments])
        actions, agent_infos = self.agent.get_actions(prev_obs)
        accumulated_rewards = np.zeros(self.n_envs)
        infos = [{} for _ in range(self.n_envs)]

        zs = torch.tensor([actions], dtype=torch.float)
        completes = [False] * len(self._envs)
        for i in range(self.N_STEPS_PER_ACTION):
            with torch.no_grad():
                obss = torch.tensor(
                    [self.split_observation(prev_obs)], dtype=torch.float
                )
                env_actions = self.model.get_skill_actions(obss=obss, skills=zs)
                env_actions = env_actions.action[0].cpu().detach().numpy()

            import ipdb

            ipdb.set_trace()
            for i, action in enumerate(env_actions):
                frag = self._fragments[i]
                if self._episode_lengths[i] < self._max_episode_length:
                    agent_info = {k: v[i] for (k, v) in agent_infos.items()}
                    frag.step(action, agent_info)
                    self._episode_lengths[i] += 1
                if (
                    self._episode_lengths[i] >= self._max_episode_length
                    or frag.step_types[-1] == StepType.TERMINAL
                ):
                    self._episode_lengths[i] = 0
                    complete_frag = frag.to_batch()
                    self._complete_fragments.append(complete_frag)
                    self._fragments[i] = InProgressEpisode(self._envs[i])
                    completes[i] = True
        if any(completes):
            self.agent.reset(completes)
        return any(completes)

    def collect_episode(self):
        """Gather fragments from all in-progress episodes.
        Returns:
            EpisodeBatch: A batch of the episode fragments.
        """
        for i, frag in enumerate(self._fragments):
            assert frag.env is self._envs[i]
            if len(frag.rewards) > 0:
                complete_frag = frag.to_batch()
                self._complete_fragments.append(complete_frag)
                self._fragments[i] = InProgressEpisode(
                    frag.env, frag.last_obs, frag.episode_info
                )
        assert len(self._complete_fragments) > 0
        result = EpisodeBatch.concatenate(*self._complete_fragments)
        self._complete_fragments = []
        return result

    def split_observation(self, observation):
        obs_without_task = (
            observation.copy()
            if isinstance(observation, np.ndarray)
            else observation.clone()
        )

        ### HACK HACK HACK
        if obs_without_task.shape[0] != 60:
            obs_without_task = obs_without_task[..., :60]

        return obs_without_task
