### Walker Multitask: Forward, Backward, Balance, Jump, Crawl from https://github.com/youngwoon/transition
import numpy as np
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env
from garage.envs import (
    GymEnv,
    TaskNameWrapper,
    TaskOnehotWrapper,
    normalize as Normalize,
)

from environments.mujoco.walker2d import Walker2dEnv
from environments.mujoco.walker2d_forward import Walker2dForwardEnv
from environments.mujoco.walker2d_backward import Walker2dBackwardEnv
from environments.mujoco.walker2d_balance import Walker2dBalanceEnv
from environments.mujoco.walker2d_jump import Walker2dJumpEnv
from environments.mujoco.walker2d_crawl import Walker2dCrawlEnv
from environments.mujoco.asset_utils import get_asset_path

WALKER2D_TASKS = [Walker2dForwardEnv, Walker2dBackwardEnv, Walker2dBalanceEnv, Walker2dCrawlEnv]

class Walker2dMTEnv(Walker2dEnv):

    def __init__(self, include_task_id=False, prob_apply_force=[0,0,0,0]):
        super().__init__()

        self._num_tasks = len(WALKER2D_TASKS)
        self._include_task_id = include_task_id
        self._prob_apply_force = prob_apply_force

        self.min_reward = 0 # Walker2d reward positive

        self._init_envs()


    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__init__(
            self._include_task_id,
            self._prob_apply_force,
        )
        return result



    def _init_envs(self):
        self._train_envs = []
        self._test_envs = []

        ### Initialize Envs
        for env_idx, env_cls in enumerate(WALKER2D_TASKS):
            # print(env_cls, env_cls())
            # train_env = TaskOnehotWrapper(GymEnv(env_cls(), max_episode_length=1000), task_index=env_idx, n_total_tasks=self._num_tasks)
            # test_env = TaskOnehotWrapper(GymEnv(env_cls(), max_episode_length=1000), task_index=env_idx, n_total_tasks=self._num_tasks)

            self._train_envs.append(env_cls(task_id=env_idx, n_total_tasks=self._num_tasks, prob_apply_force=self._prob_apply_force[env_idx]))
            self._test_envs.append(env_cls(task_id=env_idx, n_total_tasks=self._num_tasks, prob_apply_force=self._prob_apply_force[env_idx]))

        self._curr_env = self._train_envs[0]

        mujoco_env.MujocoEnv.__init__(self, get_asset_path('walker_v1.xml'), 4)
        utils.EzPickle.__init__(self, self._include_task_id, self._prob_apply_force)

    @property
    def num_tasks(self):
        return self._num_tasks

    def get_train_envs(self):
        return self._train_envs

    def get_test_envs(self):
        return self._test_envs

    def reset_model(self):
        self._curr_env = self._train_envs[self._count % self._num_tasks]
        self._count += 1
        return self._curr_env.reset_model()

    def step(self, a):
        return self._curr_env.step(a)

    def get_task_id(self, observation):
        if isinstance(observation, np.ndarray):
            id_array = np.argmax(
                observation[..., -self._num_tasks :],
                axis=-1,
            )
        else:
            id_array = torch.argmax(
                observation[..., -self._num_tasks :],
                dim=-1,
            )
        if len(id_array.shape) == 0:
            id_array = id_array[()]
        return id_array

    def split_observation(self, observation):
        obs_without_task = (
            observation.copy()
            if isinstance(observation, np.ndarray)
            else observation.clone()
        )

        if not self._include_task_id:
            ### zero out task id
            obs_without_task[...,  -self._num_tasks :] = 0

        task_info = observation

        return obs_without_task, task_info

