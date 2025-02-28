import numpy as np
import torch

from gym import utils
from gym.envs.mujoco import mujoco_env

from environments.mujoco.asset_utils import get_asset_path
from environments.mujoco.jaco_reach_multistage_base import (
    JacoReachMultistageEnv,
    JacoReachMTEnv,
)


class JacoReachMT5Env(JacoReachMTEnv):
    def __init__(
        self,
        include_task_id=False,
        add_action_noise=False,
    ):
        super().__init__(with_rot=True, add_action_noise=add_action_noise)

        # config
        self._config.update(
            {
                "action_penalty": 0.01,
                "velocity_penalty": 0.001,
                "acceleration_penalty": 0.00001,
                "fail_penalty": 1e2,
                "init_randomness": 0.01,
                "success_threshold": 0.05,
            }
        )

        self._sparse_tasks = [3]
        self._include_task_id = include_task_id
        self._reward_shift = [0, -2, 0, 0, 0]
        self._time_reward = [True, False, True, True, True]

        self._t = 0
        self._steps_success = 0
        self._task_id = 0
        self._stage_id = 0
        self._subgoals = [
            [0.2, 0.3, 0.5],
            [0.3, 0, 0.3],
            [0.4, -0.3, 0.4],
            [0.4, 0.3, 0.2],
        ]

        self._goal_locations = [
            [self._subgoals[i] for i in [0, 1, 2]],
            [self._subgoals[i] for i in [0, 1, 3]],
            [self._subgoals[i] for i in [1, 3, 2]],
            [self._subgoals[i] for i in [1, 2, 0]],
            [None, None, None],
        ]

        self._goal = None
        self._num_tasks = 5
        self._num_stages = 3
        self._count = 0

        self.min_reward = -5

        self._init_envs()

class JacoReachMT5StochasticEnv(JacoReachMT5Env):
    def __init__(
        self,
        include_task_id=False,
    ):
        super().__init__(include_task_id=include_task_id, add_action_noise=True)

class JacoReachMT3Env(JacoReachMTEnv):
    def __init__(
        self,
        include_task_id=False,
    ):
        super().__init__(with_rot=True)

        # config
        self._config.update(
            {
                "action_penalty": 0.01,
                "velocity_penalty": 0.001,
                "acceleration_penalty": 0.00001,
                "fail_penalty": 1e2,
                "init_randomness": 0.01,
                "success_threshold": 0.05,
            }
        )

        self._sparse_tasks = [2]
        self._include_task_id = include_task_id
        self._reward_shift = [0, -2, 0]
        self._time_reward = [True, False, True]

        self._t = 0
        self._steps_success = 0
        self._task_id = 0
        self._stage_id = 0
        self._subgoals = [
            [0.2, 0.3, 0.5],
            [0.3, 0, 0.3],
            [0.4, -0.3, 0.4],
            [0.4, 0.3, 0.2],
        ]

        self._goal_locations = [
            [self._subgoals[i] for i in [0, 1, 2]],
            [self._subgoals[i] for i in [0, 1, 3]],
            [None, None, None],
        ]

        self._goal = None
        self._num_tasks = 3
        self._num_stages = 3
        self._count = 0

        self._train_envs = []
        self._test_envs = []
        for i in range(self._num_tasks):
            self._train_envs.append(
                JacoReachMultistageEnv(
                    task_id=i,
                    num_tasks=self._num_tasks,
                    goal_locations=self._goal_locations[i],
                    sparse_reward=i in self._sparse_tasks,
                    include_task_id=self._include_task_id,
                    reward_shift=self._reward_shift[i],
                    time_reward=self._time_reward[i],
                )
            )
            self._test_envs.append(
                JacoReachMultistageEnv(
                    task_id=i,
                    num_tasks=self._num_tasks,
                    goal_locations=self._goal_locations[i],
                    sparse_reward=i in self._sparse_tasks,
                    include_task_id=self._include_task_id,
                    reward_shift=self._reward_shift[i],
                    time_reward=self._time_reward[i],
                )
            )
        self._curr_env = self._train_envs[0]

        asset_path = get_asset_path("jaco_reach_multistage.xml")
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)


class JacoReachMultistageMTEnv(JacoReachMTEnv):
    def __init__(
        self,
        sparse_tasks=[],
        include_task_id=False,
        reward_type="shift",
        reward_params=[0, 0],
        version=1,
        time_reward=True,
    ):
        super().__init__(with_rot=True)
        assert reward_type == "shift"

        # config
        self._config.update(
            {
                "action_penalty": 0.01,
                "velocity_penalty": 0.001,
                "acceleration_penalty": 0.00001,
                "fail_penalty": 1e2,
                "init_randomness": 0.01,
                "success_threshold": 0.05,
            }
        )

        self._sparse_tasks = sparse_tasks
        self._include_task_id = include_task_id

        self._t = 0
        self._steps_success = 0
        self._task_id = 0
        self._stage_id = 0
        self._subgoals = [
            [0.2, 0.3, 0.5],
            [0.3, 0, 0.3],
            [0.4, -0.3, 0.4],
            [0.4, 0.3, 0.2],
        ]
        if version == 1:
            ## Original Goals
            self._goal_locations = [
                [self._subgoals[i] for i in [0, 1, 2]],
                [self._subgoals[i] for i in [0, 1, 3]],
            ]
        elif version == 2:
            ## Shared Final Goal
            self._goal_locations = [
                [self._subgoals[i] for i in [2, 1, 3]],
                [self._subgoals[i] for i in [0, 1, 3]],
            ]
        elif version == 3:
            ## Swapped Orders
            self._goal_locations = [
                [self._subgoals[i] for i in [1, 3, 2]],
                [self._subgoals[i] for i in [0, 1, 3]],
            ]
        self._goal = None
        self._num_tasks = 2
        self._num_stages = 3
        self._count = 0

        self._train_envs = []
        self._test_envs = []
        for i in range(self._num_tasks):
            self._train_envs.append(
                JacoReachMultistageEnv(
                    task_id=i,
                    num_tasks=self._num_tasks,
                    goal_locations=self._goal_locations[i],
                    sparse_reward=i in sparse_tasks,
                    include_task_id=include_task_id,
                    reward_shift=reward_params[i],
                    time_reward=time_reward,
                )
            )
            self._test_envs.append(
                JacoReachMultistageEnv(
                    task_id=i,
                    num_tasks=self._num_tasks,
                    goal_locations=self._goal_locations[i],
                    sparse_reward=i in sparse_tasks,
                    include_task_id=include_task_id,
                    reward_shift=reward_params[i],
                    time_reward=time_reward,
                )
            )
        self._curr_env = self._train_envs[0]

        asset_path = get_asset_path("jaco_reach_multistage.xml")
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)
