from garage.envs import (
    GymEnv,
    TaskNameWrapper,
    TaskOnehotWrapper,
    normalize as Normalize,
)
import gym
from gym.spaces import Box
import torch
import numpy as np
from collections import OrderedDict

from environments.kitchen import (
    KITCHEN_MT10,
    KITCHEN_MT1,
    KITCHEN_MT3,
    KITCHEN_MT3_v2,
    KITCHEN_MT3_v3,
    KITCHEN_HARD,
    KITCHEN_MT5,
    KITCHEN_MT4,
    KITCHEN_MT2,
    KITCHEN_ALL,
    KITCHEN_CABINET,
    KITCHEN_MICROWAVE,
    KITCHEN_MT_EASY,
    KITCHEN_MT_EASY5,
)

from environments.kitchen.kitchen_spirl import SkillWrapper
from environments.kitchen.spirl.spirl_skill_decoder import load_skill_decoder
from environments.kitchen.kitchen_multistage import KitchenMultistageEnv

from learning.utils.general import SuppressStdout

### SPiRL Kitchen env
from d4rl.kitchen import KitchenMicrowaveKettleBottomBurnerLightV0
from environments.kitchen.v0.kitchen_tasks import KitchenMicrowaveKettleBottomBurnerV0


class KitchenMultiTaskEnv(gym.Env):
    def __init__(
        self,
        benchmark_type,
        task_name=None,
        sparse_tasks=[],
        include_task_id=False,
        normalize=False,
        terminate_on_success=True,
        max_episode_steps=70,
        control_penalty=0.0,
        use_skill_space=False,
        accumulate_reward=True,
        vectorized_skills=False,
    ):
        ### HACK just checking
        assert normalize is False and control_penalty == 0
        self._include_task_id = include_task_id
        self._normalize = normalize
        self._terminate_on_success = terminate_on_success
        self._use_skill_space = use_skill_space
        self._vectorized_skills = vectorized_skills

        if self._use_skill_space:  # and not self._vectorized_skills:
            ### Load spirl skill model here
            print("Loading SPiRL skill decoder...")
            with SuppressStdout():
                self._skill_model = load_skill_decoder()

        if benchmark_type == "MT10":
            self._num_tasks = 10
            self._benchmark = KITCHEN_MT10
        elif benchmark_type == "MT1":
            assert task_name in KITCHEN_MT1
            self._num_tasks = 1
            self._benchmark = KITCHEN_MT1[task_name]
        elif benchmark_type == "MT3":
            self._num_tasks = 3
            self._benchmark = KITCHEN_MT3
        elif benchmark_type == "MT3_v2":
            self._num_tasks = 3
            self._benchmark = KITCHEN_MT3_v2
        elif benchmark_type == "MT3_v3":
            self._num_tasks = 3
            self._benchmark = KITCHEN_MT3_v3
        elif benchmark_type == "HARD":
            self._num_tasks = 3
            self._benchmark = KITCHEN_HARD
        elif benchmark_type == "MT5":
            self._num_tasks = 5
            self._benchmark = KITCHEN_MT5
        elif benchmark_type == "MT2":
            self._num_tasks = 2
            self._benchmark = KITCHEN_MT2
        elif benchmark_type == "MT4":
            self._num_tasks = 4
            self._benchmark = KITCHEN_MT4
        elif benchmark_type == "MIXED":
            self._num_tasks = 1
            self._benchmark = OrderedDict(
                (
                    (
                        "microwave-kettle-bottom-burner",
                        KitchenMicrowaveKettleBottomBurnerV0,
                    ),
                )
            )
        elif benchmark_type == "ALL":
            self._num_tasks = 14
            self._benchmark = KITCHEN_ALL
        elif benchmark_type == "CABINET":
            self._num_tasks = 4
            self._benchmark = KITCHEN_CABINET
        elif benchmark_type == "MICROWAVE":
            self._num_tasks = 2
            self._benchmark = KITCHEN_MICROWAVE
        elif benchmark_type == "MT_EASY":
            self._num_tasks = 10
            self._benchmark = KITCHEN_MT_EASY
        elif benchmark_type == "MT_EASY5":
            self._num_tasks = 5
            self._benchmark = KITCHEN_MT_EASY5
        else:
            raise NotImplementedError()

        self._sparse_tasks = sparse_tasks
        assert all(0 <= t < self._num_tasks for t in self._sparse_tasks)

        def _make_env(env_name, env_idx, train=True):
            env_cls = self._benchmark[env_name]
            env = env_cls(
                sparse_reward=env_idx in self._sparse_tasks,
                terminate_on_success=self._terminate_on_success,
                control_penalty=control_penalty,
                early_termination_bonus=max_episode_steps,
            )
            if self._use_skill_space:
                if not train or not self._vectorized_skills:
                    env = SkillWrapper(
                        env=env,
                        model=self._skill_model,
                        aggregate_infos=getattr(env, "aggregate_infos", None),
                        accumulate_reward=accumulate_reward,
                    )
                else:
                    env.action_space = Box(
                        low=np.array([-2] * 10), high=np.array([2] * 10)
                    )  ### Fix this --> for uniform prior is ok?
            env = GymEnv(env, max_episode_length=max_episode_steps)
            env = TaskNameWrapper(env, task_name=env_name, task_id=env_idx)
            env = TaskOnehotWrapper(
                env, task_index=env_idx, n_total_tasks=self._num_tasks
            )
            if self._normalize:
                env = Normalize(env)
            return env

        self._train_envs = []
        self._test_envs = []
        for idx, env_name in enumerate(self._benchmark.keys()):
            self._train_envs.append(_make_env(env_name, idx, train=True))
            self._test_envs.append(_make_env(env_name, idx, train=False))
        self._sample_env = self._train_envs[0]
        self._max_episode_steps = self._sample_env.spec.max_episode_length
        self.metadata = {  # support different versions of gym
            "render.modes": ["human", "rgb_array", True, False],
            "render_modes": ["human", "rgb_array", True, False],
            "video.frames_per_second": int(np.round(1.0 / self._sample_env.dt)),
        }
        self.min_reward = -20 # -1 * reward scale (20)

        if self._include_task_id:
            self.observation_space = self._sample_env.observation_space
        else:
            task_id_obs = self._sample_env.observation_space
            self.observation_space = Box(
                high=task_id_obs.high[: -self._num_tasks],
                low=task_id_obs.low[: -self._num_tasks],
                dtype=task_id_obs.dtype,
            )

    @property
    def action_space(self):
        return self._sample_env.action_space

    @property
    def num_tasks(self):
        return self._num_tasks

    def get_train_envs(self):
        return self._train_envs[0] if self._num_tasks < 2 else self._train_envs

    def get_test_envs(self):
        return self._test_envs[0] if self._num_tasks < 2 else self._test_envs

    def get_task_id(self, observation):
        if isinstance(observation, np.ndarray):
            id_array = np.argmax(observation[..., -self._num_tasks :], axis=-1)
        else:
            id_array = torch.argmax(observation[..., -self._num_tasks :], dim=-1)
        if len(id_array.shape) == 0:
            id_array = id_array[()]
        return id_array

    def get_stage_id(self, observation):
        # always returns 0
        if self._use_skill_space:
            raise NotImplementedError
        id_array = observation[..., -1]
        if isinstance(id_array, np.ndarray):
            id_array = id_array.astype(int)
        else:
            id_array = id_array.int()
        id_array[...] = 0

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
            obs_without_task = obs_without_task[..., : -self._num_tasks]

        task_info = observation

        return obs_without_task, task_info


class KitchenMixed(gym.Env):
    def __init__(
        self,
        task_name=None,
        sparse_tasks=[],
        include_task_id=False,
        normalize=False,
        terminate_on_success=True,
        max_episode_steps=70,
        control_penalty=0.0,
        use_skill_space=False,
        accumulate_reward=True,
    ):
        # self._sparse_tasks = sparse_tasks
        # assert all(0 <= t < self._num_tasks for t in self._sparse_tasks)
        # self._include_task_id = include_task_id
        # self._normalize = normalize
        # self._terminate_on_success = terminate_on_success
        # self._use_skill_space = use_skill_space
        assert use_skill_space
        n_steps_per_action = 10
        ### Load spirl skill model here
        self._skill_model = load_skill_decoder()
        self._num_tasks = 1

        def _make_env(env_idx):
            env = KitchenMicrowaveKettleBottomBurnerLightV0(
                ref_min_score=0.0,
                ref_max_score=4.0,
                # max_episode_steps=max_episode_steps * n_steps_per_action,
            )
            env = SkillWrapper(
                env=env,
                model=self._skill_model,
                aggregate_infos=getattr(env, "aggregate_infos", None),
                accumulate_reward=accumulate_reward,
            )
            env = GymEnv(env, max_episode_length=max_episode_steps)
            env = TaskNameWrapper(
                env, task_name="microwave kettle bottom burner light", task_id=env_idx
            )
            return env

        self._train_envs = []
        self._test_envs = []
        for idx in range(self._num_tasks):
            self._train_envs.append(_make_env(idx))
            self._test_envs.append(_make_env(idx))
        self._sample_env = self._train_envs[0]
        self._max_episode_steps = self._sample_env.spec.max_episode_length
        self.metadata = {  # support different versions of gym
            "render.modes": ["human", "rgb_array", True, False],
            "render_modes": ["human", "rgb_array", True, False],
            "video.frames_per_second": int(np.round(1.0 / self._sample_env.dt)),
        }

    @property
    def action_space(self):
        return self._sample_env.action_space

    @property
    def observation_space(self):
        return self._sample_env.observation_space

    @property
    def num_tasks(self):
        return self._num_tasks

    def get_train_envs(self):
        return self._train_envs[0] if self._num_tasks < 2 else self._train_envs

    def get_test_envs(self):
        return self._test_envs[0] if self._num_tasks < 2 else self._test_envs

    def get_task_id(self, observation):
        return 0

    def get_stage_id(self, observation):
        raise NotImplementedError

    def split_observation(self, observation):
        return observation, observation


class KitchenMultiTaskMultistageEnv(KitchenMultiTaskEnv):
    def __init__(
        self,
        task_names,
        sparse_tasks=[],
        include_task_id=False,
        normalize=False,
        terminate_on_success=True,
        max_episode_steps=70,
        control_penalty=0.0,
        use_skill_space=False,
        accumulate_reward=True,
    ):
        self._num_tasks = len(task_names)
        self._num_stages = len(task_names[0])
        self._sparse_tasks = sparse_tasks
        assert all(0 <= t < self._num_tasks for t in self._sparse_tasks)
        self._include_task_id = include_task_id
        self._normalize = normalize
        self._terminate_on_success = terminate_on_success
        self._use_skill_space = use_skill_space

        if self._use_skill_space:
            n_steps_per_action = 10
            ### Load spirl skill model here
            self._skill_model = load_skill_decoder()
        else:
            n_steps_per_action = 1

        def _make_multistage_env(env_idx):
            env_cls = KitchenMultistageEnv
            env = env_cls(
                task_names=task_names[env_idx],
                sparse_reward=env_idx in self._sparse_tasks,
                terminate_on_success=self._terminate_on_success,
                control_penalty=control_penalty,
                max_episode_steps=max_episode_steps * n_steps_per_action,
            )
            if self._use_skill_space:
                env = SkillWrapper(
                    env=env,
                    model=self._skill_model,
                    aggregate_infos=env.aggregate_infos,
                    accumulate_reward=accumulate_reward,
                )

            env = GymEnv(env, max_episode_length=max_episode_steps)
            env_name = ", ".join(task_names[env_idx])
            env = TaskNameWrapper(env, task_name=env_name, task_id=env_idx)
            ### HACK HACK HACK
            if not self._use_skill_space:
                env = TaskOnehotWrapper(
                    env, task_index=env_idx, n_total_tasks=self._num_tasks
                )
            if self._normalize:
                env = Normalize(env)
            return env

        self._train_envs = []
        self._test_envs = []
        for idx in range(self._num_tasks):
            self._train_envs.append(_make_multistage_env(idx))
            self._test_envs.append(_make_multistage_env(idx))
        self._sample_env = self._train_envs[0]
        self._max_episode_steps = max_episode_steps
        self.metadata = {  # support different versions of gym
            "render.modes": ["human", "rgb_array", True, False],
            "render_modes": ["human", "rgb_array", True, False],
            "video.frames_per_second": int(np.round(1.0 / self._sample_env.dt)),
        }

    @property
    def num_stages(self):
        return self._num_stages

    def get_stage_id(self, observation):
        # always returns 0'
        ### ???
        raise NotImplementedError
        id_array = observation[..., -1]
        if isinstance(id_array, np.ndarray):
            id_array = id_array.astype(int)
        else:
            id_array = id_array.int()
        id_array[...] = 0

        if len(id_array.shape) == 0:
            id_array = id_array[()]
        return id_array

    def get_train_envs(self):
        return self._train_envs  # [0] if self._num_tasks < 2 else self._train_envs

    def get_test_envs(self):
        return self._test_envs  # [0] if self._num_tasks < 2 else self._test_envs
