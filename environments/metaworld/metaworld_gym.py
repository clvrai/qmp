from akro import Box
from garage import Wrapper, EnvSpec
from garage.envs import normalize as Normalize
from garage.experiment.task_sampler import MetaWorldTaskSampler
import gym
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from environments.metaworld.metaworld_MT3 import MT3_V2_dict, MT3
from environments.metaworld.metaworld_cds import cds_envs_dict, CDSEnvs, cds_envs_dict_v1, CDSEnvsV1
import numpy as np


def task_getter(benchmark, class_dict):
    def _task_getter(env):
        unwrapped = getattr(env, "unwrapped", env)
        return [
            task
            for task in benchmark.train_tasks
            if type(unwrapped) == class_dict[task.env_name]
        ]

    return _task_getter


class MetaWorldEnv(gym.Env):
    def __init__(
        self,
        benchmark_type,
        task_name=None,
        sparse_tasks=[],
        include_task_id=False,
        normalize=False,
        fixed_train_task=False,
        fixed_test_task=False,
    ):
        self._sparse_tasks = sparse_tasks
        assert all(0 <= t < self._num_tasks for t in self._sparse_tasks)
        self._include_task_id = include_task_id

        base_seed = np.random.randint(1000000)
        if benchmark_type == "MT10":
            class_dict = _env_dict.MT10_V2
            self._num_tasks = 10
            self._benchmark = metaworld.MT10(seed=base_seed)
            self._benchmark_test = metaworld.MT10(seed=base_seed + 100)
        elif benchmark_type == "MT50":
            class_dict = _env_dict.MT50_V2
            self._num_tasks = 50
            self._benchmark = metaworld.MT50(seed=base_seed)
            self._benchmark_test = metaworld.MT50(seed=base_seed + 100)
        elif benchmark_type == "MT1":
            class_dict = _env_dict.ALL_V2_ENVIRONMENTS
            assert task_name in class_dict
            self._num_tasks = 1
            self._benchmark = metaworld.MT1(task_name, seed=base_seed)
            self._benchmark_test = metaworld.MT1(task_name, seed=base_seed + 100)
        elif benchmark_type == "MT3":
            class_dict = MT3_V2_dict
            self._num_tasks = 3
            self._benchmark = MT3(seed=base_seed)
            self._benchmark_test = MT3(seed=base_seed + 100)
        elif benchmark_type == "CDS":
            class_dict = cds_envs_dict
            self._num_tasks = 4
            self._benchmark = CDSEnvs(seed=base_seed)
            self._benchmark_test = CDSEnvs(seed=base_seed + 100)

        elif benchmark_type == "CDS_v1":
            class_dict = cds_envs_dict_v1
            self._num_tasks = 4
            self._benchmark = CDSEnvsV1(seed=base_seed)
            self._benchmark_test = CDSEnvsV1(seed=base_seed + 100)
        else:
            raise NotImplementedError()

        self._train_task_sampler = MetaWorldTaskSampler(
            self._benchmark,
            "train",
            lambda env, _: CustomWrapper(
                Normalize(env, normalize_reward=True) if normalize else env,
                get_tasks=task_getter(self._benchmark, class_dict),
                fixed_task=fixed_train_task,
            ),
            add_env_onehot=True,
        )

        self._test_task_sampler = MetaWorldTaskSampler(
            self._benchmark_test,
            "train",
            lambda env, _: CustomWrapper(
                Normalize(env) if normalize else env,
                get_tasks=task_getter(self._benchmark_test, class_dict),
                fixed_task=fixed_test_task,
            ),
            add_env_onehot=True,
        )


        self._train_envs = self._train_task_sampler.sample(self._num_tasks)
        self._test_envs = [
            env_up() for env_up in self._test_task_sampler.sample(self._num_tasks)
        ]
        self.min_reward = 0 # MT-v2 rewards are positive

        self._sample_env = self._train_envs[0]()
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
        id_array = observation[..., -self._num_tasks :].argmax(-1)
        if len(id_array.shape) == 0:
            id_array = id_array[()]
        return id_array

    def get_stage_id(self, observation):
        # always returns 0
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
            obs_without_task[..., -self._num_tasks :] = 0

        # if self._goal_type == 0:  # as is
        #     pass
        # elif self._goal_type == 1:  # remove task id
        #     obs_without_task[..., -self._num_tasks :] = 0
        # else:
        #     raise ValueError("Unsupported goal type {}".format(self._goal_type))

        task_info = observation

        return obs_without_task, task_info


class CustomWrapper(Wrapper):
    def __init__(
        self,
        env,
        get_tasks,
        fixed_task,
    ):
        super().__init__(env)

        self._tasks = get_tasks(self)
        assert len(self._tasks) == 50
        self._fixed_task = fixed_task
        if not self._fixed_task:
            self._task_order = np.random.permutation(len(self._tasks))
            self._curr_task_idx = 0

        self.render_modes.extend(["rgb_array", True, False])
        self._observation_space = Box(
            low=self._env.observation_space.low,
            high=self._env.observation_space.high,
            dtype=np.float64,  # match actual observation dtype
        )
        self._spec = EnvSpec(
            observation_space=self._observation_space,
            action_space=self._env.action_space,
            max_episode_length=self._env.spec.max_episode_length,
        )

    @property
    def spec(self):
        return self._spec

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        if not self._fixed_task:
            task_id = self._task_order[self._curr_task_idx % len(self._task_order)]
            self._env.set_task(self._tasks[task_id])
            self._curr_task_idx += 1
            if self._curr_task_idx >= len(self._task_order):
                np.random.shuffle(self._task_order)
                self._curr_task_idx = 0
        return super().reset()

    def render(self, mode="human"):
        offscreen = mode != "human"
        return super().render(mode=offscreen)
