from d4rl.pointmaze import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE, U_MAZE_EVAL, MEDIUM_MAZE_EVAL, LARGE_MAZE_EVAL
from garage.envs import (
    GymEnv,
    TaskNameWrapper,
    TaskOnehotWrapper,
    normalize as Normalize,
)
import gym
from gym.spaces import Box
import numpy as np
import torch

from environments.maze import MazeEnv, MazeTask
from environments.maze.maze_layout import rand_layout
from environments.maze.maze_task import get_tasks
from environments.kitchen.kitchen_spirl import SkillWrapper  # TODO
from environments.kitchen.spirl.spirl_skill_decoder import load_skill_decoder  # TODO

# TODO: use skill

maze_specs = {
    "OPEN": OPEN,
    "U_MAZE": U_MAZE,
    "MEDIUM_MAZE": MEDIUM_MAZE,
    "LARGE_MAZE": LARGE_MAZE,
    "U_MAZE_EVAL": U_MAZE_EVAL,
    "MEDIUM_MAZE_EVAL": MEDIUM_MAZE_EVAL,
    "LARGE_MAZE_EVAL": LARGE_MAZE_EVAL,
    "MT10": LARGE_MAZE,
}


class MazeTaskEnv(MazeEnv):
    render_width = 400
    render_height = 400
    render_device = -1

    complete_threshold = 0.5

    def __init__(
        self, maze_spec, task, sparse_reward, max_episode_steps,
        done_on_completed=True, include_task_id=False, position_only=False,
    ):
        self._sparse_reward = sparse_reward
        self._max_episode_steps = max_episode_steps
        self._t = 0
        self._seed_num = np.random.randint(1000000)

        self.min_reward = self._sparse_reward - 1.0

        reward_type = self.reward_types[0 if sparse_reward else 1]

        if isinstance(maze_spec, int):
            self.maze_size = maze_spec
            self.maze_spec = rand_layout(size=maze_spec, seed=0)
        elif maze_spec in maze_specs:
            self.maze_spec = maze_specs[maze_spec]
        elif maze_spec in maze_specs.values():
            self.maze_spec = maze_spec
        else:
            raise ValueError("Unsupported maze spec of type {}: {}".format(
                type(maze_spec), maze_spec,
            ))

        # for initialization
        self.task = MazeTask([0, 0], [0, 0])
        self.done_on_completed = False
        self.position_only = position_only

        super(MazeEnv, self).__init__(self.maze_spec, reward_type, reset_target=False)
        self.seed(self._seed_num)

        # backward compatilbility for gym mujoco
        render_modes = self.metadata.get("render_modes", self.metadata.get("render.modes", None))
        self.metadata.update({
            "render.modes": render_modes,
            "render_modes": render_modes,
        })

        self._set_task(task)
        self.done_on_completed = done_on_completed
        self.include_task_id = include_task_id

        gym.utils.EzPickle.__init__(
            self, maze_spec, task, sparse_reward, max_episode_steps,
            done_on_completed=done_on_completed, include_task_id=include_task_id,
            position_only=position_only,
        )

    def _set_task(self, task):
        self.task = task
        self.set_target(task.goal_loc)
        self.set_marker()  # for rendering

    def _get_obs(self):
        obs = super()._get_obs()
        if self.position_only:
            obs = obs[..., :2]
        return obs

    def reset_model(self):
        self._t = 0
        return super().reset_model()

    def step(self, action):
        obs, reward, done, env_info = super().step(action)
        self._t += 1
        if not self._sparse_reward:
            reward -= 1  # time penalty to make the reward negative
        env_info["timeout"] = self._t >= self._max_episode_steps
        return obs, reward, done, env_info

    @classmethod
    def aggregate_infos(cls, infos, info):  # TODO: adapt to info from MazeEnv
        for k, v in info.items():
            if k == "success" or k == "timeout":
                infos[k] = int(infos[k] or v)
            elif "dist" in k or k == "time":
                infos[k] = v
            elif "penalty" in k or "score" in k:
                infos[k] += v
        return infos


class MazeMultistageEnv(MazeTaskEnv):
    def _set_task(self, task):
        self._multistage_task = task
        self._num_stages = len(task) - 1
        self.set_stage(0)

    def set_stage(self, stage_id):
        self._curr_stage_id = stage_id
        super().set_task(
            self._multistage_task[stage_id],
            self._multistage_task[stage_id+1],
        )

    def reset_model(self):
        self.set_stage(0)
        return super().reset_model()

    def step(self, action):
        obs, reward, done, env_info = super().step(action)

        if env_info["success"]:
            # Advance stage
            self._curr_stage_id += 1
            if self._curr_stage_id < self._num_stages:  # more stages to go
                env_info["success"] = False
                self.set_stage(self._curr_stage_id)
            reward += 1  # bonus for completing the stage
            obs = self._concat_stage_id(obs, replace=True)  # TODO: needed?

        env_info["stages_completed"] = self._curr_stage_id
        return obs, reward, done, env_info

    def _get_obs(self):
        obs = super()._get_obs()
        return self._concat_stage_id(obs)

    def _concat_stage_id(self, vec, replace=False):
        stage_encoding = np.zeros(self._num_stages, dtype=vec.dtype)
        stage_encoding[self._curr_stage_id] = 1
        if replace:
            vec = vec[:-self._num_stages]
        return np.concatenate([vec, stage_encoding]).ravel()


class MazeMultitaskEnv(gym.Env):
    task_env_cls = MazeTaskEnv

    def __init__(
        self,
        maze_spec,
        task_name=None,
        num_tasks=10,
        num_stages=1,
        sparse_tasks=[],
        include_task_id=False,
        normalize=False,
        terminate_on_success=True,
        position_only=False,
        max_episode_steps=200,
        use_skill_space=False,
        accumulate_reward=True,
        vectorized_skills=False,
    ):
        self._maze_spec = maze_spec
        self._num_tasks = num_tasks
        self._num_stages = num_stages
        self._sparse_tasks = sparse_tasks
        self._include_task_id = include_task_id
        self._normalize = normalize
        self._terminate_on_success = terminate_on_success
        self._position_only = position_only
        self._use_skill_space = use_skill_space
        self._vectorized_skills = vectorized_skills
        self._count = 0

        if self._use_skill_space:  # and not self._vectorized_skills:
            ### Load spirl skill model here
            print("Loading SPiRL skill decoder...")
            self._skill_model = load_skill_decoder()

        def _make_env(task, env_idx, train=True):
            env = self.task_env_cls(
                maze_spec=maze_spec,
                task=task,
                sparse_reward=env_idx in self._sparse_tasks,
                max_episode_steps=max_episode_steps,
                include_task_id=self._include_task_id,
                done_on_completed=self._terminate_on_success,
                position_only=self._position_only
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
            env = TaskNameWrapper(env, task_id=env_idx)
            env = TaskOnehotWrapper(
                env, task_index=env_idx, n_total_tasks=self._num_tasks
            )
            if self._normalize:
                env = Normalize(env)
            return env

        if task_name is not None:
            task_indices = [int(i) for i in task_name.strip().split(',')]
            self._num_tasks = len(task_indices)
        else:
            task_indices = None

        train_tasks = get_tasks(self._maze_spec, num_tasks, task_indices, train=True)
        test_tasks = get_tasks(self._maze_spec, num_tasks, task_indices, train=True)  # same as train

        self._train_envs = []
        self._test_envs = []

        for idx, task in enumerate(train_tasks):
            self._train_envs.append(_make_env(task, idx, train=True))

        for idx, task in enumerate(test_tasks):
            self._test_envs.append(_make_env(task, idx, train=False))

        self.min_reward = np.min([train_env.min_reward for train_env in self._train_envs])
        self._sample_env = self._train_envs[0]
        self._max_episode_steps = self._sample_env.spec.max_episode_length
        self.metadata = {  # backward compatilbility for gym mujoco
            "render.modes": ["human", "rgb_array", True, False],
            "render_modes": ["human", "rgb_array", True, False],
            "video.frames_per_second": int(np.round(1.0 / self._sample_env.dt)),
        }

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

    @property
    def num_stages(self):
        return self._num_stages

    def get_train_envs(self):
        return self._train_envs

    def get_test_envs(self):
        return self._test_envs

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

    def set_render_options(self, width=400, height=400, device=-1, fps=30, frame_drop=1):
        for env in self._train_envs:
            env.set_render_options(width, height, device, fps, frame_drop)

        for env in self._test_envs:
            env.set_render_options(width, height, device, fps, frame_drop)


class MazeMultitaskMultistageEnv(MazeMultitaskEnv):
    task_env_cls = MazeMultistageEnv

    def get_stage_id(self, observation):
        id_array = np.argmax(
            observation[..., -self._num_stages:],
            axis=-1,
        )
        if len(id_array.shape) == 0:
            id_array = id_array[()]
        return id_array
