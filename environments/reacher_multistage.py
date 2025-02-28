import mujoco_py
import numpy as np
import torch

from environments.reacher import ReacherRandomizedEnv


class ReacherMultistageMTEnv(ReacherRandomizedEnv):
    def __init__(
        self,
        fixed_stage_lengths=False,
        sparse_tasks=[],
        include_task_id=False,
        random_reset=False,
    ):
        self.task_id = 0
        self._stage = 0
        self._initial_pos = None
        self._steps = 0
        self._stages_completed = [False, False, False]
        self.stage_length = 30
        self.fixed_stage_lengths = fixed_stage_lengths
        self.goal_locations = [  # dirty hack for easy task id retrieval
            [[0, 0.1401], [-0.05, -0.0501], [-0.1, 0.08]],
            [[0, 0.1402], [-0.05, -0.0502], [0.1, 0.1]],
        ]
        self._random_reset = random_reset
        self._sparse_tasks = sparse_tasks
        self._include_task_id = include_task_id
        self._num_tasks = 3
        self._count = 0
        self._train_envs = []
        self._test_envs = []
        for i in range(self._num_tasks):
            self._train_envs.append(
                ReacherMultistageEnv(
                    task_id=i,
                    fixed_stage_lengths=fixed_stage_lengths,
                    sparse_reward=i in sparse_tasks,
                    include_task_id=include_task_id,
                    random_reset=random_reset,
                )
            )
            self._test_envs.append(
                ReacherMultistageEnv(
                    task_id=i,
                    fixed_stage_lengths=fixed_stage_lengths,
                    sparse_reward=i in sparse_tasks,
                    include_task_id=include_task_id,
                    random_reset=random_reset,
                )
            )
        self._curr_env = self._train_envs[0]
        super().__init__()

    @staticmethod
    def disc2cont(action):
        assert 0 <= action < 9
        return np.array([-0.5, 0.0, 0.5])[[action // 3, action % 3]]

    @staticmethod
    def get_num_discrete_actions():
        return 9

    @property
    def num_tasks(self):
        return self._num_tasks

    def get_train_envs(self):
        return self._train_envs

    def get_test_envs(self):
        return self._test_envs

    def reset_model(self):
        self._curr_env = self._train_envs[self._count]
        self._count += 1
        return self._curr_env.reset_model()

    def step(self, a):
        return self._curr_env.step(a)

    def _get_task_id(
        self, goal, return_stage=False
    ):  # dirty hack for easy task id retrieval
        goal_dist = np.linalg.norm(
            np.asarray(self.goal_locations) - np.asarray(goal),
            axis=-1,
        )
        idx = np.where(goal_dist < 1e-6)
        if return_stage:
            idx = idx[1]
        else:
            idx = idx[0]

        if len(idx) > 0:
            return idx[0]
        else:
            return 0 if return_stage else 2

    def get_task_id(self, observation, return_stage=False):
        if len(observation.shape) > 1:
            goal_pos = self.to_numpy(observation[:, 4:6])
            ids = [self._get_task_id(pos, return_stage) for pos in goal_pos]
            if isinstance(observation, torch.Tensor):
                return torch.tensor(
                    ids, dtype=observation.dtype, device=observation.device
                )
            return np.array(ids, dtype=observation.dtype)
        else:
            goal_pos = self.to_numpy(observation[4:6])
            return self._get_task_id(goal_pos, return_stage)

    def _get_stage_id(self, goal):
        return goal[1].astype(int)

    def get_stage_id(self, observation):
        if len(observation.shape) > 1:
            goal_pos = self.to_numpy(observation[:, 4:6])
            ids = [self._get_stage_id(pos) for pos in goal_pos]
            if isinstance(observation, torch.Tensor):
                return torch.tensor(
                    ids, dtype=observation.dtype, device=observation.device
                )
            return np.array(ids, dtype=observation.dtype)
        else:
            goal_pos = self.to_numpy(observation[4:6])
            return self._get_stage_id(goal_pos)

    def to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        else:
            return np.asarray(x)

    def split_observation(self, observation):
        obs_without_task = (
            observation.copy()
            if isinstance(observation, np.ndarray)
            else observation.clone()
        )

        if self._include_task_id:
            obs_without_task[..., 4] = self.get_task_id(observation)
            obs_without_task[..., 5] = self.get_task_id(observation, return_stage=True)
        else:
            obs_without_task[..., 4] = 0
            obs_without_task[..., 5] = self.get_task_id(observation, return_stage=True)

        # if self._goal_type == 0:  # goal location
        #     pass
        # elif self._goal_type == 1:  # task + stage id
        #     obs_without_task[..., 4] = self.get_task_id(observation)
        #     obs_without_task[..., 5] = self.get_task_id(observation, return_stage=True)
        # elif self._goal_type == 2:  # task id only
        #     obs_without_task[..., 4] = self.get_task_id(observation)
        #     obs_without_task[..., 5] = 0
        # elif self._goal_type == 3:  # stage id only
        #     obs_without_task[..., 4] = 0
        #     obs_without_task[..., 5] = self.get_task_id(observation, return_stage=True)
        # elif self._goal_type == 4:  # none
        #     obs_without_task[..., 4:6] = 0
        # else:
        #     raise ValueError("Unsupported goal type {}".format(self._goal_type))

        task_info = observation  # not altering to exploit previous get_task_id, etc.
        return obs_without_task, task_info


class ReacherMultistageEnv(ReacherRandomizedEnv):
    def __init__(
        self,
        task_id,
        fixed_stage_lengths=False,
        sparse_reward=False,
        include_task_id=False,
        random_reset=False,
    ):
        self.task_id = task_id
        self._stage = 0
        self._initial_pos = None
        self._steps = 0
        self._stages_completed = [False, False, False]
        self.stage_length = 30
        self.fixed_stage_lengths = fixed_stage_lengths
        self.goal_locations = [  # dirty hack for easy task id retrieval
            [[0, 0.1401], [-0.05, -0.0501], [-0.1, 0.08]],
            [[0, 0.1402], [-0.05, -0.0502], [0.1, 0.1]],
        ]
        self._random_reset = random_reset
        self._sparse_reward = sparse_reward
        self._include_task_id = include_task_id
        super().__init__()

    def set_goal(self, goal):
        old_state = self.sim.get_state()
        qpos = old_state.qpos
        qpos[-2:] = goal
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, old_state.qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset_model(self):
        # self.task_id = self._count % self._num_tasks
        # self._count += 1
        self._stage = 0
        self._steps = 0
        self._stages_completed = [False, False, False]

        if self.task_id == 0:
            joint1 = np.pi / 2 + self.np_random.uniform(
                low=-0.1 * np.pi / 2,
                high=0.1 * np.pi / 2,
            )
            joint2 = np.pi / 2 + self.np_random.uniform(
                low=-0.1 * np.pi / 2,
                high=0.1 * np.pi / 2,
            )
            self.goal = self.goal_locations[self.task_id][0]

        elif self.task_id == 1:
            joint1 = 0 + self.np_random.uniform(
                low=-0.1 * np.pi / 2,
                high=0.1 * np.pi / 2,
            )
            joint2 = np.pi / 2 + self.np_random.uniform(
                low=-0.1 * np.pi / 2,
                high=0.1 * np.pi / 2,
            )
            self.goal = self.goal_locations[self.task_id][0]
        elif self.task_id == 2:
            if np.random.uniform() < 0.5:
                joint1 = np.pi / 2 + self.np_random.uniform(
                    low=-0.1 * np.pi / 2,
                    high=0.1 * np.pi / 2,
                )
                joint2 = np.pi / 2 + self.np_random.uniform(
                    low=-0.1 * np.pi / 2,
                    high=0.1 * np.pi / 2,
                )
            else:
                joint1 = 0 + self.np_random.uniform(
                    low=-0.1 * np.pi / 2,
                    high=0.1 * np.pi / 2,
                )
                joint2 = np.pi / 2 + self.np_random.uniform(
                    low=-0.1 * np.pi / 2,
                    high=0.1 * np.pi / 2,
                )
            self.goal = [0, 0]

        if self._random_reset:
            joint1 = self.np_random.uniform(low=-np.pi, high=np.pi)
            joint2 = self.np_random.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)

        qpos = np.array([0.0, 0.0, 0.0, 0.0])
        qpos[0] += joint1
        qpos[1] += joint2

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005,
            high=0.005,
            size=self.model.nv,
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        if self.task_id == 2:
            self.goal = self.get_body_com("fingertip")[:2]
            if self.task_id < 2:  # dirty hack for easy task id retrieval
                self.goal += [1e-5, 1e-5]
            self.set_goal(self.goal)

        return self._get_obs()

    def step(self, a):

        if self.fixed_stage_lengths:
            ### Calculate which stage based on self._steps so far
            new_stage = min(self._steps // self.stage_length, 2)
            if new_stage != self._stage:
                self._stage = new_stage
                if self.task_id == 0 or self.task_id == 1:
                    self.goal = self.goal_locations[self.task_id][self._stage]
                    self.set_goal(self.goal)

        ob, rew, done, infos = super().step(a)
        if self._sparse_reward:
            rew = float(infos["success"])

        if self.fixed_stage_lengths:
            ### Keep track of the number of stages with successes
            if self.task_id != 2 and infos["success"]:
                self._stages_completed[self._stage] = True

            infos["stages_completed"] = np.sum(self._stages_completed)

        if not self.fixed_stage_lengths:
            ### Check if success to advance stage
            if self.task_id != 2 and infos["success"]:
                self._stage += 1
                if self._stage > 2:
                    infos["reward_stage"] = 0
                    infos["stages_completed"] = self._stage
                    infos["task_id"] = self.task_id
                    return ob, rew, True, infos
                self.goal = self.goal_locations[self.task_id][self._stage]
                self.set_goal(self.goal)

            infos["stages_completed"] = self._stage

        if self.task_id == 2:
            self._stage = 2 * infos["success"]
            infos["stages_completed"] = 3 * infos["success"]

        ### Add reward based on stage
        reward_stage = 0.25 * (self._stage - 2)
        if self._sparse_reward:
            reward_stage = 0
        rew += reward_stage
        infos["reward_stage"] = reward_stage
        infos["task_id"] = self.task_id

        self._steps += 1

        return ob, rew, done, infos

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__init__(
            self.task_id,
            fixed_stage_lengths=self.fixed_stage_lengths,
            sparse_reward=self._sparse_reward,
            include_task_id=self._include_task_id,
            random_reset=self._random_reset,
        )
        return result
