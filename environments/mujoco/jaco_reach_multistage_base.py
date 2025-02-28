import numpy as np
import torch

from gym import utils
from gym.envs.mujoco import mujoco_env

from environments.mujoco.jaco import JacoEnv
from environments.mujoco.asset_utils import get_asset_path

SIGMA=0.1

class JacoReachMultistageEnv(JacoEnv):
    def __init__(
        self,
        task_id,
        num_tasks,
        goal_locations,
        sparse_reward=False,
        include_task_id=False,
        reward_shift=0,
        time_reward=True,
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

        self.task_id = task_id
        self._num_tasks = num_tasks
        self._sparse_reward = sparse_reward
        self._reward_shift = reward_shift
        self._include_task_id = include_task_id
        self._time_reward = time_reward

        self._t = 0
        self._steps_success = 0
        self._stage_id = 0

        self._goal_locations = goal_locations
        self._goal = None
        self._num_stages = 3
        self._stay = False  # Whether current task is a "stay" type task

        asset_path = get_asset_path("jaco_reach_multistage.xml")
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

        for i in range(self._num_stages):
            if self._goal_locations[i] is None:
                self._goal_locations[i] = self._get_hand_pos()
                self._stay = True

    def reset_model(self):
        self._stage_id = 0
        self._steps_success = 0
        self._goal = None

        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(
            low=-init_randomness, high=init_randomness, size=self.model.nq
        )
        qvel = self.init_qvel + np.random.uniform(
            low=-init_randomness * 0.1, high=init_randomness * 0.1, size=self.model.nv
        )

        # if self.task_id == 2:  # make sure hand stays high enough
        #     qpos[1] -= 0.4
        #     qpos[2] += 0.1

        self.set_state(qpos, qvel)
        self._set_goal()

        return self._get_obs()

    def step(self, a):
        self._t += 1
        prev_ob = self._get_obs()
        if self._add_action_noise:
            a += np.random.normal(0, SIGMA, a.shape)
            # clip a to be within the action space
            a = np.clip(a, self.action_space.low, self.action_space.high)
        sim_success = self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        ctrl_reward = self._ctrl_reward(a)
        dist_target = self._get_distance_hand("target")
        dist_reward = -dist_target
        success = dist_target < self._config["success_threshold"]
        done = False

        if not sim_success:
            ctrl_reward += -self._config["fail_penalty"]  # large penalty
            ob = prev_ob.copy()  # ob is reset; use previous one
            done = True

        ### Advance Stage
        if success:
            if self._stay:
                self._steps_success += 1
                self._stage_id = self._steps_success // 30
                if self._stage_id >= self._num_stages:
                    done = True
            else:
                self._stage_id += 1
                if self._stage_id >= self._num_stages:
                    done = True
                else:
                    self._set_goal()

        if self._sparse_reward:
            dist_reward = 0  # TODO: set other rewards to 0 as well?

        if self._time_reward:
            stage_reward = -0.4 * max(3 - self._stage_id, 0)
        else:
            stage_reward = (
                (abs(self._reward_shift) + 1) * 200 / self._num_stages
            ) * success
        if self._stay:
            if self._sparse_reward:
                stage_reward = (
                    success * (self._steps_success % 30 == 0) * self._steps_success
                )
            else:
                stage_reward = 0  # success
        reward = dist_reward + stage_reward + ctrl_reward + self._reward_shift
        info = {
            "stage_reward": stage_reward,
            "ctrl_reward": ctrl_reward,
            "dist_reward": dist_reward,
            "success": success,
            "stages_completed": self._stage_id,
            "task_id": self.task_id,
        }
        return ob, reward, done, info

    def reward_fn(self, obs, acs):
        ctrl_reward = -self._config["action_penalty"] * np.square(acs).sum(axis=1)
        # ctrl_reward += -self._config["velocity_penalty"] * np.abs(self.data.qvel).mean()
        # ctrl_reward += -self._config["acceleration_penalty"] * np.abs(self.data.qacc).mean()
        stage_ids = self.get_stage_id(obs)
        # How to calculate???
        pos = [self._goal_locations[idx] for idx in stage_ids]
        hand_pos = obs[..., - 3 :]
        dist_target = np.linalg.norm(pos - hand_pos, axis=1)
        dist_reward = -dist_target
        success = dist_target < self._config["success_threshold"]

        if self._sparse_reward:
            dist_reward = np.zeros(len(obs))

        if self._time_reward:
            stage_reward = -0.4 * np.maximum(3 - stage_ids, np.zeros(len(obs)))
        else:
            stage_reward = (
                (abs(self._reward_shift) + 1) * 200 / self._num_stages
            ) * success
        if self._stay:
            stage_reward = np.zeros(len(obs))  # success
        reward = dist_reward + stage_reward + ctrl_reward + self._reward_shift
        return np.expand_dims(reward, axis=-1)

    def _set_goal(self):
        self._goal = self._goal_locations[self._stage_id]
        self.sim.data.mocap_pos[0] = self._goal

    def _get_hand_pos(self):
        return self._get_pos("jaco_link_hand")

    ### Changes for reward_fn
    def _get_obs(self):
        hand_pos = self._get_hand_pos()
        # goal_pos = self._get_target_pos()
        stage_encoding = np.zeros(self._num_stages)
        stage_encoding[self._stage_id] = 1
        task_encoding = np.zeros(self._num_tasks)
        task_encoding[self.task_id] = 1
        return (
            np.concatenate(
                [
                    self.data.qpos,
                    self.data.qvel,
                    # hand_pos - goal_pos, ### No more goal diff info
                    stage_encoding,
                    task_encoding,
                    hand_pos,
                ]
            )
            .ravel()
            .astype(np.float32)
        )

    def get_stage_id(self, observation):
        id_array = np.argmax(
            observation[..., -(self._num_stages + self._num_tasks+3) : -(self._num_tasks+3)],
            axis=-1,
        )
        if len(id_array.shape) == 0:
            id_array = id_array[()]
        return id_array

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__init__(
            self.task_id,
            self._num_tasks,
            self._goal_locations,
            sparse_reward=self._sparse_reward,
            include_task_id=self._include_task_id,
            reward_shift=self._reward_shift,
            time_reward=self._time_reward
        )
        return result


class JacoReachMTEnv(JacoEnv):
    def _init_envs(self):
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
                    add_action_noise=self._add_action_noise,
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
                    add_action_noise=self._add_action_noise,
                )
            )
        self._curr_env = self._train_envs[0]

        asset_path = get_asset_path("jaco_reach_multistage.xml")
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

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

    def reset_model(self):
        self._curr_env = self._train_envs[self._count % self._num_tasks]
        self._count += 1
        return self._curr_env.reset_model()

    def step(self, a):
        return self._curr_env.step(a)

    def get_task_id(self, observation):
        if isinstance(observation, np.ndarray):
            id_array = np.argmax(
                observation[..., -(self._num_tasks+3) :-3],
                axis=-1,
            )
        else:
            id_array = torch.argmax(
                observation[..., -(self._num_tasks+3) :-3],
                dim=-1,
            )
        if len(id_array.shape) == 0:
            id_array = id_array[()]
        return id_array

    def get_stage_id(self, observation):
        id_array = np.argmax(
            observation[..., -(self._num_stages + self._num_tasks+3) : -(self._num_tasks+3)],
            axis=-1,
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

        ### Zero out hand pos
        obs_without_task[..., -3 :] = 0

        if not self._include_task_id:
            ### zero out task id
            obs_without_task[..., -(self._num_tasks+3) :] = 0

        task_info = observation

        return obs_without_task, task_info
