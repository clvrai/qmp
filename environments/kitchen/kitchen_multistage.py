"""Environments using kitchen and Franka robot."""

import os
import sys


# class SuppressStdout:  # TODO: temporarily added here to suppress D4RL output
#     def __enter__(self):
#         self._prev_stdout = sys.stdout
#         sys.stdout = open(os.devnull, "w")
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._prev_stdout
# from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1


import mujoco_py
import numpy as np
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES

from environments.kitchen.v0.kitchen_v0 import (
    KitchenSingleTaskEnv,
    OBS_ELEMENT_SITES,
    OBS_ELEMENT_INITS,
    OBS_ELEMENT_GOALS,
)

from environments.kitchen import KITCHEN_ALL


@configurable(pickleable=True)
class KitchenMultistageEnv(KitchenSingleTaskEnv):
    def __init__(self, task_names=None, **kwargs):
        ### HACK HACK HACK
        if task_names is None:
            task_names = ["slide cabinet-open", "kettle-push", "bottom burner-on"]
        self.task_names = task_names

        self._num_stages = len(task_names)
        self._stage_id = 0
        self.TASK_NAME = self.task_names[self._stage_id]

        super().__init__(**kwargs)
        ### Setting initial states for all tasks
        for subtask in self.task_names:
            element = subtask.split("-")[0]
            self.init_qpos[OBS_ELEMENT_INDICES[element]] = OBS_ELEMENT_INITS[subtask]
        self.seed(self._seed_num)

    @property
    def env_name(self):
        raise NotImplementedError
        # return self.TASK_NAME

    def _set_subtask(self, subtask):
        self.TASK_NAME = subtask
        self._element = subtask.split("-")[0]
        self._element_idx = OBS_ELEMENT_INDICES[self._element]
        self._element_site = OBS_ELEMENT_SITES[self._element]
        self._element_init = OBS_ELEMENT_INITS[self.TASK_NAME]
        self._element_goal = OBS_ELEMENT_GOALS[self.TASK_NAME]
        self.goal = self._get_task_goal()

        ### Recalculate init stats
        self._dist_goal_init = np.linalg.norm(self._element_init - self._element_goal)
        hand_to_goal = self._get_obj_pos() - self._get_hand_pos()
        self._dist_hand_init = np.linalg.norm(hand_to_goal)
        self._dist_hand_yz_init = np.linalg.norm(
            hand_to_goal + np.array([-hand_to_goal[0], 0.0, 0.0])
        )
        self._init_left_pad, self._init_right_pad = self._get_pad_pos()

    def reset_model(self):
        ### setting first subtask
        self._set_subtask(self.task_names[0])
        self._stage_id = 0

        ret = super().reset_model()
        return ret

    def set_stage(self, stage_id):
        self._stage_id = stage_id
        self._set_subtask(self.task_names[self._stage_id])
        print(
            f"Task: {self.TASK_NAME}, Element: {self._element_site}, Element Init: {self._element_init}, Element Goal: {self._element_goal}"
        )

    def step(self, action):
        obs, reward, done, env_info = super().step(action)

        success = env_info["success"]
        if success:
            ## Advance stage
            self._stage_id += 1
        ### Re-calculate done
        done = False
        if self._terminate_on_success and self._stage_id == self._num_stages:
            done |= success
        if self._t >= self.MAX_EPISODE_STEPS:
            # done = True
            env_info["timeout"] = True
        if self.initializing:
            done = False

        if success and self._stage_id < self._num_stages:
            self._set_subtask(self.task_names[self._stage_id])

        env_info["stages_completed"] = self._stage_id

        return obs, reward, done, env_info

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super()._get_reward_n_score(obs_dict)
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        next_element = next_obj_obs[..., self._element_idx - idx_offset]
        gripper_pos = self._get_gripper_pos()
        left_pad, right_pad = self._get_pad_pos()
        obj_pos = self._get_obj_pos()
        gripper_yz = gripper_pos + np.array([-gripper_pos[0], 0.0, 0.0])
        obj_yz = obj_pos + np.array([-obj_pos[0], 0.0, 0.0])
        dists = {
            "goal": np.linalg.norm(next_element - self._element_goal),
            "goal_init": self._dist_goal_init,
            "hand": np.linalg.norm(
                obj_pos
                # - self._get_hand_pos()
                - gripper_pos
            ),
            "hand_init": self._dist_hand_init,
            "gripper": self._get_gripper_dist(next_q_obs),
            "left_pad_x": left_pad[0] - obj_pos[0],
            "right_pad_x": right_pad[0] - obj_pos[0],
            "hand_yz": np.linalg.norm(gripper_yz - obj_yz, ord=2),
        }
        reward, success = self._compute_reward(obs_dict, dists)
        ### Stage-wise bonus
        bonus = float(success) * (self._stage_id + 1)
        if self._sparse_reward:
            reward = bonus
        else:
            ### Check if all stages done
            if (
                self._terminate_on_success
                and success
                and self._stage_id == self._num_stages - 1
            ):
                reward = 1.0 * self.MAX_EPISODE_STEPS

            ### Stage-wise rewards
            else:
                reward += 1.0 * (self._stage_id)
            reward *= self._reward_scale
        reward_dict["bonus"] = bonus
        reward_dict["dist_goal"] = dists["goal"]
        reward_dict["dist_hand"] = dists["hand"]
        reward_dict["dist_left_pad_x"] = dists["left_pad_x"]
        reward_dict["dist_right_pad_x"] = dists["right_pad_x"]
        reward_dict["dist_hand_yz"] = dists["hand_yz"]
        reward_dict["r_total"] = reward
        reward_dict["success"] = success
        score = bonus
        return reward_dict, score

    def _compute_reward(self, obs_dict, dists):
        ### Need to identify subtask compute_reward function
        ret = KITCHEN_ALL[self.TASK_NAME]._compute_reward(obs_dict, dists)
        return ret
