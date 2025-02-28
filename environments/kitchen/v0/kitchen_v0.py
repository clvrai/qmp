"""Environments using kitchen and Franka robot."""

import os
import sys


class SuppressStdout:  # TODO: temporarily added here to suppress D4RL output
    def __enter__(self):
        self._prev_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._prev_stdout


import mujoco_py
import numpy as np
from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES
from environments.kitchen import reward_utils


OBS_ELEMENT_SITES = {
    "bottom burner": "knob2_site",
    "top burner": "knob4_site",
    "light switch": "light_site",
    "slide cabinet": "slide_site",
    "hinge cabinet": "hinge_site2",
    "microwave": "microhandle_site",
    "kettle": "kettle_site",
}

OBS_ELEMENT_INITS = {
    "bottom burner-on": np.array([3.12877220e-05, -4.51199853e-05]),
    "bottom burner-off": np.array([-0.88, -0.01]),
    "top burner-on": np.array([6.28065475e-05, 4.04984708e-05]),
    "top burner-off": np.array([-0.92, -0.01]),
    "light switch-on": np.array([4.62730939e-04, -2.26906415e-04]),
    "light switch-off": np.array([-0.69, -0.05]),
    "slide cabinet-open": np.array([-4.65501369e-04]),
    "slide cabinet-close": np.array([0.37]),
    "hinge cabinet-open": np.array([-6.44129196e-03, -1.77048263e-03]),
    "hinge cabinet-close": np.array([0.0, 1.45]),
    "microwave-open": np.array([1.08009684e-03]),
    "microwave-close": np.array([-0.75]),
    "kettle-push": np.array(
        [
            -2.69397440e-01,
            3.50383255e-01,
            1.61944683e00,
            1.00618764e00,
            4.06395120e-03,
            -6.62095997e-03,
            -2.68278933e-04,
        ]
    ),
    "kettle-pull": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

OBS_ELEMENT_GOALS = {
    "bottom burner-on": np.array([-0.88, -0.01]),
    "bottom burner-off": np.array([3.12877220e-05, -4.51199853e-05]),
    "top burner-on": np.array([-0.92, -0.01]),
    "top burner-off": np.array([6.28065475e-05, 4.04984708e-05]),
    "light switch-on": np.array([-0.69, -0.05]),
    "light switch-off": np.array([4.62730939e-04, -2.26906415e-04]),
    "slide cabinet-open": np.array([0.37]),
    "slide cabinet-close": np.array([-4.65501369e-04]),
    "hinge cabinet-open": np.array([0.0, 1.45]),
    "hinge cabinet-close": np.array([-6.44129196e-03, -1.77048263e-03]),
    "microwave-open": np.array([-0.75]),
    "microwave-close": np.array([1.08009684e-03]),
    "kettle-push": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    "kettle-pull": np.array(
        [
            -2.69397440e-01,
            3.50383255e-01,
            1.61944683e00,
            1.00618764e00,
            4.06395120e-03,
            -6.62095997e-03,
            -2.68278933e-04,
        ]
    ),
}


@configurable(pickleable=True)
class KitchenSingleTaskEnv(KitchenTaskRelaxV1):
    TASK_NAME = None
    BONUS_THRESH = 0.3

    def __init__(
        self,
        sparse_reward=False,
        terminate_on_success=True,
        control_penalty=0.0,
        early_termination_bonus=0,
        **kwargs
    ):
        self._sparse_reward = sparse_reward
        self._control_penalty = control_penalty
        self._terminate_on_success = terminate_on_success
        self._early_termination_bonus = early_termination_bonus
        self._element = self.TASK_NAME.split("-")[0]
        self._element_idx = OBS_ELEMENT_INDICES[self._element]
        self._element_site = OBS_ELEMENT_SITES[self._element]
        self._element_init = OBS_ELEMENT_INITS[self.TASK_NAME]
        self._element_goal = OBS_ELEMENT_GOALS[self.TASK_NAME]
        self._dist_goal_init = 0.0
        self._dist_hand_init = 0.0
        self._dist_hand_yz_init = 0.0
        self._init_left_pad = np.zeros(3)
        self._init_right_pad = np.zeros(3)
        self._reward_scale = 20.0
        self._t = 0
        self._seed_num = np.random.randint(1000000)

        self.viewer = None
        with SuppressStdout():
            super().__init__(**kwargs)
        self.init_qpos[self._element_idx] = self._element_init
        self.seed(self._seed_num)

    @property
    def env_name(self):
        return self.TASK_NAME

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        new_goal[self._element_idx] = self._element_goal
        ### karl's kitchen
        obj = ["microwave", "kettle", "bottom burner", "light switch"]
        task = ["microwave-open", "kettle-push", "bottom burner-on", "light switch-on"]
        for element, t in zip(obj, task):
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[t]
            new_goal[element_idx] = element_goal
        return new_goal

    def reset_model(self):
        self._t = 0
        ret = super().reset_model()
        self._dist_goal_init = np.linalg.norm(self._element_init - self._element_goal)
        hand_to_goal = self._get_obj_pos() - self._get_hand_pos()
        self._dist_hand_init = np.linalg.norm(hand_to_goal)
        self._dist_hand_yz_init = np.linalg.norm(
            hand_to_goal + np.array([-hand_to_goal[0], 0.0, 0.0])
        )
        self._init_left_pad, self._init_right_pad = self._get_pad_pos()
        return ret

    def _get_pos(self, name):
        if name in self.model.site_names:
            return self.data.get_site_xpos(name).copy()
        if name in self.model.geom_names:
            return self.data.get_geom_xpos(name).copy()
        if name in self.model.body_names:
            return self.data.get_body_xpos(name).copy()
        raise ValueError("{} not found in the model".format(name))

    def _get_hand_pos(self):
        return self._get_pos("end_effector")

    def _get_pad_pos(self):
        left_pad, right_pad = self._get_pos("panda0_leftfinger"), self._get_pos(
            "panda0_rightfinger"
        )
        return left_pad, right_pad

    def _get_gripper_pos(self):
        finger1, finger2 = self._get_pad_pos()
        pos = (finger1 + finger2) / 2
        return pos

    def _get_obj_pos(self):
        return self._get_pos(OBS_ELEMENT_SITES[self._element])


    def _get_gripper_dist(self, qpos):
        """Gripper joint values range roughly between -0.01 and 0.048,
        indicating the displacement from the middle point.
        Returns the distance scaled to [0, 1].
        """
        finger1, finger2 = qpos[7], qpos[8]
        dist = min(max(finger1 + finger2, 0), 0.09)
        return dist / 0.09

    def step(self, action):
        obs, reward, done, env_info = super().step(action)
        self._t += 1

        ctrl_penalty = np.square(action).sum()  # control penalty
        if not self._sparse_reward:
            reward += -(ctrl_penalty * self._control_penalty)

        success = env_info["rewards"]["success"]
        env_info["success"] = success
        env_info["timeout"] = False
        env_info["VIS:dist_goal"] = env_info["rewards"]["dist_goal"]
        env_info["VIS:dist_hand"] = env_info["rewards"]["dist_hand"]
        env_info["VIS:dist_left_pad_x"] = env_info["rewards"]["dist_left_pad_x"]
        env_info["VIS:dist_right_pad_x"] = env_info["rewards"]["dist_right_pad_x"]
        env_info["VIS:dist_hand_yz"] = env_info["rewards"]["dist_hand_yz"]
        env_info["VIS:ctrl_penalty"] = float(ctrl_penalty) * self._control_penalty

        if self._terminate_on_success:
            done |= success
        env_info["timeout"] = False  ###HACk
        if self.initializing:
            done = False

        # remove dictionary from env_info since garage doesn't support it
        del env_info["obs_dict"]
        del env_info["rewards"]
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
        bonus = float(success)
        if self._sparse_reward:
            reward = bonus
        else:
            if self._terminate_on_success and success:
                reward = 1.0 * self._early_termination_bonus
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
        in_place = reward_utils.tolerance(
            dists["goal"],
            bounds=(0, self.BONUS_THRESH),
            margin=abs(dists["goal_init"] - self.BONUS_THRESH),
            # sigmoid="long_tail", ##"gaussian"
            sigmoid="gaussian", ##"gaussian"
        )

        handle_reach_radius = 0.08
        reach = reward_utils.tolerance(
            dists["hand"],
            bounds=(0, handle_reach_radius),
            margin=abs(dists["hand_init"] - handle_reach_radius),
            sigmoid="gaussian",
        )

        # reward = reward_utils.hamacher_product(reach, in_place)
        reward = 0.4 * in_place + 0.6 * reach

        stage_success = False
        if dists["goal"] < self.BONUS_THRESH:
            stage_success = True

        # Make rewards negative by default
        reward -= 1.0

        return reward, stage_success



    @classmethod
    def aggregate_infos(cls, infos, info):
        for k, v in info.items():
            if k == "success" or k == "timeout":
                infos[k] = int(infos[k] or v)
            elif "dist" in k or k == "time":
                infos[k] = v
            elif "penalty" in k or "score" in k:
                infos[k] += v
        return infos

    def seed(self, seed=None):
        self._seed_num = seed
        super().seed(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def render(self, mode="human", width=400, height=400):
        if mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                np.copyto(self.viewer.cam.lookat, [-0.2, 0.5, 2.0])
                self.viewer.cam.distance = 2.2
                self.viewer.cam.azimuth = 70
                self.viewer.cam.elevation = -35
            render_device = os.environ.get("CUDA_VISIBLE_DEVICES", -1)
            img = self.sim.render(width=width, height=height, device_id=render_device)[
                ::-1
            ]
            return img
        else:
            return super().render(mode=mode)
    #def render(self, mode="human", width=400, height=400):
    #    return []
        # if mode == "rgb_array":
        #     if self.viewer is None:
        #         self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        #         np.copyto(self.viewer.cam.lookat, [-0.2, 0.5, 2.0])
        #         self.viewer.cam.distance = 2.2
        #         self.viewer.cam.azimuth = 70
        #         self.viewer.cam.elevation = -35
        #     render_device = os.environ.get("CUDA_VISIBLE_DEVICES", -1)
        #     img = self.sim.render(width=width, height=height, device_id=render_device)[
        #         ::-1
        #     ]
        #     return img
        # else:
        #     return super().render(mode=mode)



class KitchenBottomBurnerOnEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "bottom burner-on"
    BONUS_THRESH = 0.3
class KitchenBottomBurnerOffEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "bottom burner-off"
    BONUS_THRESH = 0.3
class KitchenTopBurnerOnEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "top burner-on"
    BONUS_THRESH = 0.3
class KitchenTopBurnerOffEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "top burner-off"
    BONUS_THRESH = 0.3
class KitchenLightSwitchOnEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "light switch-on"
    BONUS_THRESH = 0.3
class KitchenLightSwitchOffEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "light switch-off"
    BONUS_THRESH = 0.3
class KitchenSlideCabinetOpenEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "slide cabinet-open"
    BONUS_THRESH = 0.1
class KitchenSlideCabinetCloseEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "slide cabinet-close"
    BONUS_THRESH = 0.1
class KitchenHingeCabinetOpenEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "hinge cabinet-open"
    BONUS_THRESH = 0.1
class KitchenHingeCabinetCloseEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "hinge cabinet-close"
    BONUS_THRESH = 0.3
class KitchenMicrowaveOpenEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "microwave-open"
    BONUS_THRESH = 0.1
class KitchenMicrowaveCloseEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "microwave-close"
    BONUS_THRESH = 0.1
class KitchenKettlePushEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "kettle-push"
    BONUS_THRESH = 0.1
class KitchenKettlePullEnvV0(KitchenSingleTaskEnv):
    TASK_NAME = "kettle-pull"
    BONUS_THRESH = 0.1
