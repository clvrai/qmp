import os
import numpy as np
import gym
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from collections import OrderedDict
from metaworld import Benchmark, _make_tasks, _MT_OVERRIDE
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerDoorEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerDrawerCloseEnvV2,
)


class DoorOpenEnvV2(SawyerDoorEnvV2):
    def __init__(self):
        super().__init__()

        obj_low = (-0.1, 0.85, 0.15)
        obj_high = (0.0, 0.95, 0.15)
        goal_low = (-0.4, 0.4, 0.1499)
        goal_high = (-0.3, 0.5, 0.1501)

        self.init_config["obj_init_pos"] = np.array([0.0, 0.95, 0.15])
        self.init_config["hand_init_pos"] = np.array([0.0, 0.6, 0.2])

        self.goal = np.array([-0.3, 0.7, 0.15])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        # return full_v2_path_for("sawyer_xyz/sawyer_door_pull.xml")
        return os.path.join(
            os.path.dirname(__file__), "assets/sawyer_xyz/door_drawer.xml"
        )

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        """
        Copy and reformat obs to match door env
        """
        obs_door = obs.copy()
        obs_door[4:11] = obs[11:18]
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
        ) = self.compute_reward(action, obs_door)

        success = float(abs(obs_door[4] - self._target_pos[0]) <= 0.08)

        info = {
            "success": success,
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_curr_obs_combined_no_goal(self):
        """
        Set observation to include both drawer and door obs
        """
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)

        ## To Do: Replace with drawer and door pos
        drawer_pos = self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])
        door_pos = self.data.get_geom_xpos("handle").copy()
        obj_pos = np.concatenate([drawer_pos, door_pos])
        assert len(obj_pos) % 3 == 0

        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        if self.isV2:
            ## To Do: Replace with drawer and door quat
            drawer_quat = self.sim.data.get_body_xquat("drawer_link")
            door_quat = Rotation.from_matrix(
                self.data.get_geom_xmat("handle")
            ).as_quat()
            obj_quat = np.concatenate([drawer_quat, door_quat])
            assert len(obj_quat) % 4 == 0
            obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
            obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
                [
                    np.hstack((pos, quat))
                    for pos, quat in zip(obj_pos_split, obj_quat_split)
                ]
            )
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
        else:
            # is a v1 environment
            obs_obj_padded[: len(obj_pos)] = obj_pos
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, obs_obj_padded))


class DoorCloseEnvV2(SawyerDoorCloseEnvV2):
    def __init__(self):
        super().__init__()

        obj_low = (-0.1, 0.85, 0.15)
        obj_high = (0.0, 0.95, 0.15)
        goal_low = (0.3, 0.65, 0.1499)
        goal_high = (0.4, 0.75, 0.1501)

        self.init_config["obj_init_pos"] = np.array([0.0, 0.95, 0.15])
        self.init_config["hand_init_pos"] = np.array([-0.6, 0.6, 0.2])

        self.goal = np.array([0.3, 0.8, 0.15])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        # return full_v2_path_for("sawyer_xyz/sawyer_door_pull.xml")
        return os.path.join(
            os.path.dirname(__file__), "assets/sawyer_xyz/door_drawer.xml"
        )

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obs_door = obs.copy()
        obs_door[4:11] = obs[11:18]
        reward, obj_to_target, in_place = self.compute_reward(action, obs_door)
        info = {
            "obj_to_target": obj_to_target,
            "in_place_reward": in_place,
            "success": float(obj_to_target <= 0.08),
            "near_object": 0.0,
            "grasp_success": 1.0,
            "grasp_reward": 1.0,
            "unscaled_reward": reward,
        }
        return reward, info

    def _get_curr_obs_combined_no_goal(self):
        """Combines the end effector's {pos, closed amount} and the object(s)'
            {pos, quat} into a single flat observation. The goal's position is
            *not* included in this.
        Returns:
            np.ndarray: The flat observation array (18 elements)
        """
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)

        ## To Do: Replace with drawer and door pos
        drawer_pos = self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])
        door_pos = self.data.get_geom_xpos("handle").copy()
        obj_pos = np.concatenate([drawer_pos, door_pos])
        assert len(obj_pos) % 3 == 0

        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        if self.isV2:
            ## To Do: Replace with drawer and door quat
            drawer_quat = self.sim.data.get_body_xquat("drawer_link")
            door_quat = Rotation.from_matrix(
                self.data.get_geom_xmat("handle")
            ).as_quat()
            obj_quat = np.concatenate([drawer_quat, door_quat])
            assert len(obj_quat) % 4 == 0
            obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
            obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
                [
                    np.hstack((pos, quat))
                    for pos, quat in zip(obj_pos_split, obj_quat_split)
                ]
            )
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
        else:
            # is a v1 environment
            obs_obj_padded[: len(obj_pos)] = obj_pos
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, obs_obj_padded))


class DrawerOpenEnvV2(SawyerDrawerOpenEnvV2):
    def __init__(self):
        super().__init__()

        obj_low = (0.3, 0.9, 0.0)
        obj_high = (0.4, 0.9, 0.0)

        self.init_config["obj_init_pos"] = np.array([0.4, 0.9, 0.0])
        self.init_config["hand_init_pos"] = np.array([0, 0.6, 0.2])

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )

    @property
    def model_name(self):
        # return full_v2_path_for("sawyer_xyz/sawyer_door_pull.xml")
        return os.path.join(
            os.path.dirname(__file__), "assets/sawyer_xyz/door_drawer.xml"
        )

    def reset_model(self):
        obs = super().reset_model()
        self.sim.model.site_pos[self.model.site_name2id("goal")] = self._target_pos
        return obs

    def _get_curr_obs_combined_no_goal(self):
        """Combines the end effector's {pos, closed amount} and the object(s)'
            {pos, quat} into a single flat observation. The goal's position is
            *not* included in this.
        Returns:
            np.ndarray: The flat observation array (18 elements)
        """
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)

        ## To Do: Replace with drawer and door pos
        drawer_pos = self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])
        door_pos = self.data.get_geom_xpos("handle").copy()
        obj_pos = np.concatenate([drawer_pos, door_pos])
        assert len(obj_pos) % 3 == 0

        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        if self.isV2:
            ## To Do: Replace with drawer and door quat
            drawer_quat = self.sim.data.get_body_xquat("drawer_link")
            door_quat = Rotation.from_matrix(
                self.data.get_geom_xmat("handle")
            ).as_quat()
            obj_quat = np.concatenate([drawer_quat, door_quat])
            assert len(obj_quat) % 4 == 0
            obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
            obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
                [
                    np.hstack((pos, quat))
                    for pos, quat in zip(obj_pos_split, obj_quat_split)
                ]
            )
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
        else:
            # is a v1 environment
            obs_obj_padded[: len(obj_pos)] = obj_pos
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, obs_obj_padded))


class DrawerCloseEnvV2(SawyerDrawerCloseEnvV2):
    def __init__(self):
        super().__init__()

        obj_low = (0.3, 0.9, 0.0)
        obj_high = (0.4, 0.9, 0.0)

        self.init_config["obj_init_pos"] = np.array([0.4, 0.9, 0.0])
        self.init_config["hand_init_pos"] = np.array([0, 0.6, 0.2])

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )

    @property
    def model_name(self):
        # return full_v2_path_for("sawyer_xyz/sawyer_door_pull.xml")
        return os.path.join(
            os.path.dirname(__file__), "assets/sawyer_xyz/door_drawer.xml"
        )

    def reset_model(self):
        obs = super().reset_model()
        self.sim.model.site_pos[self.model.site_name2id("goal")] = self._target_pos
        return obs

    def _get_curr_obs_combined_no_goal(self):
        """Combines the end effector's {pos, closed amount} and the object(s)'
            {pos, quat} into a single flat observation. The goal's position is
            *not* included in this.
        Returns:
            np.ndarray: The flat observation array (18 elements)
        """
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)

        ## To Do: Replace with drawer and door pos
        drawer_pos = self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])
        door_pos = self.data.get_geom_xpos("handle").copy()
        obj_pos = np.concatenate([drawer_pos, door_pos])
        assert len(obj_pos) % 3 == 0

        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        if self.isV2:
            ## To Do: Replace with drawer and door quat
            drawer_quat = self.sim.data.get_body_xquat("drawer_link")
            door_quat = Rotation.from_matrix(
                self.data.get_geom_xmat("handle")
            ).as_quat()
            obj_quat = np.concatenate([drawer_quat, door_quat])
            assert len(obj_quat) % 4 == 0
            obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
            obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
                [
                    np.hstack((pos, quat))
                    for pos, quat in zip(obj_pos_split, obj_quat_split)
                ]
            )
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
        else:
            # is a v1 environment
            obs_obj_padded[: len(obj_pos)] = obj_pos
            assert len(obs_obj_padded) in self._obs_obj_possible_lens
            return np.hstack((pos_hand, obs_obj_padded))


cds_envs_dict = OrderedDict(
    (
        ("door-open-v2", DoorOpenEnvV2),
        ("door-close-v2", DoorCloseEnvV2),
        ("drawer-open-v2", DrawerOpenEnvV2),
        ("drawer-close-v2", DrawerCloseEnvV2),
    )
)

# cds_envs_dict = OrderedDict(
#     (
#         ("door-open-v2", SawyerDoorEnvV2),
#         ("door-close-v2", SawyerDoorCloseEnvV2),
#         ("drawer-open-v2", SawyerDrawerOpenEnvV2),
#         ("drawer-close-v2", SawyerDrawerCloseEnvV2),
#     )
# )

cds_envs_dict_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(cds_envs_dict.keys()).index(key)})
    for key, _ in cds_envs_dict.items()
}


class CDSEnvs(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = cds_envs_dict
        self._test_classes = OrderedDict()
        train_kwargs = cds_envs_dict_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )
        self._test_tasks = []


class DoorOpenEnvV1V2(DoorOpenEnvV2):
    @property
    def observation_space(self):
        obs_obj_max_len = self._obs_obj_max_len if self.isV2 else 6

        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable else self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0

        return (
            Box(
                np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, goal_high)),
            )
            if self.isV2
            else Box(
                np.hstack((self._HAND_SPACE.low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, obj_high, goal_high)),
            )
        )

    def _get_obs(self):
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # don't do frame stacking for now
        obs = np.hstack((curr_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs


class DoorCloseEnvV1V2(DoorCloseEnvV2):
    @property
    def observation_space(self):
        obs_obj_max_len = self._obs_obj_max_len if self.isV2 else 6

        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable else self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0

        return (
            Box(
                np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, goal_high)),
            )
            if self.isV2
            else Box(
                np.hstack((self._HAND_SPACE.low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, obj_high, goal_high)),
            )
        )

    def _get_obs(self):
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # don't do frame stacking for now
        obs = np.hstack((curr_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs


class DrawerOpenEnvV1V2(DrawerOpenEnvV2):
    @property
    def observation_space(self):
        obs_obj_max_len = self._obs_obj_max_len if self.isV2 else 6

        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable else self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0

        return (
            Box(
                np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, goal_high)),
            )
            if self.isV2
            else Box(
                np.hstack((self._HAND_SPACE.low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, obj_high, goal_high)),
            )
        )

    def _get_obs(self):
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # don't do frame stacking for now
        obs = np.hstack((curr_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs


class DrawerCloseEnvV1V2(DrawerCloseEnvV2):
    @property
    def observation_space(self):
        obs_obj_max_len = self._obs_obj_max_len if self.isV2 else 6

        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable else self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0

        return (
            Box(
                np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, goal_high)),
            )
            if self.isV2
            else Box(
                np.hstack((self._HAND_SPACE.low, obj_low, goal_low)),
                np.hstack((self._HAND_SPACE.high, obj_high, goal_high)),
            )
        )

    def _get_obs(self):
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # don't do frame stacking for now
        obs = np.hstack((curr_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs


cds_envs_dict_v1 = OrderedDict(
    (
        ("door-open-v1", DoorOpenEnvV1V2),
        ("door-close-v1", DoorCloseEnvV1V2),
        ("drawer-open-v1", DrawerOpenEnvV1V2),
        ("drawer-close-v1", DrawerCloseEnvV1V2),
    )
)
cds_envs_dict_kwargs_v1 = {
    key: dict(args=[], kwargs={"task_id": list(cds_envs_dict_v1.keys()).index(key)})
    for key, _ in cds_envs_dict_v1.items()
}


class CDSEnvsV1(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = cds_envs_dict_v1
        self._test_classes = OrderedDict()
        train_kwargs = cds_envs_dict_kwargs_v1
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )
        self._test_tasks = []
