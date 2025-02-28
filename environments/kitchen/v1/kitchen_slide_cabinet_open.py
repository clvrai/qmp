import numpy as np

from environments.kitchen import reward_utils
from environments.kitchen.v0.kitchen_v0 import KitchenSingleTaskEnv


class KitchenSlideCabinetOpenEnvV1(KitchenSingleTaskEnv):

    TASK_NAME = "slide cabinet-open"
    BONUS_THRESH = 0.1
    handle_reach_radius = 0.07
    obj_radius = 0.022  ### Update 0.022 (cabinet)

    def _compute_reward(self, obs_dict, dists):
        raise NotImplementedError()
        ### need to make into class method
        in_place = reward_utils.tolerance(
            dists["goal"],
            bounds=(0, self.BONUS_THRESH),
            margin=abs(dists["goal_init"] - self.BONUS_THRESH),
            sigmoid="long_tail",
        )

        object_grasped = self._gripper_caging_reward(dists)

        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped, in_place
        )
        reward = in_place_and_object_grasped

        ### To test if object is grasped?
        # if (
        #     dists["hand"] < self.handle_reach_radius
        #     and (
        #         0.05 > dists["gripper"] * 0.09 > self.obj_radius
        #     )  # Unnormalize gripper dist
        #     and (dists["goal"] < dists["goal_init"])
        # ):
        #     reward += 1.0 + 5.0 * in_place
        # if dists["goal"] < self.BONUS_THRESH:
        #     reward = 10.0

        if dists["goal"] < self.BONUS_THRESH:
            reward = 1.0
        return reward

    def _gripper_caging_reward(self, dists):
        pad_success_margin = 0.05  ### Update?
        y_z_success_margin = 0.005  ### Update?

        # Gripper average and finger positions
        obj_position = self._get_obj_pos()

        ### For kitchen, grasping axis is x
        delta_object_x_left_pad = dists["left_pad_x"]
        delta_object_x_right_pad = dists["right_pad_x"]
        right_caging_margin = abs(
            abs(obj_position[0] - self._init_right_pad[0]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_position[0] - self._init_left_pad[0]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_x_right_pad,
            bounds=(self.obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_x_left_pad,
            bounds=(self.obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        x_caging = reward_utils.hamacher_product(left_caging, right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_obj_norm_y_z = dists["hand_yz"]

        # used for computing the tcp to object object margin in the x_z plane
        tcp_obj_y_z_margin = abs(self._dist_hand_yz_init - y_z_success_margin)

        y_z_caging = reward_utils.tolerance(
            tcp_obj_norm_y_z,
            bounds=(0, y_z_success_margin),
            margin=tcp_obj_y_z_margin,
            sigmoid="long_tail",
        )

        gripper_closed = 1.0 - dists["gripper"]
        caging = reward_utils.hamacher_product(x_caging, y_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        caging_and_gripping = (caging_and_gripping + caging) / 2
        return caging_and_gripping
