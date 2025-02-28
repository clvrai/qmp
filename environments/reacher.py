import numpy as np
import os
import random

from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.reacher import ReacherEnv


class GymReacherEnv(ReacherEnv):
    """Wrapper for Gym Reacher that record success rate"""

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        success = -reward_dist < 0.02
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return (
            ob,
            reward,
            done,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, success=success),
        )


class ReacherRandomizedEnv(GymReacherEnv):
    """Reacher with completely randomized goal and initial position."""

    def __init__(self):
        super().__init__()

    def reset_model(self):
        self.init_qpos = np.array([0.0, 0.0, 0.0, 0.0])
        joint1 = self.np_random.uniform(low=-np.pi, high=np.pi)
        joint2 = self.np_random.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
        qpos = self.init_qpos.copy()
        qpos[0] += joint1
        qpos[1] += joint2

        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            dist_from_origin = np.linalg.norm(self.goal)
            if 0.1 < dist_from_origin < 0.2:
                break

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_quadrant(self, pos):

        ### ASDF quadrant numbers are incorrect, 2 & 3 are swapped

        quadrants = 2 * (pos[:, 0] < 0).astype(np.uint8) + (pos[:, 1] < 0).astype(
            np.uint8
        )
        return quadrants

    def get_goal_quadrant_partition(self, observation):
        if len(observation.shape) > 1:
            goal_pos = observation[:, 4:6]
            return self._get_quadrant(goal_pos)
        else:
            goal_pos = np.array([observation[4:6]])
            return self._get_quadrant(goal_pos)[0]


class ReacherClusterEnv(ReacherRandomizedEnv):
    """Reacher environment with 2 - 4 goal clusters and modifiable reward."""

    def __init__(self, reward_type="shift", reward_params=[0, 0]):
        self.cluster_id = 0
        self.num_clusters = len(reward_params)
        self.cluster_locations = [
            {"xlow": 0.08, "xhigh": 0.12, "ylow": -0.02, "yhigh": 0.02},
            {"xlow": -0.14, "xhigh": -0.1, "ylow": -0.14, "yhigh": -0.1},
            {"xlow": -0.12, "xhigh": -0.08, "ylow": 0.08, "yhigh": 0.12},
            {"xlow": 0.06, "xhigh": 0.1, "ylow": -0.14, "yhigh": -0.1},
        ]
        assert reward_type in ["scale", "shift", "shaped"]
        self.reward_type = reward_type
        self.reward_params = reward_params
        self._count = np.random.randint(0, self.num_clusters)
        super().__init__()

    def reset_model(self):
        self.init_qpos = np.array([0.0, 0.0, 0.0, 0.0])

        ### fix initial start position to quadrant 1 and have small variability
        joint1 = np.pi / 2 + self.np_random.uniform(
            low=-0.1 * np.pi / 2, high=0.1 * np.pi / 2
        )
        joint2 = -np.pi / 2 + self.np_random.uniform(
            low=-0.1 * np.pi / 2, high=0.1 * np.pi / 2
        )
        qpos = self.init_qpos.copy()
        qpos[0] += joint1
        qpos[1] += joint2

        ### Flip Coin and sample goal from cluster
        self.cluster_id = self._count % self.num_clusters
        # self.cluster_id = np.random.randint(0, self.num_clusters)
        self._count += 1
        bounds = self.cluster_locations[self.cluster_id]
        self.goal = [
            self.np_random.uniform(low=bounds["xlow"], high=bounds["xhigh"]),
            self.np_random.uniform(low=bounds["ylow"], high=bounds["yhigh"]),
        ]

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        dist = np.linalg.norm(vec)
        reward_dist = -dist
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        ### ASDF make reward scale even worse
        if self.reward_type == "scale":
            reward = reward * self.reward_params[self.cluster_id]

        ### ASDF reward shift
        elif self.reward_type == "shift":
            reward = reward + self.reward_params[self.cluster_id]

        ### ASDF make shaped reward
        elif self.reward_type == "shaped":
            ### 0: distance, 1: sparse, 2: sparse checkpoint?
            if self.reward_params[self.cluster_id] == 0:
                shaped_reward_dist = -dist
            elif self.reward_params[self.cluster_id] == 1:
                shaped_reward_dist = dist < 0.01
            elif self.reward_params[self.cluster_id] == 2:
                if dist < 0.01:
                    shaped_reward_dist = 1
                elif dist < 0.2:
                    shaped_reward_dist = 0.5
                else:
                    shaped_reward_dist = 0
            elif self.reward_params[self.cluster_id] == 3:
                x, y, _ = self.get_body_com("fingertip")
                if dist < 0.01:
                    shaped_reward_dist = 1
                elif x > 0 and y > 0:
                    shaped_reward_dist = -0.5
                elif x * y < 0:
                    shaped_reward_dist = 0
                else:
                    shaped_reward_dist = 0.5

            elif self.reward_params[self.cluster_id] == 4:
                x, y, _ = self.get_body_com("fingertip")
                if dist < 0.01:
                    shaped_reward_dist = 1
                elif x > 0 and y > 0:
                    shaped_reward_dist = 0
                elif x * y < 0:
                    shaped_reward_dist = -0.5
                else:
                    shaped_reward_dist = 0.5

            else:
                shaped_reward_dist = (
                    0.2 * np.exp(-(dist ** 2) / self.reward_params[self.cluster_id])
                    - 0.2
                )
            shaped_reward = shaped_reward_dist + reward_ctrl
            reward = shaped_reward

        success = -reward_dist < 0.01
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return (
            ob,
            reward,
            done,
            dict(
                reward_dist=reward_dist,
                reward_ctrl=reward_ctrl,
                success=success,
                task_id=self.cluster_id,
            ),
        )

    def _get_goal_cluster(self, goal):

        self.cluster_locations = [
            {"xlow": 0.08, "xhigh": 0.12, "ylow": -0.02, "yhigh": 0.02},
            {"xlow": -0.14, "xhigh": -0.1, "ylow": -0.14, "yhigh": -0.1},
            {"xlow": -0.12, "xhigh": -0.08, "ylow": 0.08, "yhigh": 0.12},
            {"xlow": 0.06, "xhigh": 0.1, "ylow": -0.14, "yhigh": -0.1},
        ]

        if len(goal) != 1:
            import ipdb

            ipdb.set_trace()
            if self.num_clusters == 2:
                return goal > 0
            else:
                raise NotImplementedError
            ### ASDF fix this

        if goal[0, 0] > 0:
            if goal[0, 1] <= self.cluster_locations[3]["yhigh"]:
                return [3]
            else:
                return [0]
        else:
            if goal[0, 1] <= self.cluster_locations[1]["yhigh"]:
                return [1]
            else:
                return [2]

    def get_goal_cluster(self, observation):
        if len(observation.shape) > 1:
            goal_pos = observation[:, 4:6]
            return self._get_goal_cluster(goal_pos)
        else:
            goal_pos = np.array([observation[4:6]])
            return self._get_goal_cluster(goal_pos)[0]


class ReacherRewardScaleEnv(ReacherClusterEnv):
    def __init__(self):
        super().__init__(reward_type="scale", reward_params=[1, 10])


class ReacherRewardShiftEnv(ReacherClusterEnv):
    def __init__(self):
        super().__init__(reward_type="shift", reward_params=[0, -5])


class ReacherRewardShapedEnv(ReacherClusterEnv):
    def __init__(self):
        super().__init__(reward_type="shaped", reward_params=[None, 0.001])


class ReacherGoalClusterEnv(ReacherClusterEnv):
    def __init__(self, cluster_id):
        super().__init__(reward_type="shift", reward_params=[-5] * 2)
        self.cluster_id = cluster_id

    def reset_model(self):
        self.init_qpos = np.array([0.0, 0.0, 0.0, 0.0])

        ### fix initial start position to quadrant 1 and have small variability
        joint1 = np.pi / 2 + self.np_random.uniform(
            low=-0.1 * np.pi / 2, high=0.1 * np.pi / 2
        )
        joint2 = -np.pi / 2 + self.np_random.uniform(
            low=-0.1 * np.pi / 2, high=0.1 * np.pi / 2
        )
        qpos = self.init_qpos.copy()
        qpos[0] += joint1
        qpos[1] += joint2

        ### Flip Coin and sample goal from cluster
        bounds = self.cluster_locations[self.cluster_id]
        self.goal = [
            self.np_random.uniform(low=bounds["xlow"], high=bounds["xhigh"]),
            self.np_random.uniform(low=bounds["ylow"], high=bounds["yhigh"]),
        ]

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()


class ReacherSlowObstacleEnv(ReacherRandomizedEnv):
    """
    Reacher Randomized environment with clockwise or counter-clockwise obstacle.
    Currently super slow because I reload the xml model each reset
    """

    def __init__(self):
        self.model_files = [
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "assets/reacher_obstacle{}.xml".format(i),
            )
            for i in range(4)
        ]
        utils.EzPickle.__init__(self)
        self._obstacle_id = 0
        mujoco_env.MujocoEnv.__init__(
            self,
            self.model_files[0],
            2,
        )

        self.geom_names_to_indices = {
            name: index for index, name in enumerate(self.model.geom_names)
        }
        self.contacts = None
        self._obstacle_threshold = 0.01

    def reset_model(self):

        self._obstacle_id = np.random.choice([0, 1, 2, 3])
        mujoco_env.MujocoEnv.__init__(
            self,
            self.model_files[self._obstacle_id],
            2,
        )
        self.geom_names_to_indices = {
            name: index for index, name in enumerate(self.model.geom_names)
        }

        ### Setting reacher position
        while True:
            self.sim.reset()
            joint1 = self.np_random.uniform(low=-np.pi, high=np.pi)
            joint2 = self.np_random.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
            qpos = self.init_qpos.copy()
            qpos[0] += joint1
            qpos[1] += joint2

            ### Setting goal
            while True:
                if self._obstacle_id == 0:
                    # self.goal = self.np_random.uniform(low=(0, -0.2), high=(0.2, 0.2))
                    self.goal = random.choice(
                        (
                            self.np_random.uniform(
                                low=(0, -0.2), high=(0.2, -self._obstacle_threshold)
                            ),
                            self.np_random.uniform(
                                low=(0, self._obstacle_threshold), high=(0.2, 0.2)
                            ),
                        )
                    )
                elif self._obstacle_id == 1:
                    self.goal = random.choice(
                        (
                            self.np_random.uniform(
                                low=(-0.2, 0), high=(-self._obstacle_threshold, 0.2)
                            ),
                            self.np_random.uniform(
                                low=(self._obstacle_threshold, 0), high=(0.2, 0.2)
                            ),
                        )
                    )
                elif self._obstacle_id == 2:
                    self.goal = random.choice(
                        (
                            self.np_random.uniform(
                                low=(-0.2, -0.2), high=(0, -self._obstacle_threshold)
                            ),
                            self.np_random.uniform(
                                low=(-0.2, self._obstacle_threshold), high=(0, 0.2)
                            ),
                        )
                    )
                elif self._obstacle_id == 3:
                    self.goal = random.choice(
                        (
                            self.np_random.uniform(
                                low=(-0.2, -0.2), high=(-self._obstacle_threshold, 0)
                            ),
                            self.np_random.uniform(
                                low=(self._obstacle_threshold, -0.2), high=(0.2, 0)
                            ),
                        )
                    )
                dist_from_origin = np.linalg.norm(self.goal)
                if 0.1 <= dist_from_origin <= 0.2:
                    break

            ### Other Stuff
            qpos[-2:] = self.goal
            qvel = self.init_qvel + self.np_random.uniform(
                low=-0.005, high=0.005, size=self.model.nv
            )
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            if self._check_collision() == False:
                return self._get_obs()

    def _check_collision(self):
        # Checks collision of reacher and goal with the obstacle.
        obstacle_name = "obstacle{}".format(self._obstacle_id)
        other_names = ["link0", "link1", "fingertip", "target"]
        idx1 = self.geom_names_to_indices[obstacle_name]
        idx2 = [self.geom_names_to_indices[name] for name in other_names]
        for contact in self.data.contact:
            if (contact.geom1 == idx1 and contact.geom2 in idx2) or (
                contact.geom1 in idx2 and contact.geom2 == idx1
            ):
                self.contacts = self.data.contact
                return True
        return False

    def _get_obstacle_orientation(self, obstacle_id, goal_quadrant):
        return (obstacle_id != goal_quadrant).astype(np.uint8)  # clockwise

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
                np.array([self._obstacle_id]),
            ]
        )

    def get_obstacle_id_partition(self, observation):
        if len(observation.shape) > 1:
            obstacle_id = observation[:, -1].astype(np.uint8)
        else:
            obstacle_id = observation[-1].astype(np.uint8)
        return obstacle_id

    def get_obstacle_orientation_partition(self, observation):
        ### ASDF only for two policies

        if len(observation.shape) > 1:
            goal_pos = observation[:, 4:6]
            goal_quadrant = self._get_quadrant(goal_pos)
            obstacle_id = observation[:, -1]
        else:
            goal_pos = np.array([observation[4:6]])
            goal_quadrant = self._get_quadrant(goal_pos)[0]
            obstacle_id = observation[-1]

        obstacle_orientation = self._get_obstacle_orientation(
            obstacle_id, goal_quadrant
        )
        return obstacle_orientation
