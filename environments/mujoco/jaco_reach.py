import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from environments.mujoco.jaco import JacoEnv
from environments.mujoco.asset_utils import get_asset_path


class JacoReachEnv(JacoEnv):
    def __init__(self, with_rot=1, no_finger=False):
        super().__init__(with_rot=with_rot)
        self._no_finger = no_finger

        # config
        self._config.update({
            "dist_reward": 1,
            "random_target": 1,
            "init_randomness": 0.01,
            "random_steps": 10,
            "hold_duration": 50,
        })

        # state
        self._t = 0
        self._hold_duration = 0
        self._picked = False
        self._pick_height = 0
        self._dist_target = 0

        # env info
        self.reward_type += ["dist_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        asset = "jaco_reach_nofinger.xml" if no_finger else "jaco_reach.xml"
        mujoco_env.MujocoEnv.__init__(self, get_asset_path(asset), 2)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self._t += 1
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        ctrl_reward = self._ctrl_reward(a)
        dist_target = self._get_distance_hand('target')
        dist_reward = -dist_target
        success = dist_target < 0.03
        # done = success
        done = False

        reward = ctrl_reward + dist_reward
        info = {"ctrl_reward": ctrl_reward,
                "dist_reward": dist_reward,
                "success": success}
        return ob, reward, done, info

    def _get_hand_pos(self):
        return self._get_pos('jaco_link_hand')

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        hand_pos = self._get_hand_pos()
        if self._no_finger:
            qpos = np.concatenate([qpos, np.zeros(3)])
            qvel = np.concatenate([qvel, np.zeros(3)])
            qacc = np.concatenate([qacc, np.zeros(3)])
        dummy_pos = np.zeros(7)
        dummy_vel = np.zeros(6)
        dummy_acc = np.zeros(6)
        return np.concatenate([qpos, dummy_pos,
                               np.clip(qvel, -30, 30), dummy_vel,
                               qacc, dummy_acc,
                               hand_pos]).ravel().astype(np.float32)

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :31],
                'acc': ob[:, 31:46],
                'hand': ob[:, 46:49]
            }
        else:
            return {
                'joint': ob[:31],
                'acc': ob[31:46],
                'hand': ob[46:49]
            }

    def reset_model(self):
        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nv)
        target_xy = np.random.uniform(low=-0.5, high=0.5, size=2)  # target x, y
        target_z = np.random.uniform(low=0.05, high=1, size=1)  # target z
        self.sim.data.mocap_pos[0] = np.concatenate([target_xy, target_z])
        self.set_state(qpos, qvel)

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        return self._get_obs()
