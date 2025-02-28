import numpy as np
import torch
import gym
from gym.spaces import Box


class SkillWrapper(gym.Wrapper):
    def __init__(
        self,
        env=None,
        model=None,
        aggregate_infos=None,
        accumulate_reward=True,
        discount=0.99,
    ):
        """
        New action space: skill space
        Load spirl skill decoder
        Define n_steps
        """
        self._env = env
        gym.Wrapper.__init__(self, env)
        self.model = model
        assert aggregate_infos is not None
        self.aggregate_infos = aggregate_infos
        self.accumulate_reward = accumulate_reward
        self.n_steps = 10
        self._discount = discount
        self.action_space = Box(
            low=np.array([-2] * 10), high=np.array([2] * 10)
        )  ### Fix this --> for uniform prior is ok?
        self._last_obs = None
        self._frames = []
        self._record_frames = False

    def reset(self):
        obs = super().reset()

        self._last_obs = obs
        self._frames = []
        return obs

    def step(self, action, width=400, height=400):
        """
        Decode skill then roll out environment for n steps
        Discount reward?
        stack observations?
        rendering might be messed up
        """
        z = torch.tensor([action], dtype=torch.float)
        accumulated_reward = 0
        done = False
        ### How to aggregate infos?
        infos = {}
        self._frames = []
        for i in range(self.n_steps):
            with torch.no_grad():
                obs = torch.tensor([self._last_obs], dtype=torch.float)
                env_action = self.model.get_skill_action(obs=obs, skill=z)
                env_action = env_action.action[0].cpu().detach().numpy()
            ob, reward, done, info = self.env.step(env_action)
            # print(info.keys())
            if infos == {}:
                infos = info
            else:
                if self.aggregate_infos:
                    infos = self.aggregate_infos(infos, info)
                else:
                    infos = info

            ## No Discount
            if self.accumulate_reward:
                accumulated_reward += reward  # self._discount ** i * reward
            else:
                accumulated_reward = reward
            if done:
                break
            self._last_obs = ob
            if self._record_frames:
                ### HACK for karl's kitchen
                self._frames.append(
                    self.env.render(mode="rgb_array")  # , width=width, height=height)
                )
        return ob, accumulated_reward, done, infos

    def set_record_frames(self, record_frames):
        self._record_frames = record_frames

    def render(self, mode="human", width=400, height=400):
        if len(self._frames) > 0:
            return self._frames
        else:
            ### HACK for karl's kitchen
            return super().render(mode=mode)  # , width=width, height=height)
