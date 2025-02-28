import copy
import warnings

import akro
import gym
import numpy as np

from garage import Environment, EnvSpec, EnvStep, StepType
from garage.envs import GymEnv


def _get_time_limit(env, max_episode_length):
    """
    from garage.envs
    """
    spec_steps = None
    if hasattr(env, "spec") and env.spec and hasattr(env.spec, "max_episode_steps"):
        spec_steps = env.spec.max_episode_steps
    elif hasattr(env, "_max_episode_steps"):
        spec_steps = getattr(env, "_max_episode_steps")

    if spec_steps:
        if max_episode_length is not None and max_episode_length != spec_steps:
            warnings.warn(
                "Overriding max_episode_length. Replacing gym time"
                "limit ({}), with {}".format(spec_steps, max_episode_length)
            )
            return max_episode_length
        return spec_steps
    return max_episode_length


class ArgsGymEnv(GymEnv):
    """Garage gym environment but passes arguments to gym.make.  Overwrote __deepcopy__ so env_args gets passed to copies"""

    def __init__(self, env, env_args={}, is_image=False, max_episode_length=None):

        self.env_args = env_args
        self.env_name = env
        self._env = None
        if isinstance(env, str):
            self._env = gym.make(env, **env_args)
        elif isinstance(env, gym.Env):
            self._env = env
        else:
            raise ValueError(
                "GymEnv can take env as either a string, "
                "or an Gym environment, but got type {} "
                "instead.".format(type(env))
            )

        super().__init__(
            self._env, is_image=is_image, max_episode_length=max_episode_length
        )

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls, self.env_name, self.env_args)
        result.__init__(self.env_name, self.env_args)
        return result
