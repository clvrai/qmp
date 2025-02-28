import abc
import numpy as np

from garage.torch.policies.policy import Policy


class EnvPartitionPolicy(Policy, abc.ABC):
    """Policy assigner with fixed environment partition.

    Args:
        mode ("random", "fixed")
        partitions should be a num_partitions long list of functions

    """

    def __init__(
        self, env_spec, mode, num_partitions, partitions=None, name="EnvPartitionPolicy"
    ):
        super().__init__(env_spec, name)

        self._mode = mode
        self._num_partitions = num_partitions
        self._partitions = partitions
        self._new_episode = True
        self._curr_action = None

    def get_action(self, observation):

        if self._new_episode:

            if self._mode == "random":
                self._curr_action = np.random.randint(0, self._num_partitions)
            else:
                self._curr_action = self._partitions(observation)
            self._new_episode = False

        return self._curr_action, {}

    def get_actions(self, observations):
        if self._mode == "random":
            actions = np.random.randint(0, self._num_partitions, len(observations))
        else:
            actions = self._partitions(observations)

        return actions, {}

    def reset(self, do_resets=None):
        self._new_episode = True
