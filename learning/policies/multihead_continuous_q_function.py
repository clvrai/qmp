"""This modules creates a continuous Q-function network."""

import torch
import torch.nn.functional as F
from torch import nn

from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.q_functions import ContinuousMLPQFunction
class DECAFQFunction(MLPModule):
    """
    DECAF Q function.
    """

    def __init__(self,
                 env_spec,
                 all_q_functions,
                 task_id,
                 hidden_sizes,
                 hidden_nonlinearity,
                 layer_normalization,
                 **kwargs,
                 ):

        self._env_spec = env_spec
        self._n_tasks = len(all_q_functions)
        self._task_id = task_id
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        super().__init__(input_dim=self._obs_dim+self._action_dim,
                       output_dim=self._n_tasks,
                       hidden_sizes=hidden_sizes,
                       output_nonlinearity=F.softmax,
                       hidden_nonlinearity=hidden_nonlinearity,
                       layer_normalization=layer_normalization,
                       **kwargs)

        self._all_q_functions = all_q_functions
        self._task_q_function = nn.ModuleList([self._all_q_functions[self._task_id]])

    def forward(self, observations, actions):
        input = torch.cat([observations, actions], 1)
        weights = super().forward(input)
        qvalues = []
        for i in range(self._n_tasks):
            if i == self._task_id:
                q = self._all_q_functions[i](observations, actions)
            else:
                with torch.no_grad():
                    q = self._all_q_functions[i](observations, actions)
            qvalues.append(q)

        qvalues = torch.cat(qvalues, dim=-1)
        q_train = torch.multiply(qvalues, weights)
        q_train = torch.sum(q_train, dim=-1)
        return q_train
    # def __deepcopy__(self, memo):
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     result.__dict__.update(self.__dict__)
    #     for i in range(self._n_tasks):
    #         if i != self._task_id:
    #             result.__dict__["_all_q_functions"][i] = self._all_q_functions[i]
    #     return result


class MultiheadContinuousMLPQFunction(MultiHeadedMLPModule):
    """Implements a continuous MLP Q-value network.
    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(
        self,
        env_spec,
        num_heads,
        policy_assigner=None,
        split_observation=None,
        **kwargs
    ):
        """Initialize class with multiple attributes.
        Args:
            env_spec (EnvSpec): Environment specification.
            num_heads (int): Number of network heads
            **kwargs: Keyword arguments.
        """
        self._env_spec = env_spec
        self._n_heads = num_heads
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MultiHeadedMLPModule.__init__(
            self,
            n_heads=self._n_heads,
            input_dim=self._obs_dim + self._action_dim,
            output_dims=1,
            **kwargs
        )

        self._policy_assigner = policy_assigner
        self.split_observation = split_observation or (lambda x: (x, x))

    # pylint: disable=arguments-differ
    def forward(self, observations, actions):
        """Return Q-value(s).
        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.
        Returns:
            torch.Tensor: Output value
        """
        obss, tasks = self.split_observation(observations)
        curr_policies = self._policy_assigner.get_actions(tasks)[0]
        idx = list(range(len(curr_policies)))
        qvalues = super().forward(torch.cat([obss, actions], 1))
        qvalues = torch.cat(qvalues, dim=-1)
        qvalues = qvalues[idx, curr_policies.long()]
        return qvalues
