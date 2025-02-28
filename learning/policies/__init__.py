import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions import Normal

from garage.torch.policies.tanh_gaussian_mlp_policy import TanhGaussianMLPPolicy
from garage.torch.modules import GaussianMLPTwoHeadedModule, MLPModule
from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule

from learning.policies.multihead_policy import (
    MultiheadTanhGaussianMLPPolicy,
    MultiheadTanhGaussianSPiRLPolicy,
)
from learning.policies.multihead_continuous_q_function import (
    MultiheadContinuousMLPQFunction,
)
from learning.policies.tanh_policy import (
    NamedTanhGaussianMLPPolicy,
    NamedTanhGaussianSPiRLPolicy,
    NamedTanhGaussianResidualSPiRLPolicy,
)


class DiscreteMLPQFunction(MLPModule):
    ### only difference to garage.torch.modules is output_dim argument instead of output_dim = env_spec.action_space
    def __init__(
        self,
        env_spec,
        output_dim,
        hidden_sizes,
        hidden_nonlinearity=nn.ReLU,
        hidden_w_init=nn.init.xavier_normal_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_normal_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
    ):

        input_dim = env_spec.observation_space.flat_dim
        super().__init__(
            input_dim,
            output_dim,
            hidden_sizes,
            hidden_nonlinearity,
            hidden_w_init,
            hidden_b_init,
            output_nonlinearity,
            output_w_init,
            output_b_init,
            layer_normalization,
        )


class CategoricalMLPPolicy(StochasticPolicy):
    def __init__(
        self,
        env_spec,
        output_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
        name="CategoricalPolicy",
    ):
        StochasticPolicy.__init__(self, env_spec, name=name)

        self._obs_dim = env_spec.observation_space.flat_dim

        self._module = MLPModule(
            input_dim=self._obs_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
        )

    def forward(self, observations):
        logits = self._module(observations)
        return Categorical(logits=logits), dict()
