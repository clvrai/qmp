import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from garage.torch.policies.tanh_gaussian_mlp_policy import TanhGaussianMLPPolicy
from garage.torch.modules import GaussianMLPTwoHeadedModule, MLPModule
from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule

from environments.kitchen.spirl.spirl_skill_prior import load_skill_prior
from learning.utils.general import SuppressStdout


class DnCMultiHeadedMLPModule(MultiHeadedMLPModule):
    def get_representation(self, input_val):
        """returns last hidden layer of module"""
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return x


class DnCGaussianMLPTwoHeadedModule(GaussianMLPTwoHeadedModule):
    """From garage/torch/modules/gaussian_mlp_module.py"""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=torch.tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        learn_std=True,
        init_std=1.0,
        min_std=1e-6,
        max_std=None,
        std_parameterization="exp",
        layer_normalization=False,
        normal_distribution_cls=Normal,
    ):
        super(GaussianMLPTwoHeadedModule, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            normal_distribution_cls=normal_distribution_cls,
        )

        self._shared_mean_log_std_network = DnCMultiHeadedMLPModule(
            n_heads=2,
            input_dim=self._input_dim,
            output_dims=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=[
                nn.init.zeros_,
                lambda x: nn.init.constant_(x, self._init_std.item()),
            ],
            layer_normalization=self._layer_normalization,
        )

    def get_representation(self, inputs):
        return self._shared_mean_log_std_network.get_representation(inputs)


class NamedTanhGaussianMLPPolicy(TanhGaussianMLPPolicy):
    """
    - Names Policy with argument name
    - enables getting just representation: last hidden layer
    """

    def __init__(
        self,
        env_spec,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=nn.ReLU,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        init_std=1.0,
        min_std=np.exp(-20.0),
        max_std=np.exp(2.0),
        std_parameterization="exp",
        layer_normalization=False,
        name="TanhGaussianPolicy",
        split_observation=None,
    ):
        StochasticPolicy.__init__(self, env_spec, name=name)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self.split_observation = split_observation or (lambda x: (x, x))

        self._module = DnCGaussianMLPTwoHeadedModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            normal_distribution_cls=TanhNormal,
        )

    def get_representation(self, observations):
        rep = self._module.get_representation(observations)
        return rep

    def get_actions(self, observation):
        obs, task = self.split_observation(observation)
        return super().get_actions(obs)


class NamedTanhGaussianSPiRLPolicy(NamedTanhGaussianMLPPolicy):
    """
    - Names Policy with argument name
    - enables getting just representation: last hidden layer
    """

    def __init__(
        self,
        env_spec,
        min_std=np.exp(-20.0),
        max_std=np.exp(2.0),
        name="TanhGaussianPolicy",
        split_observation=None,
    ):
        StochasticPolicy.__init__(self, env_spec, name=name)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self.split_observation = split_observation or (lambda x: (x, x))
        self._normal_distribution_cls = TanhNormal
        self._min_std_param = torch.Tensor([min_std]).log()
        self._max_std_param = torch.Tensor([max_std]).log()

        print("Loading SPiRL skill prior ...")
        with SuppressStdout():
            self._module = load_skill_prior()

    def forward(self, observations):
        """Compute the action distributions from the observations.
        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
        """
        dist = self._module(observations)
        mean = dist.mu
        log_std = (dist.sigma).log()
        if self._min_std_param or self._max_std_param:
            log_std = log_std.clamp(
                min=(
                    None if self._min_std_param is None else self._min_std_param.item()
                ),
                max=(
                    None if self._max_std_param is None else self._max_std_param.item()
                ),
            )
        std = log_std.exp()

        ret_dist = self._normal_distribution_cls(mean, std)
        ret_mean = mean.cpu()
        ret_log_std = log_std.cpu()
        return ret_dist, dict(mean=ret_mean, log_std=ret_log_std)

    def get_representation(self, observations):
        raise NotImplementedError


class NamedTanhGaussianResidualSPiRLPolicy(NamedTanhGaussianMLPPolicy):
    def __init__(
        self,
        *args,
        min_std=np.exp(-20.0),
        max_std=np.exp(2.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._normal_distribution_cls = TanhNormal
        self._min_std_param = torch.Tensor([min_std]).log()
        self._max_std_param = torch.Tensor([max_std]).log()
        print("Loading SPiRL skill prior ...")
        with SuppressStdout():
            self._skill_prior = load_skill_prior()

    def forward(self, observations):
        prior_dist = self._skill_prior(observations)
        prior_mean = prior_dist.mu
        prior_log_std = (prior_dist.sigma).log()

        dist = self._module(observations)
        res_mean = dist.mean
        res_log_std = (dist.variance.sqrt()).log()

        mean = prior_mean + res_mean
        log_std = prior_log_std + res_log_std

        if self._min_std_param or self._max_std_param:
            log_std = log_std.clamp(
                min=(
                    None if self._min_std_param is None else self._min_std_param.item()
                ),
                max=(
                    None if self._max_std_param is None else self._max_std_param.item()
                ),
            )
        std = log_std.exp()

        ret_dist = self._normal_distribution_cls(mean, std)
        ret_mean = mean.cpu()
        ret_log_std = log_std.cpu()
        return ret_dist, dict(mean=ret_mean, log_std=ret_log_std)

    def get_representation(self, observations):
        raise NotImplementedError
