"""TanhGaussianMLPPolicy."""
import numpy as np
import torch
import akro

from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal
from garage.torch.modules import GaussianMLPTwoHeadedModule
from garage.torch.policies.stochastic_policy import StochasticPolicy

from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
from garage.torch.modules.gaussian_mlp_module import GaussianMLPBaseModule
from environments.kitchen.spirl.spirl_skill_prior import load_skill_prior
from learning.utils import list_to_tensor, np_to_torch
from learning.utils.general import SuppressStdout


class MultiheadTanhGaussianMLPPolicy(StochasticPolicy):
    """
    Multihead TanhGaussianMLPPolicy for Multiheaded Multi-task policy
    """

    def __init__(
        self,
        env_spec,
        num_heads=1,
        policy_assigner=None,
        split_observation=None,
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
    ):
        super().__init__(env_spec, name="TanhGaussianPolicy")

        self._num_heads = num_heads
        self._policy_assigner = policy_assigner
        self.split_observation = split_observation or (lambda x: (x, x))
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._module = GaussianMLPMultiHeadedModule(
            num_heads=self._num_heads,
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

    def forward(self, observations, policy_id=None):
        """
        If task_id is provided, return that head of the policy, otherwise, extract from observation with split_observation and policy_assigner
        """
        ### ASDF: assumes task info is in observation before split_observation
        obss, tasks = self.split_observation(observations)
        curr_policies = self._policy_assigner.get_actions(tasks)[0]
        if policy_id is not None:
            curr_policies = torch.ones_like(curr_policies) * policy_id
        dists = self._module(obss)

        idx = list(range(len(curr_policies)))
        ret_means = dists.mean[curr_policies.long(), idx]
        ret_log_stds = dists.variance[curr_policies.long(), idx].sqrt().log()
        dist = TanhNormal(ret_means, ret_log_stds.exp())
        return dist, dict(mean=ret_means.cpu(), log_std=ret_log_stds.cpu())

    def get_all_actions(self, observation):
        """
        Gets action from every single head and returns as a list.
        """
        if not isinstance(observation, np.ndarray) and not isinstance(
            observation, torch.Tensor
        ):
            observation = self._env_spec.observation_space.flatten_n(observation)
        if isinstance(observation, np.ndarray) and len(observation.shape) > 1:
            observations = self._env_spec.observation_space.flatten_n(observation)
        elif isinstance(observation, torch.Tensor) and len(observation.shape) > 1:
            observations = torch.flatten(observation)
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observations = np_to_torch(observation)
            if not isinstance(observations, torch.Tensor):
                observations = list_to_tensor(observations)
            if isinstance(self._env_spec.observation_space, akro.Image):
                observations /= 255.0  # scale image
            observations = observations.unsqueeze(0)

            ### Forward
            dists = self._module(observations)

            all_actions = dists.sample().cpu().numpy()
            all_actions = [a[0][0] for a in np.split(all_actions, self._num_heads)]
            all_means = [
                m[0][0] for m in np.split(dists.mean.cpu().numpy(), self._num_heads)
            ]
            all_log_stds = [
                s[0][0]
                for s in np.split(
                    dists.variance.sqrt().log().cpu().numpy(), self._num_heads
                )
            ]
            all_infos = [
                dict(mean=m, log_std=s) for m, s in zip(all_means, all_log_stds)
            ]

            return all_actions, all_infos
            # dist, dict(mean=ret_means.cpu(), log_std=ret_log_stds.cpu())
            # return dist.sample().cpu().numpy(), {
            #     k: v.detach().cpu().numpy() for (k, v) in info.items()
            # }

    def get_action(self, observation, policy_id=None):
        """
        Overwrites and modifies get_action function from Stochastic Policy to specify a task_id.
        This is passed to get_actions()
        """
        if not isinstance(observation, np.ndarray) and not isinstance(
            observation, torch.Tensor
        ):
            observation = self._env_spec.observation_space.flatten(observation)
        elif isinstance(observation, np.ndarray) and len(observation.shape) > 1:
            observation = self._env_spec.observation_space.flatten(observation)
        elif isinstance(observation, torch.Tensor) and len(observation.shape) > 1:
            observation = torch.flatten(observation)
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = np_to_torch(observation)
            if not isinstance(observation, torch.Tensor):

                observation = list_to_tensor(observation)
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation, policy_id=policy_id)
            return action[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations, policy_id=None):
        """
        Overwrites and modifies get_actions function from Stochastic Policy to specify a task_id.
        This is passed to the forward function
        """
        if not isinstance(observations[0], np.ndarray) and not isinstance(
            observations[0], torch.Tensor
        ):
            observations = self._env_spec.observation_space.flatten_n(observations)

        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(observations[0], np.ndarray) and len(observations[0].shape) > 1:
            observations = self._env_spec.observation_space.flatten_n(observations)
        elif (
            isinstance(observations[0], torch.Tensor) and len(observations[0].shape) > 1
        ):
            observations = torch.flatten(observations, start_dim=1)
        with torch.no_grad():
            if isinstance(observations, np.ndarray):
                observations = np_to_torch(observations)
            if not isinstance(observations, torch.Tensor):

                observations = list_to_tensor(observations)

            if isinstance(self._env_spec.observation_space, akro.Image):
                observations /= 255.0  # scale image
            dist, info = self.forward(observations, policy_id=policy_id)
            return dist.sample().cpu().numpy(), {
                k: v.detach().cpu().numpy() for (k, v) in info.items()
            }


class MultiheadTanhGaussianSPiRLPolicy(MultiheadTanhGaussianMLPPolicy):
    def __init__(
        self,
        env_spec,
        num_heads=1,
        policy_assigner=None,
        split_observation=None,
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
    ):
        super().__init__(env_spec)

        self._num_heads = num_heads
        self._policy_assigner = policy_assigner
        self.split_observation = split_observation or (lambda x: (x, x))
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        ### To Do: need to do some surgery to make multiple heads with last layer of skill prior

        self._module = GaussianSPiRLMultiHeadedModule(
            num_heads=self._num_heads,
        )


class GaussianSPiRLMultiHeadedModule(nn.Module):
    def __init__(self, num_heads):
        self.num_heads = num_heads

        ### load multi-headed spirl prior
        print("Loading SPiRL skill prior ...")
        # with SuppressStdout():
        self._shared_mean_log_std_network = load_skill_prior(
            num_heads=self.num_heads
        )

    def _get_mean_and_log_std(self, *inputs):
        return self._shared_mean_log_std_network(*inputs)

    def forward(self, *inputs):
        dist = self._get_mean_and_log_std(*inputs)
        import ipdb

        ipdb.set_trace()
        means = torch.stack(dist.mu)
        log_std_uncentereds = torch.stack(dist.sigma.log())

        if self._min_std_param or self._max_std_param:
            log_std_uncentereds = log_std_uncentereds.clamp(
                min=(
                    None if self._min_std_param is None else self._min_std_param.item()
                ),
                max=(
                    None if self._max_std_param is None else self._max_std_param.item()
                ),
            )

        stds = log_std_uncentereds.exp()

        dists = self._norm_dist_class(means, stds)

        if not isinstance(dists, TanhNormal):
            dists = Independent(dists, 1)

        return dists


class GaussianMLPMultiHeadedModule(GaussianMLPBaseModule):
    """
    MultiHeaded Gaussian MLPModule which produces num_heads Gaussian distributions with shared network.
    """

    def __init__(
        self,
        num_heads,
        input_dim,
        output_dim,
        hidden_sizes=(32, 32),
        *,
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
        normal_distribution_cls=Normal
    ):
        self._num_heads = num_heads
        super().__init__(
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

        self._shared_mean_log_std_network = MultiHeadedMLPModule(
            n_heads=2 * self._num_heads,
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
            ]
            * self._num_heads,
            layer_normalization=self._layer_normalization,
        )

    def _get_mean_and_log_std(self, *inputs):
        """
        from GaussianMLPTwoHeadedModule
        """
        return self._shared_mean_log_std_network(*inputs)

    def forward(self, *inputs):
        """Forward method.
        Args:
            *inputs: Input to the module.
        Returns:
            num_heads torch.distributions.independent.Independent: Independent
                distributions.
        """
        mean_std_outputs = self._get_mean_and_log_std(*inputs)
        means = torch.stack(mean_std_outputs[::2])
        log_std_uncentereds = torch.stack(mean_std_outputs[1::2])

        if self._min_std_param or self._max_std_param:
            log_std_uncentereds = log_std_uncentereds.clamp(
                min=(
                    None if self._min_std_param is None else self._min_std_param.item()
                ),
                max=(
                    None if self._max_std_param is None else self._max_std_param.item()
                ),
            )

        if self._std_parameterization == "exp":
            stds = log_std_uncentereds.exp()
        else:
            stds = log_std_uncentereds.exp().exp().add(1.0).log()

        dists = self._norm_dist_class(means, stds)

        if not isinstance(dists, TanhNormal):
            dists = Independent(dists, 1)

        return dists
