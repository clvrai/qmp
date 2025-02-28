import abc
import numpy as np
import torch
from torch.distributions import Categorical

from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy


class TwoColumnWrapper(StochasticPolicy, abc.ABC):
    """Policy wrapper for two-column structure with central policy and task-specific policy.

    Args:
        central_policy (Policy): Central policy.
        policy (Policy): Task-specific policy.
        alpha (float): Coefficient for entropy (c_Ent).
        beta (float): Coefficient for KL (c_KL).

    """

    def __init__(self, central_policy, policy, alpha, beta):
        super().__init__(env_spec=policy.env_spec, name=policy.name)
        self._central_policy = central_policy
        self._policy = policy
        self._c_ent = alpha
        self._c_kl = beta

    def set_alpha(self, alpha):
        self._c_ent = alpha

    def set_beta(self, beta):
        self._c_kl = beta

    def forward(self, observations):
        action_dist = self._policy(observations)[0]
        action_mean, action_var = action_dist.mean, action_dist.variance
        central_dist = self._central_policy(observations)[0]
        central_mean, central_var = central_dist.mean, central_dist.variance

        # Get mean and variance of weighted mixture of Gaussians
        # Mixture ratio is alpha to beta, where alpha is c_kl / (c_ent + c_kl)
        # and beta is 1 / (c_ent + c_kl), hence equivalent to c_kl to 1.
        w1 = self._c_kl / (self._c_kl + 1.0)
        w2 = 1.0 / (self._c_kl + 1.0)
        avg_mean = w1 * central_mean + w2 * action_mean
        avg_var = (
            w1 * central_var + w2 * action_var
            + w1 * w2 * (central_mean - action_mean) ** 2
        )
        dist = TanhNormal(avg_mean, avg_var.sqrt())

        ret_mean = avg_mean.cpu()
        ret_log_std = (avg_var.sqrt()).log().cpu()
        return dist, dict(mean=ret_mean, log_std=ret_log_std)


class DiscreteTwoColumnWrapper(TwoColumnWrapper):
    def forward(self, observations):
        action_dist = self._policy(observations)[0]
        central_dist = self._central_policy(observations)[0]
        paper_alpha = self._c_kl / (self._c_ent + self._c_kl)
        paper_beta = 1.0 / (self._c_ent + self._c_kl)
        logits = (
            paper_alpha * central_dist.logits
            + paper_beta * action_dist.logits
        )
        return Categorical(logits=logits), dict()
