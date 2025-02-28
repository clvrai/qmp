import abc
import numpy as np

from garage.torch.policies.policy import Policy


def sample_categorical(p_vals):
    samples = np.random.multinomial(1, p_vals).argmax()
    return samples


class MixtureOfPoliciesWrapper(Policy, abc.ABC):
    """Policy wrapper for methods with ensembles of policies with a once-per-episode policy_assigner.

    Args:
        policies (list[Policy]): list of Policy objects that make up the policy ensemble
        task_identifier (Policy): Policy object that returns task_id given the current observation
        mixture_probs (n_policies x n_policies Numpy Array): Contains sampling probabilities for mixture
        sampling_freq : frequency at which to resample a policy from the mixture
    """

    def __init__(
        self,
        policies,
        task_identifier,
        mixture_probs,
        sampling_freq=1,
        split_observation=None,
        stagewise=False,
        stage_identifier=None,
    ):
        super().__init__(env_spec=None, name="MixtureOfPoliciesWrapper")

        self.policies = policies
        self.task_identifier = task_identifier
        self.mixture_probs = mixture_probs
        self.sampling_freq = sampling_freq
        self.split_observation = split_observation or (lambda x: (x, x))
        self.stagewise = stagewise
        self.stage_identifier = stage_identifier

        self._curr_policy = None
        self._curr_policies = None
        self._count = 0
    
    def get_action(self, observation, evaluation_mode="mop"):
        """
        Args:
            observation:
            evaluation_mode:
                - mop: mixture of policies
                - p: task policy
                - !p: not task policy, switch 0 and 1
        """

        obs, task = self.split_observation(observation)

        task_id = self.task_identifier.get_action(task)[0]

        if evaluation_mode == "p":
            self._curr_policy = task_id
        if evaluation_mode == "mop" and self._count % self.sampling_freq == 0:
            if self.stagewise:
                stage_id = self.stage_identifier(obs)  ### observation or obs?
                self._curr_policy = sample_categorical(
                    self.mixture_probs[task_id][stage_id]
                )
            else:
                self._curr_policy = sample_categorical(self.mixture_probs[task_id])

        curr_action, curr_action_info = self.policies[self._curr_policy].get_action(obs)

        self._count += 1
        policy_id = task_id

        return (
            curr_action,
            {
                **curr_action_info,
                "policy_id": policy_id,
                "real_policy_id": self._curr_policy,
                "task_id": task_id,
            },
        )

    def get_actions(self, observations, evaluation_mode="mop"):
        raise NotImplementedError
        obss, tasks = self.split_observation(observations)

        task_ids = self.task_identifier.get_actions(tasks)[0]

        ### ASDF : vectorize
        actions, infos = [], []

        for (observation, task_id) in zip(obss, task_ids):
            if not train:
                self._curr_policies = task_id
            if train and self._count % self.sampling_freq == 0:
                self._curr_policies = sample_categorical(
                    self.mixture_probs[curr_policy]
                )
            a, i = self.policies[curr_policy].get_action(observation)
            actions.append(a)
            infos.append(i)
        curr_actions = np.vstack(actions)
        keys = infos[0].keys()
        infos = {k: np.vstack([i[k] for i in infos]) for k in keys}

        self._count += 1

        return curr_actions, {"policy_id": self._curr_policies, **infos}

    def reset(self, do_resets=None):
        self.task_identifier.reset(do_resets=do_resets)

        for policy in self.policies:
            policy.reset(do_resets=do_resets)

        self._curr_policy = None
        self._curr_policies = None
        self._count = 0

    def get_param_values(self):
        task_identifier_params = self.task_identifier.state_dict()
        policies_params = [policy.state_dict() for policy in self.policies]
        mixture_probs = self.mixture_probs

        return {
            "task_identifier": task_identifier_params,
            "policies": policies_params,
            "mixture_probs": mixture_probs,
        }

    def set_param_values(self, state_dict):

        self.task_identifier.load_state_dict(state_dict["task_identifier"])

        for i in range(len(self.policies)):
            self.policies[i].load_state_dict(state_dict["policies"][i])

        self.mixture_probs = state_dict["mixture_probs"]
