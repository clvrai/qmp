import abc
import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F

from garage.torch.policies.policy import Policy
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch import as_torch


def sample_categorical(p_vals):
    if np.sum(p_vals) != 1:
        if np.abs(np.sum(p_vals) - 1) > 1e-5:
            raise ValueError(
                "p_vals must sum to 1, but instead sum to {}".format(np.sum(p_vals))
            )
        else:
            p_vals = p_vals / np.sum(p_vals)
    samples = np.random.multinomial(1, p_vals).argmax()
    return samples

def compute_gaussian_entropy(mean, log_std):
    # Number of dimensions
    k = mean.shape[1]
    # Constant term
    constant_term = 0.5 * (1 + np.log(2 * np.pi))
    # Sum of log_std
    sum_log_std = np.sum(log_std)
    # Entropy
    entropy = k * constant_term + sum_log_std
    return entropy

class QswitchMixtureOfPoliciesWrapper(StochasticPolicy):
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
        policy_architecture,
        score_functions,
        score_function2s,
        score_architecture,
        task_identifier,
        sampling_freq=1,
        split_observation=None,
    ):
        super().__init__(env_spec=None, name="MixtureOfPoliciesWrapper")

        self.policies = policies
        self.policy_architecture = policy_architecture
        self.score_functions = score_functions
        self.score_function2s = score_function2s
        self.score_architecture = score_architecture
        self.task_identifier = task_identifier
        self.sampling_freq = sampling_freq
        self.split_observation = split_observation or (lambda x: (x, x))

        self._curr_policy = None
        self._curr_policies = None
        self._count = 0

    def parameters(self):
        #### HACK HACK HACK
        assert self.policy_architecture == "multihead"
        return self.policies[0].parameters()

    def forward(self, obs):
        #### HACK HACK HACK
        assert self.policy_architecture == "multihead"
        return self.policies[0](obs)

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
            curr_policy = task_id
            curr_action, curr_action_info = self._get_policy_action(
                obs, observation, curr_policy
            )
            self._curr_policy = curr_policy

        elif evaluation_mode == "mop":
            if ((self._count - self._previous_count) % self.sampling_freq == 0):
                self._previous_count = self._count

                ### Score policies
                with torch.no_grad():
                    (
                        candidate_actions,
                        candidate_action_infos,
                    ) = self._get_all_candidate_actions(
                        obss=np.expand_dims(obs, axis=0), task_id=task_id
                    )
                    ### Sample by minimum task probs
                    ### IMPORTANT: score function needs full observation
                    scoring_actions = [x["mean"] for x in candidate_action_infos]
                    score1s, score2s = self._get_scores(
                        observations=np.expand_dims(observation, axis=0),
                        obss=np.expand_dims(obs, axis=0),
                        candidate_actions=scoring_actions,
                        task_id=task_id,
                    )
                    score1s, score2s = score1s[0], score2s[0]

                scores = score1s
                self._curr_policy = np.argmax(scores)
                curr_action = candidate_actions[self._curr_policy][0]
                curr_action_info = {}
                for k in candidate_action_infos[0].keys():
                    curr_action_info[k] = candidate_action_infos[self._curr_policy][
                        k
                    ][0]
            else:
                curr_action, curr_action_info = self._get_policy_action(
                    obs, observation, self._curr_policy
                )
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
        assert self.sampling_freq == 1
        batch_size = observations.shape[0]
        obss, tasks = self.split_observation(observations)
        task_ids = self.task_identifier.get_actions(tasks)[0]
        assert len(set(task_ids)) == 1  # assume all tasks are the same
        task_id = task_ids[0]

        if evaluation_mode == "p":
            curr_policy = task_id
            curr_policies = task_ids
            curr_actions, curr_action_infos = self._get_policy_actions(
                obss, observations, curr_policy
            )

        elif evaluation_mode == "mop":
            ### Score policies
            with torch.no_grad():
                ### TO DO: Make get all candidate actions compatible with batch observations
                (
                    candidate_actions,
                    candidate_action_infos,
                ) = self._get_all_candidate_actions(
                    obss=obss, task_id=task_id
                )  # (num_policies, batch, action_dim)
                ### Sample by minimum task probs
                ### ASDF IMPORTANT: score function needs full observation, why???
                ### TO DO: Make get scores compatible with batch observations
                score1s, score2s = self._get_scores(
                    observations=observations,
                    obss=obss,
                    candidate_actions=candidate_actions,
                    task_id=task_id,
                )  # (batch, num_policies)

            scores = np.minimum(score1s, score2s)
            curr_policies = np.argmax(scores, axis=1)

            curr_actions = np.array(candidate_actions)[
                curr_policies, list(range(batch_size))
            ]
            curr_action_infos = {}
            for key in candidate_action_infos[0].keys():
                curr_action_infos[key] = np.array(
                    [info[key] for info in candidate_action_infos]
                )[curr_policies, list(range(batch_size))]
        self._count += 1
        policy_ids = task_ids

        return (
            curr_actions,
            {
                **curr_action_infos,
                "policy_id": policy_ids,
                "real_policy_id": curr_policies,
                "task_id": task_ids,
            },
        )

    def _get_policy_action(self, obs, observation, policy_id):

        with torch.no_grad():
            if self.policy_architecture == "separate":
                action, info = self.policies[policy_id].get_action(obs)

            elif self.policy_architecture == "shared":
                raise NotImplementedError

            elif self.policy_architecture == "multihead":
                ### Need to be able to get by policy id, not just task
                action, info = self.policies[0].get_action(observation, policy_id)

        return action, info

    def _get_policy_actions(self, obss, observations, policy_id):

        with torch.no_grad():
            if self.policy_architecture == "separate":
                actions, infos = self.policies[policy_id].get_actions(obss)
                # actions, infos = [], {}
                # for obs, policy_id in zip(obss, policy_ids):
                #     action, info = self.policies[policy_id].get_actions(obs)
                #     actions.append(action)
                #     for k, v in info.items():
                #         if k not in infos:
                #             infos[k] = []
                #         infos[k].append(v)
                # actions = np.stack(actions)

            elif self.policy_architecture == "shared":
                raise NotImplementedError

            elif self.policy_architecture == "multihead":
                ### Need to be able to get by policy id, not just task
                raise NotImplementedError
                action, info = self.policies[0].get_actions(observations, policy_id)

        return actions, infos

    def _get_all_candidate_actions(self, obss, task_id):
        ## Takes in batched obss, returns (batch, num_policies, action_dim)
        candidate_actions, candidate_action_infos = [], []

        with torch.no_grad():
            if self.policy_architecture == "separate":
                for i in range(len(self.policies)):
                    candidate_action, candidate_action_info = self.policies[
                        i
                    ].get_actions(obss)
                    candidate_actions.append(candidate_action)
                    candidate_action_infos.append(candidate_action_info)
            elif self.policy_architecture == "shared":
                ## somehow replace observation task_id with the policies
                pass

            elif self.policy_architecture == "multihead":
                ### Not implemented for batched data
                assert obss.shape[0] == 1
                # obss = obss[0]
                (candidate_actions, candidate_action_infos,) = self.policies[
                    0
                ].get_all_actions(obss)

        return candidate_actions, candidate_action_infos

    def _get_scores(self, observations, obss, candidate_actions, task_id):

        if self.score_architecture == "multihead":
            assert observations.shape[0] == 1 and obss.shape[0] == 1

        input_observations = as_torch(
            np.repeat(observations, len(candidate_actions), axis=0)
        )
        input_obs = as_torch(np.repeat(obss, len(candidate_actions), axis=0))
        input_acs = as_torch(np.vstack(candidate_actions))
        # (batch, num_policies)
        batch_size = observations.shape[0]
        if self.score_architecture == "separate":
            num_policies = len(self.score_functions)
            score1s = (
                self.score_functions[task_id](
                    input_obs,
                    input_acs,
                )
                .flatten()
                .cpu()
                .numpy()
                .reshape((batch_size, num_policies))
            )

            score2s = (
                self.score_function2s[task_id](
                    input_obs,
                    input_acs,
                )
                .flatten()
                .cpu()
                .numpy()
                .reshape((batch_size, num_policies))
            )
        elif self.policy_architecture == "shared":
            ## somehow replace observation task_id with the policies
            pass
        elif self.score_architecture == "multihead":
            ### Not implemented for batched data
            num_policies = self.score_functions[0]._n_heads
            score1s = (
                self.score_functions[0](
                    input_observations,
                    input_acs,
                )
                .flatten()
                .cpu()
                .numpy()
                .reshape((batch_size, num_policies))
            )

            score2s = (
                self.score_function2s[0](
                    input_observations,
                    input_acs,
                )
                .flatten()
                .cpu()
                .numpy()
                .reshape((batch_size, num_policies))
            )

        return score1s, score2s

    def reset(self, do_resets=None):
        self.task_identifier.reset(do_resets=do_resets)

        for policy in self.policies:
            policy.reset(do_resets=do_resets)

        self._curr_policy = None
        self._curr_policies = None
        self._count = 0
        self._previous_count = 0

    def get_param_values(self):
        task_identifier_params = self.task_identifier.state_dict()

        policies_params = [policy.state_dict() for policy in self.policies]
        score_functions_params = [fn.state_dict() for fn in self.score_functions]
        score_function2s_params = [fn.state_dict() for fn in self.score_function2s]

        """
        if self.policy_architecture == "separate":
            policies_params = [policy.state_dict() for policy in self.policies]
        else:
            policies_params = self.policies.state_dict()
        if self.score_architecture == "separate":
            score_functions_params = [fn.state_dict() for fn in self.score_functions]
            score_function2s_params = [fn.state_dict() for fn in self.score_function2s]
        else:
            score_functions_params = self.score_function.state_dict()
            score_function2s_params = self.score_function2s.state_dict()
        """


        return {
            "task_identifier": task_identifier_params,
            "policies": policies_params,
            "score_functions": score_functions_params,
            "score_function2s": score_function2s_params,
        }

    def set_param_values(self, state_dict):

        self.task_identifier.load_state_dict(state_dict["task_identifier"])

        for i in range(len(self.policies)):
            self.policies[i].load_state_dict(state_dict["policies"][i])
            self.score_functions[i].load_state_dict(state_dict["score_functions"][i])
            self.score_function2s[i].load_state_dict(state_dict["score_function2s"][i])

