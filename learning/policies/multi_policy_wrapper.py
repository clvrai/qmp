import abc
import numpy as np

from garage.torch.policies.policy import Policy


class MultiPolicyWrapper(Policy, abc.ABC):
    """Policy wrapper for methods with ensembles of policies with an every-timestep policy_assigner.

    Args:
        policies (list[Policy]): list of Policy objects that make up the policy ensemble
        policy_assigner (Policy): Policy object that assigns a policy from the ensemble given the current observation

    """

    def __init__(self, policies, policy_assigner, split_observation=None):
        super().__init__(env_spec=None, name="MultiPolicyWrapper")

        self.policies = policies
        self.policy_assigner = policy_assigner
        self.split_observation = split_observation or (lambda x: (x, x))

    def get_action(self, observation, policy_id=None, task_id=None):
        obs, task = self.split_observation(observation)

        if policy_id is None:
            ### need to think about how this should work
            curr_policy = self.policy_assigner.get_action(task)[0]
        else:
            curr_policy = policy_id

        curr_action, curr_action_info = self.policies[curr_policy].get_action(obs)

        return (
            curr_action,
            {**curr_action_info, "policy_id": curr_policy},
        )

    def get_all_actions(self, observation, task_id=None):
        obs, task = self.split_observation(observation)

        all_actions, all_infos = [], []

        for policy in self.policies:
            ac, ac_info = policy.get_action(obs)
            all_actions.append(ac)
            all_infos.append(ac_info)

        return ac, ac_info

    def get_actions(self, observations):
        obss, tasks = self.split_observation(observations)

        curr_policies = self.policy_assigner.get_actions(tasks)[0]

        ### ASDF : vectorize
        actions, infos = [], []

        for (observation, curr_policy) in zip(obss, curr_policies):
            a, i = self.policies[curr_policy].get_action(observation)
            actions.append(a)
            infos.append(i)
        curr_actions = np.vstack(actions)
        keys = infos[0].keys()
        infos = {k: np.vstack([i[k] for i in infos]) for k in keys}

        return curr_actions, {"policy_id": curr_policies, **infos}

    def reset(self, do_resets=None):
        self.policy_assigner.reset(do_resets=do_resets)

        for policy in self.policies:
            policy.reset(do_resets=do_resets)

    def get_param_values(self):
        policy_assigner_params = self.policy_assigner.state_dict()
        policies_params = [policy.state_dict() for policy in self.policies]

        return {"policy_assigner": policy_assigner_params, "policies": policies_params}

    def set_param_values(self, state_dict):

        self.policy_assigner.load_state_dict(state_dict["policy_assigner"])

        for i in range(len(self.policies)):
            self.policies[i].load_state_dict(state_dict["policies"][i])
