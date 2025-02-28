"""
This modules creates a Mixture of Policies (MoP) model that also does online unsupervised data sharing using OnlineCDS
 """

# yapf: disable
from collections import deque
import copy
import time

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp



from garage import StepType, EpisodeBatch
from garage.np.algos import RLAlgorithm
from garage.sampler import RaySampler, MultiprocessingSampler
from garage.torch import as_torch_dict, as_torch, global_device

from learning.utils import (get_path_policy_id, get_policy_ids, get_path_task_id,
                            log_multitask_performance, log_performance, log_wandb, obtain_multitask_multimode_evaluation_episodes)

from learning.algorithms import OnlineCDS, DnCSAC, MoPDnC
# yapf: enable


class QMPUDS(OnlineCDS, MoPDnC):
    def __init__(
        self, **kwargs
    ):
        super().__init__(**kwargs)

        assert self._unsupervised

    def train_once(self, trainer):
        """
        Did not make CDS change: Don't preprocess observations before storing in replay buffer. We store some pre-computed state info in obs for reward calculations later.
        Should still work for UDS
        """
        if not (
            np.all(
                [
                    self.replay_buffers[i].n_transitions_stored >= self._min_buffer_size
                    for i in range(self.n_policies)
                ]
            )
        ):
            batch_size = int(self._min_buffer_size) * self.n_policies
        else:
            batch_size = None

        if isinstance(self._sampler, RaySampler) or isinstance(
            self._sampler, MultiprocessingSampler
        ):
            # ray only supports CPU sampling
            with torch.no_grad():
                agent_update = copy.copy(self.policy)
                agent_update.policies = [
                    copy.deepcopy(policy).cpu() for policy in self.policies
                ]
                agent_update.score_functions = [
                    copy.deepcopy(qf).cpu() for qf in self._qf1s
                ]
                agent_update.score_function2s = [
                    copy.deepcopy(qf).cpu() for qf in self._qf2s
                ]
        else:
            agent_update = None

        start = time.time()
        trainer.step_episode = trainer.obtain_samples(
            trainer.step_itr, batch_size, agent_update=agent_update
        )
        (
            path_returns,
            num_samples,
            num_path_starts,
            num_path_ends,
            num_successes,
            num_stages_completed,
        ) = (
            [0] * self.n_policies,
            [0] * self.n_policies,
            [0] * self.n_policies,
            [0] * self.n_policies,
            [0] * self.n_policies,
            [0] * self.n_policies,
        )
        policy_counts = np.zeros((self.n_policies, self.n_policies))

        step_types = []
        for path in trainer.step_episode:
            task_id = get_path_task_id(path)
            step_types.extend(path["step_types"])
            terminals = np.array(
                [step_type == StepType.TERMINAL for step_type in path["step_types"]]
            ).reshape(-1, 1)

            policy_id = get_path_policy_id(path)
            self.replay_buffers[policy_id].add_path(
                dict(
                    observation=self.preproc_obs(path["observations"])[0],
                    action=path["actions"],
                    reward=path["rewards"].reshape(-1, 1),
                    next_observation=self.preproc_obs(path["next_observations"])[0],
                    terminal=terminals,
                )
            )

            path_returns[task_id] += sum(path["rewards"])
            num_samples[task_id] += len(path["actions"])
            num_path_starts[task_id] += np.sum(
                [step_type == StepType.FIRST for step_type in path["step_types"]]
            )
            num_path_ends[task_id] += np.sum(terminals)
            if "success" in path["env_infos"]:
                num_successes[task_id] += path["env_infos"]["success"].any()
            if "stages_completed" in path["env_infos"]:
                num_stages_completed[task_id] += path["env_infos"]["stages_completed"][
                    -1
                ]

            for i in range(self.n_policies):
                policy_counts[task_id][i] += (
                    path["agent_infos"]["real_policy_id"] == i
                ).sum()

        for i in range(self.n_policies):
            num_paths = max(num_path_starts[i], num_path_ends[i], 1)  # AD-HOC
            self.episode_rewards[i] = path_returns[i] / num_paths
            self.success_rates[i] = (
                num_path_ends[i] and num_successes[i] / num_path_ends[i]
            )
            self.stages_completed[i] = (
                num_path_ends[i] and num_stages_completed[i] / num_path_ends[i]
            )
            self.num_samples[i] += num_samples[i]
            self.mixture_probs[i] = policy_counts[i] / np.sum(policy_counts[i])

        data = time.time()

        ### ASDF Which way should the for loop go?  Policy id then gradient steps?
        self._sharing_proportion = [[]] * self.n_policies
        sample_transitions, optimize_policies = [], []
        critic_training, actor_obj, kl_penalty, actor_training, alpha_training = (
            [],
            [],
            [],
            [],
            [],
        )

        grad_step_count = []
        for policy_id in range(self.n_policies):
            num_grad_steps = max(
                int(
                    self._gradient_steps / np.sum(num_samples) * num_samples[policy_id]
                ),
                1,
            )
            for _ in range(num_grad_steps):
                loss_dict, a, b, c, d, e, f, g = self.train_policy_once(policy_id)
                critic_training.append(a)
                actor_obj.append(b)
                kl_penalty.append(c)
                actor_training.append(d)
                alpha_training.append(e)
                sample_transitions.append(f)
                optimize_policies.append(g)

            grad_step_count.append(num_grad_steps)
            self._log_statistics(trainer.step_itr, policy_id, loss_dict)

        training = time.time()
        # print(
        #     f"Gather {num_samples} Data Collection Time: {data - start}, Training {grad_step_count} Steps Time: {training - data}"
        # )
        # print(
        #     f"Optimize Policies: {np.sum(optimize_policies)}, Sample Transitions: {np.sum(sample_transitions)}"
        # )
        # print(
        #     f"Critic Training: {np.sum(critic_training)}, Actor Obj: {np.sum(actor_obj)}, KL Penalty: {np.sum(kl_penalty)}, Actor Training: {np.sum(actor_training)}, Alpha Training: {np.sum(alpha_training)}"
        # )

    def _log_statistics(self, step, policy_id, train_infos):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """

        infos = {}

        with torch.no_grad():
            infos["AlphaTemperature"] = self._log_alphas[policy_id].exp().mean().item()
            log_betas = self._log_kl_coeffs[policy_id].cpu().detach().numpy()
            log_betas = np.concatenate(
                [log_betas[:policy_id], log_betas[policy_id + 1 :]]
            )
            betamean = np.mean(np.exp(log_betas))
            infos["BetaKL"] = betamean
            infos["TargetEntropy"] = self._target_entropy

        for k, v in train_infos.items():
            infos[k] = float(v)
        infos["ReplayBufferSize"] = self.replay_buffers[policy_id].n_transitions_stored
        infos["AverageReturn"] = np.mean(self.episode_rewards[policy_id])
        infos["SuccessRate"] = np.mean(self.success_rates[policy_id])
        infos["StagesCompleted"] = np.mean(self.stages_completed[policy_id])
        infos["EnvSteps"] = self.num_samples[policy_id]
        infos["SharedData"] = np.mean(self._sharing_proportion[policy_id])
        infos["MixtureProbs"] = self.mixture_probs[policy_id][policy_id]

        for i in range(self.n_policies):
            infos[f"Policy{i}Prob"] = self.mixture_probs[policy_id][i]

        log_wandb(step, infos, prefix="Train/" + self.policies[policy_id].name + "/")
