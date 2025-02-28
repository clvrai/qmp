"""
Online version of CDS based on DnCSAC without policy regularization.

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

from learning.algorithms import DnCSAC
# yapf: enable


class OnlineCDS(DnCSAC):
    def __init__(
        self, *, reward_fns=None, min_reward=None, sharing_quantile=0, unsupervised=False,  **kwargs
    ):
        super().__init__(**kwargs)

        self._reward_fns = reward_fns
        self._min_reward = min_reward
        self._sharing_quantile = sharing_quantile
        self._unsupervised = unsupervised

        if self._unsupervised:
            assert self._min_reward is not None
        else:
            assert self._reward_fns is not None

    def train_once(self, trainer):
        """
        Don't preprocess observations before storing in replay buffer. We store some pre-computed state info in obs for reward calculations later.
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

        step_types = []

        for path in trainer.step_episode:
            policy_id = get_path_policy_id(path)
            step_types.extend(path["step_types"])
            terminals = np.array(
                [step_type == StepType.TERMINAL for step_type in path["step_types"]]
            ).reshape(-1, 1)
            self.replay_buffers[policy_id].add_path(
                dict(
                    observation=path["observations"],
                    action=path["actions"],
                    reward=path["rewards"].reshape(-1, 1),
                    next_observation=path["next_observations"],
                    terminal=terminals,
                )
            )
            path_returns[policy_id] += sum(path["rewards"])
            num_samples[policy_id] += len(path["actions"])
            num_path_starts[policy_id] += np.sum(
                [step_type == StepType.FIRST for step_type in path["step_types"]]
            )
            num_path_ends[policy_id] += np.sum(terminals)
            if "success" in path["env_infos"]:
                num_successes[policy_id] += path["env_infos"]["success"].any()
            if "stages_completed" in path["env_infos"]:
                num_stages_completed[policy_id] += path["env_infos"][
                    "stages_completed"
                ][-1]

        if np.any([n == 0 for n in num_samples]):
            import ipdb

            ipdb.set_trace()
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

    def train_policy_once(self, policy_id, itr=None, paths=None):
        """
        Complete 1 training iteration of SAC.

        Changed the sampling procedure to sample from other tasks as well,
        add to batch if Q-value is above self._sharing_quantile of Q-values,
        and re-label with task rewards.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """

        del itr
        del paths
        if self.replay_buffers[policy_id].n_transitions_stored >= (
            self._min_buffer_size
        ):
            x = time.time()
            policy_samples = self.replay_buffers.sample_transitions(
                self._buffer_batch_size, policy_id
            )
            all_other_samples = self.replay_buffers.sample_all_other_transitions(self._buffer_batch_size, policy_id)
            other_len = len(all_other_samples["reward"])

            # policy_samples["observation"] = self.preproc_obs(policy_samples["observation"])[0]
            # policy_samples["next_observation"] = self.preproc_obs(policy_samples["next_observation"])[0]
            # all_other_samples["observation"] = self.preproc_obs(all_other_samples["observation"])[0]
            # all_other_samples["next_observation"] = self.preproc_obs(all_other_samples["next_observation"])[0]

            if self._unsupervised:
                relabeled_rews = np.ones((other_len, 1)) * self._min_reward
            else:
                relabeled_rews = self._reward_fns[policy_id](all_other_samples["observation"], all_other_samples["action"])
            if self._sharing_quantile == 0:
                sharing_idxs = list(range(other_len))

            else:
                with torch.no_grad():
                    ### HACK HACK HACK dimensions not right here
                    policy_qs = self._qf1s[policy_id](as_torch(policy_samples["observation"]), as_torch(policy_samples["action"])).cpu().detach().numpy()
                    policy_quantile = ([-np.inf] + list(np.sort(policy_qs.flatten())))[int(len(policy_qs) * self._sharing_quantile)]
                    all_qs = self._qf1s[policy_id](as_torch(all_other_samples["observation"]), as_torch(all_other_samples["action"])).cpu().detach().numpy()
                    sharing_idxs = np.where(all_qs > policy_quantile)[0]

            for k in policy_samples.keys():
                if k == "reward":
                    shared_data = relabeled_rews[sharing_idxs]
                else:
                    shared_data = all_other_samples[k][sharing_idxs]
                policy_samples[k] = np.concatenate([policy_samples[k], shared_data])

            self._sharing_proportion[policy_id].append(len(sharing_idxs) / (len(sharing_idxs) + other_len))


            ### append sharing_idxs to policy_samples with new all_rews
            # dict_keys(['observation', 'action', 'reward', 'next_observation', 'terminal']) (32,20)
            policy_samples = as_torch_dict(policy_samples)
            all_obs = policy_samples["observation"]

            y = time.time()
            loss_dict, a, b, c, d, e = self.optimize_policy(
                all_obs, policy_samples, policy_id=policy_id
            )
            z = time.time()

        else:
            loss_dict = {}
            a, b, c, d, e, x, y, z = 0, 0, 0, 0, 0, 0, 0, 0

        self._update_targets(policy_id)
        return loss_dict, a, b, c, d, e, y - x, z - y

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

        log_wandb(step, infos, prefix="Train/" + self.policies[policy_id].name + "/")

