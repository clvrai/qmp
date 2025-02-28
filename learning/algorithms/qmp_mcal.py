"""
This modules creates a Mixture of Policies (MoP) DnC model that has options for saving and training and
mixture data as well as evaluation modes for mixture policies.
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
from learning.algorithms import MultiCriticAL
# yapf: enable


class QMPMultiCriticAL(MultiCriticAL):
    def __init__(
        self, *, min_task_probs, **kwargs
    ):
        super().__init__(**kwargs)

        self._min_task_probs = min_task_probs
        self.mixture_probs = np.zeros((self.n_tasks, self.n_tasks))

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        self.policy.set_min_task_probs(self._min_task_probs)
        for epoch in trainer.step_epochs():

            for _ in range(self._steps_per_epoch):
                self.train_once(trainer)

            if epoch % self._evaluation_frequency == 0:
                start = time.time()
                last_return = self._evaluate_policy(trainer.step_itr)
                end = time.time()
                print("Evaluation Time: ", end - start)

            infos = {}
            infos["AverageReturn"] = np.mean(
                [np.mean(self.episode_rewards[i]) for i in range(self.n_tasks)]
            )
            infos["SuccessRate"] = np.mean(
                [np.mean(self.success_rates[i]) for i in range(self.n_tasks)]
            )
            infos["StagesCompleted"] = np.mean(
                [np.mean(self.stages_completed[i]) for i in range(self.n_tasks)]
            )
            infos["TotalEnvSteps"] = trainer.total_env_steps
            log_wandb(trainer.step_itr, infos, prefix="Train/")
            trainer.step_itr += 1


        return np.mean(last_return)

    def train_once(self, trainer):
        if not (
            np.all(
                [
                    self.replay_buffers[i].n_transitions_stored >= self._min_buffer_size
                    for i in range(self.n_tasks)
                ]
            )
        ):
            batch_size = int(self._min_buffer_size) * self.n_tasks
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
            [0] * self.n_tasks,
            [0] * self.n_tasks,
            [0] * self.n_tasks,
            [0] * self.n_tasks,
            [0] * self.n_tasks,
            [0] * self.n_tasks,
        )
        policy_counts = np.zeros((self.n_tasks, self.n_tasks))

        step_types = []
        for path in trainer.step_episode:
            ### TODO: why is this not get_task_id function from env?
            task_id = get_path_task_id(path)
            step_types.extend(path["step_types"])
            terminals = np.array(
                [step_type == StepType.TERMINAL for step_type in path["step_types"]]
            ).reshape(-1, 1)

            policy_id = get_path_policy_id(path)
            self.replay_buffers[policy_id].add_path(
                dict(
                    observation=path["observations"],
                    action=path["actions"],
                    reward=path["rewards"].reshape(-1, 1),
                    next_observation=path["next_observations"],
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

            for i in range(self.n_tasks):
                policy_counts[task_id][i] += (
                    path["agent_infos"]["real_policy_id"] == i
                ).sum()

        for i in range(self.n_tasks):
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
        sample_transitions, optimize_policies = [], []
        critic_training, actor_obj, kl_penalty, actor_training, alpha_training = (
            [],
            [],
            [],
            [],
            [],
        )

        grad_step_count = []
        for policy_id in range(self.n_tasks):
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

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """

        if not self._visualizer.do_visualization(epoch):
            visualizer = None
            num_vis = 0

        else:
            self._visualizer.reset()
            visualizer = self._visualizer
            num_vis = (
                max(self._visualizer.num_videos // len(self._eval_env), 1)
                if isinstance(self._eval_env, list)
                else self._visualizer.num_videos
            )
        evaluation_modes = [None, "p"]
        if isinstance(self._eval_env, list):
            eval_episodes, eval_videos = obtain_multitask_multimode_evaluation_episodes(
                self.policy,
                self._eval_env,
                self._max_episode_length_eval,
                num_eps_per_mode=self._num_evaluation_episodes // len(self._eval_env),
                deterministic=self._use_deterministic_evaluation,
                evaluation_modes=evaluation_modes,
                num_vis=num_vis,
                visualizer=visualizer,
            )

            for evaluation_mode in evaluation_modes:
                prefix = (
                    f"Evaluation/{evaluation_mode}/"
                    if evaluation_mode is not None
                    else "Evaluation/"
                )
                last_return = log_multitask_performance(
                    epoch,
                    eval_episodes[evaluation_mode],
                    discount=self._discount,
                    prefix=prefix,
                    videos=eval_videos[evaluation_mode],
                )
        else:
            eval_episodes, eval_videos = obtain_multitask_multimode_evaluation_episodes(
                self.policy,
                [self._eval_env],
                self._max_episode_length_eval,
                num_eps_per_mode=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation,
                evaluation_modes=evaluation_modes,
                num_vis=num_vis,
                visualizer=visualizer,
            )
            for evaluation_mode in evaluation_modes:
                prefix = (
                    f"Evaluation/{evaluation_mode}/"
                    if evaluation_mode is not None
                    else "Evaluation/"
                )
                last_return = log_performance(
                    epoch,
                    eval_episodes[evaluation_mode],
                    discount=self._discount,
                    prefix=prefix,
                    videos=eval_videos[evaluation_mode][0],
                )
        return last_return

    def _log_statistics(self, step, policy_id, train_infos):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """

        infos = {}

        with torch.no_grad():
            infos["AlphaTemperature"] = self._log_alphas[0].exp().mean().item()
            infos["TargetEntropy"] = self._target_entropy

        for k, v in train_infos.items():
            infos[k] = float(v)
        infos["ReplayBufferSize"] = self.replay_buffers[policy_id].n_transitions_stored
        infos["AverageReturn"] = np.mean(self.episode_rewards[policy_id])
        infos["SuccessRate"] = np.mean(self.success_rates[policy_id])
        infos["StagesCompleted"] = np.mean(self.stages_completed[policy_id])
        infos["EnvSteps"] = self.num_samples[policy_id]
        infos["MixtureProbs"] = self.mixture_probs[policy_id][policy_id]

        for i in range(self.n_tasks):
            infos[f"Policy{i}Prob"] = self.mixture_probs[policy_id][i]

        log_wandb(step, infos, prefix="Train/Task" + str(policy_id) + "/")
