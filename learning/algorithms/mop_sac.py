"""
This modules creates a Mixture of Policies (MoP) SAC model that has options for saving and training and
mixture data as well as evaluation modes for mixture policies.
 """  # yapf: disable
from collections import deque
import copy
import time
from functools import partial

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from akro import Discrete
from garage import StepType, EpisodeBatch
from garage.np.algos import RLAlgorithm
from garage.sampler import RaySampler
from garage.torch import as_torch_dict, global_device

from learning.utils import (
    log_performance,
    get_path_task_id,
    log_multitask_performance,
    log_wandb,
    obtain_multitask_multimode_evaluation_episodes,
)
from learning.algorithms import SAC

# yapf: enable


class MoPSAC(SAC):
    def __init__(self, *, mop_policy, **kwargs):

        super().__init__(**kwargs)

        self.mop_policy = mop_policy
        self.mixture_probs = np.zeros((self._num_tasks, self._num_tasks))

        self.episode_rewards = np.zeros(self._num_tasks)
        self.success_rates = np.zeros(self._num_tasks)
        self.stages_completed = np.zeros(self._num_tasks)

    def train(self, trainer):

        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for epoch in trainer.step_epochs():

            if epoch == 0:
                a = time.time()
                last_return = self._evaluate_policy(trainer.step_itr)
                b = time.time()
                print("Evaluation time: {}".format(b - a))

            for _ in range(self._steps_per_epoch):
                if not (
                    self.replay_buffer.n_transitions_stored >= self._min_buffer_size
                ):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                if isinstance(self._sampler, RaySampler):
                    # ray only supports CPU sampling
                    with torch.no_grad():
                        agent_update = copy.deepcopy(self.mop_policy).cpu()
                else:
                    agent_update = None

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
                    [0] * self._num_tasks,
                    [0] * self._num_tasks,
                    [0] * self._num_tasks,
                    [0] * self._num_tasks,
                    [0] * self._num_tasks,
                    [0] * self._num_tasks,
                )
                policy_counts = np.zeros((self._num_tasks, self._num_tasks))
                for path in trainer.step_episode:
                    task_id = get_path_task_id(path)
                    path_actions = path["actions"]
                    if self._discrete:
                        path_actions = path_actions.reshape(-1, 1)
                    if self._multihead:
                        preprocessed_obs = path["observations"]
                        preprocessed_next_obs = path["next_observations"]
                    else:
                        preprocessed_obs = self.preproc_obs(path["observations"])[0]
                        preprocessed_next_obs = self.preproc_obs(
                            path["next_observations"]
                        )[0]
                    self.replay_buffer.add_path(
                        dict(
                            observation=preprocessed_obs,
                            action=path_actions,
                            reward=path["rewards"].reshape(-1, 1),
                            next_observation=preprocessed_next_obs,
                            terminal=np.array(
                                [
                                    step_type == StepType.TERMINAL
                                    for step_type in path["step_types"]
                                ]
                            ).reshape(-1, 1),
                        )
                    )
                    path_returns[task_id] += sum(path["rewards"])
                    num_samples[task_id] += len(path["actions"])
                    num_path_starts[task_id] += np.sum(
                        [
                            step_type == StepType.FIRST
                            for step_type in path["step_types"]
                        ]
                    )
                    num_path_ends[task_id] += np.sum(
                        [
                            step_type == StepType.TERMINAL
                            for step_type in path["step_types"]
                        ]
                    )
                    if "success" in path["env_infos"]:
                        num_successes[task_id] += path["env_infos"]["success"].any()
                    if "stages_completed" in path["env_infos"]:
                        num_stages_completed[task_id] += path["env_infos"][
                            "stages_completed"
                        ][-1]

                    for i in range(self._num_tasks):
                        policy_counts[task_id][i] += (
                            path["agent_infos"]["real_policy_id"] == i
                        ).sum()

                for i in range(self._num_tasks):
                    num_paths = max(num_path_starts[i], num_path_ends[i], 1)  # AD-HOC
                    self.episode_rewards[i] = path_returns[i] / num_paths
                    self.success_rates[i] = (
                        num_path_ends[i] and num_successes[i] / num_path_ends[i]
                    )
                    self.stages_completed[i] = (
                        num_path_ends[i] and num_stages_completed[i] / num_path_ends[i]
                    )
                    self.mixture_probs[i] = policy_counts[i] / np.sum(policy_counts[i])
                for _ in range(self._gradient_steps):
                    loss_dict = self.train_once()
            if epoch % self._evaluation_frequency == 0 and epoch > 0:
                start = time.time()
                last_return = self._evaluate_policy(trainer.step_itr)
                end = time.time()
                print("Evaluation Time: ", end - start)

            self._log_statistics(
                trainer.step_itr,
                trainer.total_env_steps,
                train_infos=loss_dict,
                lr=self._policy_optimizer.param_groups[0]["lr"],
                videos=None,
            )
            trainer.step_itr += 1
            # self._policy_scheduler.step()
            # self._qf1_scheduler.step()
            # self._qf2_scheduler.step()
            # if self._use_automatic_entropy_tuning:
            #     self._alpha_scheduler.step()

        return np.mean(last_return)

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
                self.mop_policy,
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
                self.mop_policy,
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
                last_return = log_multitask_performance(
                    epoch,
                    eval_episodes[evaluation_mode],
                    discount=self._discount,
                    prefix=prefix,
                    videos=eval_videos[evaluation_mode],
                )
        return last_return

    def _log_statistics(self, step, total_steps, train_infos, lr, videos):
        """Record training statistics to dowel such as losses and returns.

        Args:
            loss_dict (dict of torch.Tensor): losses from networks.
            lr (float): current learning rate.
            videos (wandb.Video): videos of rollouts.

        """

        infos = {}

        with torch.no_grad():
            infos["AlphaTemperature"] = self._log_alpha.exp().mean().item()
        for k, v in train_infos.items():
            infos[k] = float(v)
        infos["ReplayBufferSize"] = self.replay_buffer.n_transitions_stored
        infos["AverageReturn"] = np.mean(self.episode_rewards)
        infos["TotalEnvSteps"] = total_steps
        infos["LearningRate"] = float(lr)

        for task_id in range(self._num_tasks):
            infos[f"LocalPolicy{task_id}/AverageReturn"] = np.mean(
                self.episode_rewards[task_id]
            )
            infos[f"LocalPolicy{task_id}/SuccessRate"] = np.mean(
                self.success_rates[task_id]
            )
            infos[f"LocalPolicy{task_id}/StagesCompleted"] = np.mean(
                self.stages_completed[task_id]
            )
            infos[f"LocalPolicy{task_id}/MixtureProbs"] = self.mixture_probs[task_id][
                task_id
            ]

            for i in range(self._num_tasks):
                infos[f"LocalPolicy{task_id}/Policy{i}Prob"] = self.mixture_probs[
                    task_id
                ][i]

        log_wandb(step, infos, medias=videos, prefix="Train/")

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy.policies[0],
            self._qf1,
            self._qf2,
            self._target_qf1,
            self._target_qf2,
        ]
