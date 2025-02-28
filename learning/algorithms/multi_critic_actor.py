"""This modules creates a DnC SAC model based on garage SAC."""

# yapf: disable
from collections import deque
import copy
import time
from functools import partial

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from garage import StepType, EpisodeBatch
from garage.np.algos import RLAlgorithm
from garage.sampler import RaySampler, MultiprocessingSampler
from garage.torch import as_torch_dict, as_torch, global_device

from learning.utils import (get_path_policy_id, log_performance,
                            log_multitask_performance, log_wandb, obtain_multitask_multimode_evaluation_episodes)
# yapf: enable


class MultiCriticAL(RLAlgorithm):
    def __init__(
        self,
        env_spec,
        policy,
        qf1s,
        qf2s,
        replay_buffers,
        sampler,
        visualizer,
        get_stage_id,
        preproc_obs,
        sampling_type,
        n_tasks,
        get_task_id,
        *,  # Everything after this is numbers.
        entropy_beta=False,
        max_episode_length_eval=None,
        gradient_steps_per_itr,
        fixed_alpha=None,
        target_entropy=None,
        initial_log_entropy=0.0,
        discount=0.99,
        buffer_batch_size=64,
        min_buffer_size=int(1e4),
        target_update_tau=5e-3,
        lr=3e-4,
        reward_scale=1.0,
        optimizer=torch.optim.Adam,
        # scheduler=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.995),
        steps_per_epoch=1,
        num_evaluation_episodes=10,
        evaluation_frequency=1,
        eval_env=None,
        use_deterministic_evaluation=True,
        **kwargs
    ):
        self.get_stage_id = get_stage_id
        self.preproc_obs = preproc_obs or (lambda x: (x, x))
        self._sampling_type = sampling_type

        self._qf1s = qf1s
        self._qf2s = qf2s
        self.replay_buffers = replay_buffers
        self._tau = target_update_tau
        self._lr = lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._evaluation_frequency = evaluation_frequency
        self._eval_env = eval_env

        self._min_buffer_size = min_buffer_size
        self._steps_per_epoch = steps_per_epoch
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = env_spec.max_episode_length

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation

        self.policies = [policy]
        self.policy = policy
        self.env_spec = env_spec
        self.n_tasks = n_tasks
        self.get_task_id = get_task_id

        self._sampler = sampler
        self._visualizer = visualizer

        self._reward_scale = reward_scale
        # use 2 target q networks
        self._target_qf1s = [copy.deepcopy(qf) for qf in self._qf1s]
        self._target_qf2s = [copy.deepcopy(qf) for qf in self._qf2s]
        # optimizers and schedulers
        self._policy_optimizers = [
            self._optimizer(policy.parameters(), lr=self._lr)
            for policy in self.policies
        ]
        self._qf1_optimizers = [
            self._optimizer(qf.parameters(), lr=self._lr) for qf in self._qf1s
        ]
        self._qf2_optimizers = [
            self._optimizer(qf.parameters(), lr=self._lr) for qf in self._qf2s
        ]

        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha

        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(self.env_spec.action_space.shape).item()
            self._log_alphas = [torch.Tensor([self._initial_log_entropy]).requires_grad_()]
            self._alpha_optimizers = [
                self._optimizer([a], lr=self._lr) for a in self._log_alphas
            ]
            # self._alpha_schedulers = [
            #     scheduler(optimizer) for optimizer in self._alpha_optimizers
            # ]
        else:
            self._log_alphas = [torch.Tensor([self._fixed_alpha]).log()]

        self._entropy_beta = entropy_beta

        self.episode_rewards = np.zeros(self.n_tasks)
        self.success_rates = np.zeros(self.n_tasks)
        self.stages_completed = np.zeros(self.n_tasks)
        self.num_samples = np.zeros(self.n_tasks)

    def train(self, trainer):
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
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
        else:
            agent_update = None

        start = time.time()

        # import ipdb; ipdb.set_trace()
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

        step_types = []

        for path in trainer.step_episode:
            task_id = self.get_task_id(path["observations"][0])
            step_types.extend(path["step_types"])
            terminals = np.array(
                [step_type == StepType.TERMINAL for step_type in path["step_types"]]
            ).reshape(-1, 1)
            self.replay_buffers[task_id].add_path(
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
                num_stages_completed[task_id] += path["env_infos"][
                    "stages_completed"
                ][-1]

        if np.any([n == 0 for n in num_samples]):
            import ipdb

            ipdb.set_trace()
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
        for task_id in range(self.n_tasks):
            num_grad_steps = max(
                int(
                    self._gradient_steps / np.sum(num_samples) * num_samples[task_id]
                ),
                1,
            )
            for _ in range(num_grad_steps):
                loss_dict, a, b, c, d, e, f, g = self.train_policy_once(task_id)
                critic_training.append(a)
                actor_obj.append(b)
                kl_penalty.append(c)
                actor_training.append(d)
                alpha_training.append(e)
                sample_transitions.append(f)
                optimize_policies.append(g)

            grad_step_count.append(num_grad_steps)
            self._log_statistics(trainer.step_itr, task_id, loss_dict)

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

    def train_policy_once(self, task_id, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

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
        if self.replay_buffers[task_id].n_transitions_stored >= (
            self._min_buffer_size
        ):
            x = time.time()
            policy_samples = self.replay_buffers.sample_transitions(
                self._buffer_batch_size, task_id
            )
            policy_samples = as_torch_dict(policy_samples)
            all_obs, _ = self.replay_buffers.sample_all_transitions(
                self._buffer_batch_size
            )
            all_obs = [as_torch(obs) for obs in all_obs]

            y = time.time()
            loss_dict, a, b, c, d, e = self.optimize_policy(
                all_obs, policy_samples, task_id=task_id
            )
            z = time.time()

        else:
            loss_dict = {}
            a, b, c, d, e, x, y, z = 0, 0, 0, 0, 0, 0, 0, 0

        self._update_targets(task_id)
        return loss_dict, a, b, c, d, e, y - x, z - y

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        This function exists in case there are versions of sac that need
        access to a modified log_alpha, such as multi_task sac.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha

        """
        del samples_data
        policy_id = 0
        log_alpha = self._log_alphas[policy_id]
        return log_alpha

    def _temperature_objective(self, log_pi, samples_data, policy_id):
        """Compute the temperature/alpha coefficient loss.

        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.

        """
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            alpha_loss = (
                -(self._get_log_alpha(samples_data))
                * (log_pi.detach() + self._target_entropy)
            ).mean()
        return alpha_loss

    def _actor_objective(
        self, samples_data, all_obs, new_actions, log_pi_new_actions
    ):
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        ## TODO: fix this to get the right task_id
        min_q_new_actions = []
        idx = 0
        for task_id in range(self.n_tasks):
            num_obs = all_obs[task_id].shape[0]
            min_q_new_actions.append(torch.min(
                self._qf1s[task_id](all_obs[task_id], new_actions[idx:idx+num_obs]),
                self._qf2s[task_id](all_obs[task_id], new_actions[idx:idx+num_obs])
            ))
            idx += num_obs
        min_q_new_actions = torch.cat(min_q_new_actions)

        policy_objective = (
            (alpha * log_pi_new_actions) - min_q_new_actions.flatten()
        ).mean()
        return policy_objective

    def _critic_objective(self, samples_data, task_id):
        obs = self.preproc_obs(samples_data["observation"])[0]
        actions = samples_data["action"]
        rewards = samples_data["reward"].flatten()
        terminals = samples_data["terminal"].flatten()
        next_obs = samples_data["next_observation"]
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        q1_pred = self._qf1s[task_id](obs, actions)
        q2_pred = self._qf2s[task_id](obs, actions)

        policy_id = 0
        new_next_actions_dist = self.policies[policy_id](next_obs)[0]
        (
            new_next_actions_pre_tanh,
            new_next_actions,
        ) = new_next_actions_dist.rsample_with_pre_tanh_value()
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh
        )

        next_obs = self.preproc_obs(next_obs)[0]
        target_q_values = (
            torch.min(
                self._target_qf1s[task_id](next_obs, new_next_actions),
                self._target_qf2s[task_id](next_obs, new_next_actions),
            ).flatten()
            - (alpha * new_log_pi)
        )

        with torch.no_grad():
            q_target = (
                rewards * self._reward_scale
                + (1.0 - terminals) * self._discount * target_q_values
            )
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss, q_target, q1_pred, q2_pred

    def _update_targets(self, policy_id):
        """Update parameters in the target q-functions."""

        target_qfs = [self._target_qf1s[policy_id], self._target_qf2s[policy_id]]
        qfs = [self._qf1s[policy_id], self._qf2s[policy_id]]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(
                    t_param.data * (1.0 - self._tau) + param.data * self._tau
                )

    def optimize_policy(self, all_obs, samples_data, task_id):
        a = time.time()
        qf1_loss, qf2_loss, q_target, q1_pred, q2_pred = self._critic_objective(
            samples_data, task_id
        )

        self._qf1_optimizers[task_id].zero_grad()
        qf1_loss.backward()
        self._qf1_optimizers[task_id].step()

        self._qf2_optimizers[task_id].zero_grad()
        qf2_loss.backward()
        self._qf2_optimizers[task_id].step()

        b = time.time()

        policy_id = 0
        action_dists = self.policies[policy_id](torch.cat(all_obs))[0]
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh
        )

        policy_loss = self._actor_objective(
            samples_data, all_obs, new_actions, log_pi_new_actions
        )

        c = time.time()

        d = time.time()

        self._policy_optimizers[policy_id].zero_grad()
        policy_loss.backward()

        self._policy_optimizers[policy_id].step()
        e = time.time()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(
                log_pi_new_actions, samples_data, task_id
            )
            self._alpha_optimizers[policy_id].zero_grad()
            alpha_loss.backward()
            self._alpha_optimizers[policy_id].step()

        entropy = action_dists.entropy().mean()
        log_pi = log_pi_new_actions.mean()

        f = time.time()

        infos = dict(
            PolicyLoss=policy_loss.item(),
            Entropy=entropy.item(),
            LogPi=log_pi.item(),
            Qf1Loss=qf1_loss.item(),
            Qf2Loss=qf2_loss.item(),
            QTarget=q_target.mean().item(),
            Qf1=q1_pred.mean().item(),
            Qf2=q2_pred.mean().item(),
            AlphaLoss=alpha_loss.item(),
        )
        return (
            infos,
            b - a,
            c - b,
            d - c,
            e - d,
            f - e,
        )

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

        if isinstance(self._eval_env, list):
            eval_episodes, eval_videos = obtain_multitask_multimode_evaluation_episodes(
                self.policy,
                self._eval_env,
                self._max_episode_length_eval,
                num_eps_per_mode=self._num_evaluation_episodes // len(self._eval_env),
                deterministic=self._use_deterministic_evaluation,
                evaluation_modes=[None],
                num_vis=num_vis,
                visualizer=visualizer,
            )
            last_return = log_multitask_performance(
                epoch,
                eval_episodes[None],
                discount=self._discount,
                videos=eval_videos[None],
            )
        else:
            eval_episodes, eval_videos = obtain_multitask_multimode_evaluation_episodes(
                self.policy,
                [self._eval_env],
                self._max_episode_length_eval,
                num_eps_per_mode=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation,
                evaluation_modes=[None],
                num_vis=num_vis,
                visualizer=visualizer,
            )
            last_return = log_performance(
                epoch,
                eval_episodes[None],
                discount=self._discount,
                videos=eval_videos[None][0],
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

        log_wandb(step, infos, prefix="Train/Task" + str(policy_id) + "/")

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        networks = [
            *self.policies,
            *self._qf1s,
            *self._qf2s,
            *self._target_qf1s,
            *self._target_qf2s,
        ]
        return networks

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        if not self._use_automatic_entropy_tuning:
            self._log_alphas = [torch.Tensor([self._fixed_alpha]).log().to(device)]
        else:
            self._log_alphas = [(torch.Tensor([self._initial_log_entropy]).to(device).requires_grad_())]
            self._alpha_optimizers = [
                self._optimizer([a], lr=self._lr) for a in self._log_alphas
            ]

