"""This modules creates a sac model in PyTorch.  If num_tasks is specified, does multitask alphas."""
# yapf: disable


import time
from collections import deque
import copy
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

from learning.utils import (log_performance,
                            log_multitask_performance, log_wandb, obtain_multitask_multimode_evaluation_episodes)
from learning.utils.general import SuppressStdout
from learning.utils.pcgrad import PCGrad, separate_batch
# yapf: enable


class SAC(RLAlgorithm):
    """A SAC Model in Torch.

    Based on Soft Actor-Critic and Applications:
        https://arxiv.org/abs/1812.05905

    Soft Actor-Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        sampler (garage.sampler.Sampler): Sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        gradient_steps_per_itr (int): Number of optimization steps that should
        gradient_steps_per_itr(int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): Discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr (float): learning rate for policy optimizers.
        qf_lr (float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        eval_env (Environment): environment used for collecting evaluation
            episodes. If None, a copy of the train env is used.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.

    """

    def __init__(
        self,
        env_spec,
        policy,
        qf1,
        qf2,
        replay_buffer,
        sampler,
        visualizer,
        preproc_obs,
        multihead,
        *,  # Everything after this is numbers.
        num_tasks=1,
        get_task_id=None,
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
        grad_clip=None,
        use_pc_grad=False,
        skip_alpha_pc_grad=True,
    ):

        self._qf1 = qf1
        self._qf2 = qf2
        self.replay_buffer = replay_buffer
        self._tau = target_update_tau
        self._lr = lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._evaluation_frequency = evaluation_frequency
        self._eval_env = eval_env
        self._discrete = isinstance(env_spec.action_space, Discrete)
        self._multihead = multihead

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

        self.policy = policy
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer

        self._sampler = sampler
        self._visualizer = visualizer
        self.use_pc_grad = use_pc_grad
        self.skip_alpha_pc_grad = skip_alpha_pc_grad

        self._reward_scale = reward_scale
        # use 2 target q networks
        if isinstance(self._qf1, list):
            self._target_qf1s = [copy.deepcopy(qf) for qf in self._qf1]
            self._target_qf2s = [copy.deepcopy(qf) for qf in self._qf2]
        else:
            self._target_qf1 = copy.deepcopy(self._qf1)
            self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = self._optimizer(
            self.policy.parameters(), lr=self._lr
        )
        if self.use_pc_grad:
            self._policy_optimizer = PCGrad(self._policy_optimizer)

        if isinstance(self._qf1, list):
            self._qf1_optimizer = [
                self._optimizer(qf.parameters(), lr=self._lr)
                for qf in self._qf1
            ]
            self._qf2_optimizer = [
                self._optimizer(qf.parameters(), lr=self._lr)
                for qf in self._qf2
            ]
            if self.use_pc_grad:
                self._qf1_optimizer = [PCGrad(opt) for opt in self._qf1_optimizer]
                self._qf2_optimizer = [PCGrad(opt) for opt in self._qf2_optimizer]
        else:
            self._qf1_optimizer = self._optimizer(self._qf1.parameters(), lr=self._lr)
            self._qf2_optimizer = self._optimizer(self._qf2.parameters(), lr=self._lr)
            if self.use_pc_grad:
                self._qf1_optimizer = PCGrad(self._qf1_optimizer)
                self._qf2_optimizer = PCGrad(self._qf2_optimizer)
        # self._policy_scheduler = scheduler(self._policy_optimizer)
        # self._qf1_scheduler = scheduler(self._qf1_optimizer)
        # self._qf2_scheduler = scheduler(self._qf2_optimizer)
        self._grad_clip = grad_clip
        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        self._num_tasks = num_tasks
        self._get_task_id = get_task_id
        assert num_tasks == 1 or get_task_id is not None
        self.preproc_obs = preproc_obs or (lambda x: (x, x))

        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            elif self._discrete:
                self._target_entropy = (
                    np.log(self.env_spec.action_space.flat_dim).item() * 0.98
                )
            else:
                self._target_entropy = -np.prod(self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor(
                [self._initial_log_entropy] * self._num_tasks
            ).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha], lr=self._lr)
            if self.use_pc_grad and not self.skip_alpha_pc_grad:
                self._alpha_optimizer = PCGrad(self._alpha_optimizer)
            # self._alpha_scheduler = scheduler(self._alpha_optimizer)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha] * self._num_tasks).log()
        self.episode_rewards = deque(maxlen=30)

    def train(self, trainer):
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for epoch in trainer.step_epochs():
            ## need to fix evaluation too for kitchen skill
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
                        agent_update = copy.deepcopy(self.policy).cpu()
                else:
                    agent_update = None

                a = time.time()
                # with SuppressStdout():
                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr, batch_size, agent_update=agent_update
                )
                b = time.time()
                path_returns = []
                path_len = 0
                for path in trainer.step_episode:
                    path_actions = path["actions"]
                    path_len += len(path_actions)
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
                    path_returns.append(sum(path["rewards"]))
                assert len(path_returns) == len(trainer.step_episode)
                self.episode_rewards.append(np.mean(path_returns))
                c = time.time()
                for _ in range(self._gradient_steps):
                    loss_dict = self.train_once()
                d = time.time()
                # print(
                #     f"Gather {path_len} data: {b-a}, store data: {c-b}, train {self._gradient_steps} steps: {d-c}"
                # )
            a = time.time()
            if epoch % self._evaluation_frequency == 0 and epoch > 0:
                last_return = self._evaluate_policy(trainer.step_itr)
            b = time.time()
            # print("Evaluation time: {}".format(b - a))
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

    def train_once(self, itr=None, paths=None):
        del itr
        del paths
        loss_dict = None
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            samples = self.replay_buffer.sample_transitions(self._buffer_batch_size)
            samples = as_torch_dict(samples)
            loss_dict = self.optimize_policy(samples)
            self._update_targets()

        return loss_dict

    def _get_log_alpha(self, samples_data):
        if self._num_tasks > 1:
            obs = samples_data["observation"]
            log_alpha = self._log_alpha
            task_ids = self._get_task_id(obs)
            ret = torch.index_select(log_alpha, 0, task_ids)

        else:
            del samples_data
            ret = self._log_alpha
        return ret

    def _temperature_objective(self, log_pi, samples_data):
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            if self._discrete:
                neg_entropy = (log_pi.exp() * log_pi).sum(dim=-1)
            else:
                neg_entropy = log_pi
            alpha_loss = (
                -(self._get_log_alpha(samples_data).exp())
                * (neg_entropy.detach() + self._target_entropy)
            )
            if not self.use_pc_grad or self.skip_alpha_pc_grad:
                alpha_loss = alpha_loss.mean()
        return alpha_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        obs = samples_data["observation"]
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        if self._discrete:
            action_probs = log_pi_new_actions.exp()
            min_q_new_actions = (
                torch.min(self._qf1(obs), self._qf2(obs)) * action_probs
            ).sum(dim=-1, keepdim=True)
            neg_entropy = (action_probs * log_pi_new_actions).sum(dim=-1)
        else:
            if isinstance(self._qf1, list):
                raise NotImplementedError
                min_q_new_actions = torch.min(
                    self._qf1[policy_id](obs, new_actions),
                    self._qf2[policy_id](obs, new_actions),
                )
            min_q_new_actions = torch.min(
                self._qf1(obs, new_actions), self._qf2(obs, new_actions)
            )
            neg_entropy = log_pi_new_actions
        if self.use_pc_grad:
            policy_objective = (alpha * neg_entropy) - min_q_new_actions.flatten()
        else:
            policy_objective = ((alpha * neg_entropy) - min_q_new_actions.flatten()).mean()
        return policy_objective

    def _critic_objective(self, samples_data):
        obs = samples_data["observation"]
        actions = samples_data["action"]
        rewards = samples_data["reward"].flatten()
        terminals = samples_data["terminal"].flatten()
        next_obs = samples_data["next_observation"]
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        if self._discrete:
            q1_pred = self._qf1(obs).gather(1, actions.long())
            q2_pred = self._qf2(obs).gather(1, actions.long())

            new_next_actions_dist = self.policy(next_obs)[0]
            new_next_actions_probs = new_next_actions_dist.probs
            new_log_pi = (
                new_next_actions_probs + (new_next_actions_probs == 0.0).float() * 1e-8
            ).log()

            target_q_values = (
                (
                    torch.min(
                        self._target_qf1(next_obs),
                        self._target_qf2(next_obs),
                    )
                    - (alpha * new_log_pi)
                )
                * new_next_actions_dist.probs
            ).sum(dim=-1)
        else:  # continuous
            q1_pred = self._qf1(obs, actions)
            q2_pred = self._qf2(obs, actions)

            new_next_actions_dist = self.policy(next_obs)[0]
            (
                new_next_actions_pre_tanh,
                new_next_actions,
            ) = new_next_actions_dist.rsample_with_pre_tanh_value()
            new_log_pi = new_next_actions_dist.log_prob(
                value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh
            )

            target_q_values = (
                torch.min(
                    self._target_qf1(next_obs, new_next_actions),
                    self._target_qf2(next_obs, new_next_actions),
                ).flatten()
                - (alpha * new_log_pi)
            )

        with torch.no_grad():
            q_target = (
                rewards * self._reward_scale
                + (1.0 - terminals) * self._discount * target_q_values
            )
        if self.use_pc_grad:
            qf1_loss = F.mse_loss(q1_pred.flatten(), q_target, reduction='none')
            qf2_loss = F.mse_loss(q2_pred.flatten(), q_target, reduction='none')
        else:
            qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
            qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss, q_target, q1_pred, q2_pred

    def _compute_spirl_divergence(self, policy_dist, obs):
        prior_dist = self._spirl_prior(obs)[0]
        div = torch.distributions.kl.kl_divergence(policy_dist, prior_dist)
        div = div.sum(dim=-1)
        return div

    def _update_targets(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self._target_qf1, self._target_qf2]
        qfs = [self._qf1, self._qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(
                    t_param.data * (1.0 - self._tau) + param.data * self._tau
                )

    def optimize_policy(self, samples_data):
        obs = samples_data["observation"]
        action_dists = self.policy(obs)[0]

        if self.use_pc_grad:
            task_ids = self._get_task_id(obs)

        qf1_loss, qf2_loss, q_target, q1_pred, q2_pred = self._critic_objective(
            samples_data
        )

        self._qf1_optimizer.zero_grad()
        if self.use_pc_grad:
            qf1_loss_list = separate_batch(qf1_loss, task_ids, self._num_tasks, reduction='mean')
            self._qf1_optimizer.pc_backward(qf1_loss_list)
            qf1_loss = qf1_loss.mean()
        else:
            qf1_loss.backward()
        if self._grad_clip:
            torch.nn.utils.clip_grad_norm_(self._qf1.parameters(), self._grad_clip)
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        if self.use_pc_grad:
            qf2_loss_list = separate_batch(qf2_loss, task_ids, self._num_tasks, reduction='mean')
            self._qf2_optimizer.pc_backward(qf2_loss_list)
            qf2_loss = qf2_loss.mean()
        else:
            qf2_loss.backward()
        if self._grad_clip:
            torch.nn.utils.clip_grad_norm_(self._qf2.parameters(), self._grad_clip)
        self._qf2_optimizer.step()

        if self._discrete:
            new_actions = action_dists.sample()
            log_pi_new_actions = (
                action_dists.probs + (action_dists.probs == 0.0).float() * 1e-8
            ).log()
        else:
            (
                new_actions_pre_tanh,
                new_actions,
            ) = action_dists.rsample_with_pre_tanh_value()
            log_pi_new_actions = action_dists.log_prob(
                value=new_actions, pre_tanh_value=new_actions_pre_tanh
            )


        policy_loss = self._actor_objective(
            samples_data, new_actions, log_pi_new_actions
        )
        self._policy_optimizer.zero_grad()
        if self._grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_clip)
        if self.use_pc_grad:
            policy_loss_list = separate_batch(policy_loss, task_ids, self._num_tasks, reduction='mean')
            self._policy_optimizer.pc_backward(policy_loss_list)
            policy_loss = policy_loss.mean()
        else:
            policy_loss.backward()

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions, samples_data)
            self._alpha_optimizer.zero_grad()
            if self.use_pc_grad and not self.skip_alpha_pc_grad:
                alpha_loss_list = separate_batch(alpha_loss, task_ids, self._num_tasks, reduction='mean')
                self._alpha_optimizer.pc_backward(alpha_loss_list)
                alpha_loss = alpha_loss.mean()
            else:
                alpha_loss.backward()
            if self._grad_clip:
                torch.nn.utils.clip_grad_norm_([self._log_alpha], self._grad_clip)
            self._alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)

        entropy = action_dists.entropy().mean()
        log_pi = log_pi_new_actions.mean()
        return dict(
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
                self._visualizer.num_videos // len(self._eval_env)
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

        log_wandb(step, infos, medias=videos, prefix="Train/")

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        networks = [
            self.policy,
            self._qf1,
            self._qf2,
            self._target_qf1,
            self._target_qf2,
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
            self._log_alpha = (
                torch.Tensor([self._fixed_alpha] * self._num_tasks).log().to(device)
            )
        else:
            self._log_alpha = (
                torch.Tensor([self._initial_log_entropy] * self._num_tasks)
                .to(device)
                .requires_grad_()
            )
            self._alpha_optimizer = self._optimizer(
                [self._log_alpha], lr=self._lr
            )
            if self.use_pc_grad and not self.skip_alpha_pc_grad:
                self._alpha_optimizer = PCGrad(self._alpha_optimizer)
