import os
import sys

from collections import defaultdict
import warnings
import time
import numpy as np
import wandb
import torch
from dowel import tabular
import click

from garage import EpisodeBatch, StepType
from garage.torch.optimizers import OptimizerWrapper
from garage.np.optimizers import BatchDataset
from garage.np import stack_tensor_dict_list, discount_cumsum

from learning.utils.general import np_to_torch, list_to_tensor


def get_path_policy_id(path):
    # Returns the policy id used to gather a single path.

    ### ASDF assuming each episode is gathered by a single policy
    policy_ids = get_policy_ids(path)
    assert np.all(policy_ids == policy_ids[0])
    return policy_ids[0]


def get_policy_ids(path):
    policy_ids = path["agent_infos"]["policy_id"]
    return policy_ids


def get_path_task_id(path):
    # Returns the policy id used to gather a single path.

    ### ASDF assuming each episode is gathered by a single policy
    task_ids = path["agent_infos"]["task_id"]
    assert np.all(task_ids == task_ids[0])
    return task_ids[0]


def extract_policy_samples(samples, n_policies):
    # Returns a num_policies long list of samples sorted by the policy that gathered them.

    policy_samples = [[] for _ in range(n_policies)]

    for ep in samples.split():
        ### ASDF assuming each episode is gathered by a single policy
        policy_id = ep.agent_infos["policy_id"][0]
        policy_samples[policy_id].append(ep)

    for (i, samples) in enumerate(policy_samples):
        if len(samples) == 0:
            warnings.warn("Policy {} collected no samples this batch".format(i))
        else:
            policy_samples[i] = EpisodeBatch.concatenate(*samples)

    return policy_samples


class DnCOptimizerWrapper(OptimizerWrapper):
    """Used by DnC on-policy methods (DnCVPG, DnCPPO) to draw policy specific and shared data"""

    def get_minibatch(self, shared_inputs, *inputs):
        """
        Draws minibatch independently from shared_inputs and inputs in case they are different sizes.
        """
        shared_dataset = BatchDataset([shared_inputs], self._minibatch_size)
        batch_dataset = BatchDataset(inputs, self._minibatch_size)

        for _ in range(self._max_optimization_epochs):
            for (shared, dataset) in zip(
                shared_dataset.iterate(), batch_dataset.iterate()
            ):
                yield shared, dataset


def rollout(
    env,
    agent,
    *,
    max_episode_length=np.inf,
    animated=False,
    pause_per_frame=None,
    deterministic=False,
    save_video=False,
    evaluation_mode=None,
    visualizer=None,
):
    """from garage._functions"""
    env_steps = []
    agent_infos = []
    observations = []
    frames = []
    if hasattr(env, "set_record_frames"):
        env.set_record_frames(save_video)
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    if animated:
        env.visualize()

    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)


        if evaluation_mode is None:
            a, agent_info = agent.get_action(last_obs)
        else:
            a, agent_info = agent.get_action(last_obs, evaluation_mode=evaluation_mode)
        if deterministic and "mean" in agent_info:
            a = agent_info["mean"]
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)

        if save_video and visualizer is not None:
            info = {"step": episode_length}
            if episode_length > 0:
                info["reward"] = es.reward
                info.update(es.env_info)
                info.update(agent_info)
            frame = env.render(mode="rgb_array")
            visualizer.add(frame, info)
            if isinstance(frame, list):
                frames.extend(frame)
            else:
                frames.append(frame)

        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

    if hasattr(env, "set_record_frames"):
        env.set_record_frames(False)

    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
        # frames=frames,
    )


def obtain_evaluation_episodes(
    policy,
    env,
    max_episode_length=1000,
    num_eps=100,
    deterministic=True,
    evaluation_mode=None,
    visualizer=None,
    num_vis=0,
):
    episodes = []
    # Use a finite length rollout for evaluation.

    with click.progressbar(range(num_eps), label="Evaluating") as pbar:
        for i, _ in enumerate(pbar):
            if evaluation_mode is None:
                eps = rollout(
                    env,
                    policy,
                    max_episode_length=max_episode_length,
                    deterministic=deterministic,
                    visualizer=visualizer,
                    save_video=i < num_vis,
                )
            else:
                eps = rollout(
                    env,
                    policy,
                    max_episode_length=max_episode_length,
                    deterministic=deterministic,
                    evaluation_mode=evaluation_mode,
                    visualizer=visualizer,
                    save_video=i < num_vis,
                )
            episodes.append(eps)
    return EpisodeBatch.from_list(env.spec, episodes)


def obtain_multitask_multimode_evaluation_episodes(
    policy,
    envs,
    max_episode_length=1000,
    num_eps_per_mode=10,
    deterministic=True,
    evaluation_modes=[None],
    visualizer=None,
    num_vis=0,
):
    all_episodes = {}
    all_videos = {}

    if visualizer is not None:
        visualizer.reset()

    with click.progressbar(
        range(num_eps_per_mode * len(evaluation_modes) * len(envs)), label="Evaluating"
    ) as pbar:
        for evaluation_mode in evaluation_modes:
            episodes = []
            videos = []
            for env in envs:
                for i in range(num_eps_per_mode):
                    eps = rollout(
                        env,
                        policy,
                        max_episode_length=max_episode_length,
                        deterministic=deterministic,
                        evaluation_mode=evaluation_mode,
                        visualizer=visualizer,
                        save_video=i < num_vis,
                    )
                    episodes.append(eps)
                    pbar.update(1)
                if visualizer is not None:
                    videos.append(visualizer.get_video())
                    visualizer.reset()
                else:
                    videos.append(None)
            all_videos[evaluation_mode] = videos
            all_episodes[evaluation_mode] = EpisodeBatch.from_list(env.spec, episodes)
    return all_episodes, all_videos


def log_performance(itr, batch, discount, videos=None, prefix="Evaluation/"):
    """
    Overwriting garage log_performance function to add wandb logging.
    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []
    stages_completed = []
    wrong_stages_completed = []
    x_velocities = []
    infos = {}
    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))
        termination.append(
            float(any(step_type == StepType.TERMINAL for step_type in eps.step_types))
        )
        if "success" in eps.env_infos:
            success.append(float(eps.env_infos["success"].any()))
        if "stages_completed" in eps.env_infos:
            stages_completed.append(eps.env_infos["stages_completed"][-1])
        if "wrong_stages_completed" in eps.env_infos:
            wrong_stages_completed.append(eps.env_infos["wrong_stages_completed"][-1])
        if "x_velocity" in eps.env_infos:
            x_velocities.append(np.mean(eps.env_infos["x_velocity"]))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    infos["Iteration"] = itr
    infos["NumEpisodes"] = len(returns)
    infos["AverageEpisodeLength"] = np.mean([len(rtn) for rtn in returns])
    infos["AverageDiscountedReturn"] = average_discounted_return
    infos["AverageReturn"] = np.mean(undiscounted_returns)
    infos["StdReturn"] = np.std(undiscounted_returns)
    infos["MaxReturn"] = np.max(undiscounted_returns)
    infos["MinReturn"] = np.min(undiscounted_returns)
    infos["TerminationRate"] = np.mean(termination)

    if success:
        infos["SuccessRate"] = np.mean(success)
    if stages_completed:
        infos["StagesCompleted"] = np.mean(stages_completed)
    if wrong_stages_completed:
        infos["WrongStagesCompleted"] = np.mean(wrong_stages_completed)
    if x_velocities:
        infos["Velocity"] = np.mean(x_velocities)

    log_wandb(step=itr, infos=infos, medias=videos, prefix=prefix)

    return undiscounted_returns


def log_multitask_performance(itr, batch, discount, videos=None, prefix="Evaluation/"):
    """
    Log performance of episodes from multiple tasks.
    """
    episodes = defaultdict(list)
    for i, eps in enumerate(batch.split()):
        task_id = eps.env_infos.get("task_id", [i])[0]
        episodes[task_id].append(eps)
    for task_id, eps in episodes.items():
        task_name = prefix + "Task{}/".format(task_id)
        if videos is not None and len(videos) != 0:
            log_performance(
                itr,
                EpisodeBatch.concatenate(*eps),
                discount,
                videos=videos[task_id],
                prefix=task_name,
            )
        else:
            log_performance(
                itr, EpisodeBatch.concatenate(*eps), discount, prefix=task_name
            )

    # log average
    return log_performance(itr, batch, discount=discount, prefix=prefix)


def log_wandb(step, infos, medias=None, prefix=""):

    ### Record infos locally
    if "Task" not in prefix and "LocalPolicy" not in prefix:
        with tabular.prefix(prefix):
            for k, v in infos.items():
                tabular.record(k, v)

    ### Log infos to wandb
    for k, v in infos.items():
        wandb.log({"%s%s" % (prefix, k): v}, step=step)

    ### Why is it separate for videos/images?
    if medias is not None:
        for k, v in medias.items():
            if isinstance(v, wandb.Video):
                wandb.log({"%s%s" % (prefix, k): v}, step=step)
            elif isinstance(v, list) and isinstance(v[0], wandb.Video):
                for i, video in enumerate(v):
                    wandb.log({"%s%s_%d" % (prefix, k, i): video}, step=step)
            else:
                wandb.log({"%s%s" % (prefix, k): [wandb.Image(v)]}, step=step)


class ParameterScheduler:
    def __init__(
        self, init_value, step_size, step_type, max_value=None, min_value=None
    ):
        self.step_size = step_size
        self.step_type = step_type
        self.current_value = init_value
        self.max_value = max_value
        self.min_value = min_value

    def step(self):
        if self.step_type == "+":
            self.current_value += self.step_size
        elif self.step_type == "*":
            self.current_value *= self.step_size

        if self.max_value is not None:
            self.current_value = min(self.current_value, self.max_value)

        if self.min_value is not None:
            self.current_value = max(self.current_value, self.min_value)

        return self.current_value
