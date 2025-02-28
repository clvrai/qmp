import os
import wandb
import torch
import numpy as np

from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.q_functions import (
    ContinuousMLPQFunction,
    DiscreteMLPQFunction,
)
from garage.torch import set_gpu_mode
from garage.trainer import Trainer
from garage.sampler import (
    LocalSampler,
    RaySampler,
    MultiprocessingSampler,
    FragmentWorker,
    VecWorker,
    DefaultWorker,
)

from environments.gym_env import ArgsGymEnv

from learning.policies.multi_policy_wrapper import MultiPolicyWrapper
from learning.policies.qswitch_mixture_policies_wrapper import (
    QswitchMixtureOfPoliciesWrapper,
)
from learning.policies.env_partition_policy import EnvPartitionPolicy

from learning.policies import (
    NamedTanhGaussianMLPPolicy,
    MultiheadTanhGaussianMLPPolicy,
    MultiheadContinuousMLPQFunction,
)
from learning.utils.path_buffers import DnCPathBuffer
from learning.utils.visualizer import Visualizer
from learning.utils.mt_local_sampler import MTLocalSampler


def init_wandb(config, env_args, log_dir):
    exclude = ["device"]

    if not config.wandb:
        os.environ["WANDB_MODE"] = "dryrun"

    all_configs = {
        **{k: v for k, v in config.__dict__.items() if k not in exclude},
        **{k: v for k, v in env_args.items() if k not in exclude},
    }
    wandb_run_id = wandb.util.generate_id()
    wandb.init(
        name="_".join(
            (
                config.experiment,
                config.env,
                config.name.replace(" ", "-"),
                str(config.seed),
            )
        ),
        project=config.wandb_project,
        config=all_configs,
        dir=log_dir,
        entity=config.wandb_entity,
        notes=config.notes,
        id=wandb_run_id,
    )

    with open (os.path.join(log_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run_id)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_partition(partition, env, n_policies):
    if partition == "random":
        policy_assigner = EnvPartitionPolicy(
            env_spec=env.spec, mode="random", num_partitions=n_policies
        )
    elif partition == "task_id":
        assert hasattr(env, "get_task_id") and env.num_tasks == n_policies
        policy_assigner = EnvPartitionPolicy(
            env_spec=env.spec,
            mode="fixed",
            num_partitions=n_policies,
            partitions=env.get_task_id,
        )
    else:
        raise NotImplementedError

    return policy_assigner


def get_mt_envs(env, partition, n_policies, norm_obs, env_args=None):
    ### HACK for multi-task envs reacher vs MT10
    base_env = ArgsGymEnv(env, env_args)
    env_spec = base_env.spec
    if hasattr(base_env, "get_train_envs"):
        train_envs = base_env.get_train_envs()
        try:
            train_envs = [normalize(ArgsGymEnv(env), normalize_obs=norm_obs) for env in train_envs]
        except:
            print("Failed to normalize train envs")
    else:
        train_envs = normalize(base_env, normalize_obs=norm_obs)

    if hasattr(base_env, "get_test_envs"):
        test_envs = base_env.get_test_envs()
        try:
            test_envs = [normalize(ArgsGymEnv(env), normalize_obs=norm_obs) for env in test_envs]
        except:
            print("Failed to normalize test envs")
    else:
        test_envs = normalize(base_env, normalize_obs=norm_obs)

    split_observation = getattr(base_env, "split_observation", None)
    policy_assigner = get_partition(partition, base_env, n_policies)

    return base_env, env_spec, train_envs, test_envs, split_observation, policy_assigner


def get_policies_and_qfs(config, env_spec, policy_assigner, split_observation):

    if config.policy_architecture == "separate":
        policies = [
            NamedTanhGaussianMLPPolicy(
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                output_nonlinearity=None,
                min_std=np.exp(-20.0),
                max_std=np.exp(2.0),
                name="LocalPolicy{}".format(i),
                # split_observation=split_observation,
            )
            for i in range(config.n_policies)
        ]

    elif config.policy_architecture == "multihead":
        policies = [
            MultiheadTanhGaussianMLPPolicy(
                num_heads=config.n_policies,
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                output_nonlinearity=None,
                min_std=np.exp(-20.0),
                max_std=np.exp(2.0),
                policy_assigner=policy_assigner,
                split_observation=split_observation,
            )
        ]

    else:
        policies = [
            NamedTanhGaussianMLPPolicy(
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                output_nonlinearity=None,
                min_std=np.exp(-20.0),
                max_std=np.exp(2.0),
                split_observation=split_observation,
            )
        ]

    num_policy_parameters = np.sum([count_parameters(policy) for policy in policies])

    if config.Q_architecture == "separate":
        qf1s = [
            ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                layer_normalization=config.layer_norm,
            )
            for i in range(config.n_policies)
        ]

        qf2s = [
            ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                layer_normalization=config.layer_norm,
            )
            for i in range(config.n_policies)
        ]

    elif config.Q_architecture == "multihead":
        qf1s = [
            MultiheadContinuousMLPQFunction(
                num_heads=config.n_policies,
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                policy_assigner=policy_assigner,
                split_observation=split_observation,
                layer_normalization=config.layer_norm,
            )
        ]

        qf2s = [
            MultiheadContinuousMLPQFunction(
                num_heads=config.n_policies,
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                policy_assigner=policy_assigner,
                split_observation=split_observation,
                layer_normalization=config.layer_norm,
            )
        ]

    else:
        qf1s = [
            ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                layer_normalization=config.layer_norm,
            )
        ]

        qf2s = [
            ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                layer_normalization=config.layer_norm,
            )
        ]

    num_Q_parameters = 2 * np.sum([count_parameters(qf) for qf in qf1s])

    print(
        f"Creating policies with {config.policy_architecture} and q functions with {config.Q_architecture} with n_policies: {config.n_policies} ... "
    )

    print(
        f"Num parameters in policy: {num_policy_parameters}, Num parameters in Q functions: {num_Q_parameters}"
    )

    return policies, num_policy_parameters, qf1s, qf2s, num_Q_parameters


def setup(ctxt, config, env_args, replay_buffer_cls, policy_wrapper_cls=None):
    set_seed(config.seed)
    trainer = Trainer(snapshot_config=ctxt)

    ### Set up Logging
    init_wandb(config, env_args, trainer._snapshotter.snapshot_dir)

    visualizer = Visualizer(
        freq=config.vis_freq,
        num_videos=config.vis_num,
        imsize=(config.vis_width, config.vis_height),
        fps=config.vis_fps,
        format=config.vis_format,
    )

    ### Set up environment
    (
        base_env,
        env_spec,
        train_envs,
        test_envs,
        split_observation,
        policy_assigner,
    ) = get_mt_envs(config.env, config.partition, config.n_policies, config.norm_obs, env_args)

    if hasattr(config, "stagewise") and config.stagewise:
        assert hasattr(base_env, "get_stage_id")
        get_stage_id = base_env.get_stage_id
    else:
        get_stage_id = None

    ### Set up models/networks
    (
        policies,
        num_policy_parameters,
        qf1s,
        qf2s,
        num_Q_parameters,
    ) = get_policies_and_qfs(config, env_spec, policy_assigner, split_observation)

    if policy_wrapper_cls is MultiPolicyWrapper:
        policy = MultiPolicyWrapper(policies, policy_assigner, split_observation)
    elif policy_wrapper_cls is QswitchMixtureOfPoliciesWrapper:
        policy = QswitchMixtureOfPoliciesWrapper(
            policies,
            config.policy_architecture,
            qf1s,
            qf2s,
            config.Q_architecture,
            policy_assigner,
            sampling_freq=config.policy_sampling_freq,
            split_observation=split_observation,
        )
    elif policy_wrapper_cls is None:
        policy = policies[0]
    else:
        raise NotImplementedError

    ### Set up data collection --> Hard to do
    n_workers = len(train_envs) if isinstance(train_envs, list) else config.max_n_worker
    if config.sampler_type == "local":
        sampler_cls = LocalSampler
    elif config.sampler_type == "mp":
        sampler_cls = MultiprocessingSampler
    elif config.sampler_type == "ray":
        sampler_cls = RaySampler
    elif config.sampler_type == "mt_local":
        sampler_cls = MTLocalSampler

    worker_cls = (
        DefaultWorker if config.worker_type == "default" else FragmentWorker
    )

    sampler = sampler_cls(
        agents=policy,
        envs=train_envs,
        max_episode_length=env_spec.max_episode_length,
        worker_class=worker_cls,
        n_workers=n_workers,
    )

    if replay_buffer_cls is PathBuffer:
        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    elif replay_buffer_cls is DnCPathBuffer:
        replay_buffer = DnCPathBuffer(
            num_buffers=config.n_policies,
            capacity_in_transitions=int(1e6),
            sampling_type=config.sampling_type,
        )
    else:
        raise NotImplementedError

    if config.worker_type == "fragment":
        if config.batch_size < 10:
            config.batch_size = config.batch_size * FragmentWorker.DEFAULT_N_ENVS
            config.gradient_steps_per_itr = (
                config.gradient_steps_per_itr * FragmentWorker.DEFAULT_N_ENVS
            )
            config.steps_per_epoch = max(
                config.steps_per_epoch // FragmentWorker.DEFAULT_N_ENVS, 1
            )

    if torch.cuda.is_available() and config.gpu is not None:
        set_gpu_mode(True, gpu_id=config.gpu)
    else:
        set_gpu_mode(False)

    return trainer, dict(
        env_spec=env_spec,
        policy=policy,
        policies=policies,
        qf1s=qf1s,
        qf2s=qf2s,
        sampler=sampler,
        visualizer=visualizer,
        get_stage_id=get_stage_id,
        split_observation=split_observation,
        replay_buffer=replay_buffer,
        base_env=base_env,
        train_envs=train_envs,
        test_envs=test_envs,
    )
