#!/usr/bin/env python3
"""
from garage examples torch ppo/trpo pendulum
"""
import torch
import wandb
import os


from garage import wrap_experiment
from garage.replay_buffer import PathBuffer
from garage.torch import set_gpu_mode
from garage.experiment.deterministic import set_seed
from garage.trainer import Trainer

from learning.policies.multihead_continuous_q_function import DECAFQFunction
from learning.policies.multi_policy_wrapper import MultiPolicyWrapper
from learning.policies.qswitch_mixture_policies_wrapper import (
    QswitchMixtureOfPoliciesWrapper,
)
from learning.algorithms import (
    SAC,
    DnCSAC,
    MoPDnC,
    MoPSAC,
    OnlineCDS,
    QMPUDS,
    MultiCriticAL,
    QMPMultiCriticAL,
)

from learning.utils.path_buffers import DnCPathBuffer
from experiment_utils import setup
import environments


def run_resume_test(config):
    assert config.snapshot_dir is not None
    @wrap_experiment(
        archive_launch_repo=False,
        name=config.snapshot_dir,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        log_dir=config.snapshot_dir,
        use_existing_dir=True,
    )
    def resume_test(ctxt=None,
        config=None,):
        ### need to resume wandb also
        set_seed(config.seed)
        trainer = Trainer(snapshot_config=ctxt)
        trainer.restore(config.snapshot_dir)

        # for KAIST machine runs
        # torch.set_num_threads(4)
        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)

        trainer._algo.to()

        with open(os.path.join(config.snapshot_dir, "wandb_run_id.txt"), "r") as f:
            wandb_run_id = f.read()
        wandb.init(project=config.wandb_project, dir=config.snapshot_dir, id=wandb_run_id, resume=True)


        trainer.resume()
    resume_test(config=config)

def run_dnc_sac_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("dnc_sac", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def dnc_sac_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        assert (
            config.policy_architecture == "separate"
            and config.Q_architecture == "separate"
        )

        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=DnCPathBuffer,
            policy_wrapper_cls=MultiPolicyWrapper,
        )

        dnc_sac = DnCSAC(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            policies=setup_dict["policies"],
            qf1s=setup_dict["qf1s"],
            qf2s=setup_dict["qf2s"],
            lr=config.lr,
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            get_stage_id=setup_dict["get_stage_id"],
            preproc_obs=setup_dict["split_observation"],
            initial_kl_coeff=config.kl_coeff,
            sampling_type=config.sampling_type,
            n_policies=config.n_policies,
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffers=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
            regularize_representation=config.regularize_representation,
            distillation_period=config.distillation_period,
            distillation_n_epochs=config.distillation_n_epochs,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        dnc_sac.to()

        trainer.setup(algo=dnc_sac, env=setup_dict["train_envs"])
        trainer.train(
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
        )

    dnc_sac_test(config=config, env_args=env_args)


def run_qmp_dnc_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("qmp_dnc", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def qmp_dnc_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        assert (
            config.policy_architecture == "separate"
            and config.Q_architecture == "separate"
            and not env_args["include_task_id"]
        )
        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=DnCPathBuffer,
            policy_wrapper_cls=QswitchMixtureOfPoliciesWrapper,
        )

        dnc_sac = MoPDnC(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            policies=setup_dict["policies"],
            qf1s=setup_dict["qf1s"],
            qf2s=setup_dict["qf2s"],
            lr=config.lr,
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            get_stage_id=None,
            preproc_obs=setup_dict["split_observation"],
            initial_kl_coeff=config.kl_coeff,
            sampling_type=config.sampling_type,
            n_policies=config.n_policies,
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffers=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
            regularize_representation=config.regularize_representation,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        dnc_sac.to()

        trainer.setup(algo=dnc_sac, env=setup_dict["train_envs"])
        trainer.train(n_epochs=config.n_epochs, batch_size=config.batch_size)

    qmp_dnc_test(config=config, env_args=env_args)


def run_qmp_sac_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("sac_qmp", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def qmp_sac_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        ### only implemented for multihead architecture without task_id rewriting
        assert (
            config.policy_architecture == "multihead"
            and config.Q_architecture == "multihead"
            and not env_args["include_task_id"]
        )
        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=PathBuffer,
            policy_wrapper_cls=QswitchMixtureOfPoliciesWrapper,
        )

        num_tasks = setup_dict["base_env"].num_tasks
        get_task_id = getattr(setup_dict["base_env"], "get_task_id", None)

        mop_sac = MoPSAC(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            mop_policy=setup_dict["policy"],
            qf1=setup_dict["qf1s"][0],
            qf2=setup_dict["qf2s"][0],
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            num_tasks=num_tasks,
            get_task_id=get_task_id,  ###?
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            preproc_obs=setup_dict["split_observation"],
            multihead=(config.policy_architecture == "multihead"),
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffer=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            lr=config.lr,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
            use_pc_grad=config.use_pc_grad,
            skip_alpha_pc_grad=config.skip_alpha_pc_grad,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        mop_sac.to()

        trainer.setup(algo=mop_sac, env=setup_dict["train_envs"])
        trainer.train(n_epochs=config.n_epochs, batch_size=config.batch_size)

    qmp_sac_test(config=config, env_args=env_args)


def run_sac_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("sac", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def sac_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        """Set up environment and algorithm and run the task.
        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.
        """
        assert (
            config.policy_architecture != "separate"
            and config.Q_architecture != "separate"
        )

        trainer, setup_dict = setup(
            ctxt=ctxt, config=config, env_args=env_args, replay_buffer_cls=PathBuffer
        )

        num_tasks = (
            getattr(setup_dict["base_env"], "num_tasks", 1)
            if config.experiment == "mtsac"
            else 1
        )

        if num_tasks > 1 and (
            config.policy_architecture == "shared" or config.Q_architecture == "shared"
        ):
            assert env_args["include_task_id"]

        get_task_id = getattr(setup_dict["base_env"], "get_task_id", None)

        ### ASDF SAC does not support split observation except in multihead policies
        sac = SAC(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            qf1=setup_dict["qf1s"][0],
            qf2=setup_dict["qf2s"][0],
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            preproc_obs=setup_dict["split_observation"],
            multihead=(config.policy_architecture == "multihead"),
            num_tasks=num_tasks,
            get_task_id=get_task_id,  ###?
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffer=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            lr=config.lr,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
            use_pc_grad=config.use_pc_grad,
            skip_alpha_pc_grad=config.skip_alpha_pc_grad,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        sac.to()

        trainer.setup(algo=sac, env=setup_dict["train_envs"])
        trainer.train(n_epochs=config.n_epochs, batch_size=config.batch_size)

    sac_test(config=config, env_args=env_args)

def run_uds_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("cds", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def cds_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        assert (
            config.policy_architecture == "separate"
            and config.Q_architecture == "separate"
            and config.kl_coeff == [0]
        )

        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=DnCPathBuffer,
            policy_wrapper_cls=MultiPolicyWrapper,
        )

        reward_fns = [getattr(train_env, "reward_fn", None) for train_env in setup_dict["train_envs"]]
        min_reward = getattr(setup_dict["base_env"], 'min_reward', None)

        cds = OnlineCDS(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            policies=setup_dict["policies"],
            qf1s=setup_dict["qf1s"],
            qf2s=setup_dict["qf2s"],
            lr=config.lr,
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            get_stage_id=setup_dict["get_stage_id"],
            preproc_obs=setup_dict["split_observation"],
            initial_kl_coeff=config.kl_coeff,
            sampling_type=config.sampling_type,
            n_policies=config.n_policies,
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffers=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
            regularize_representation=config.regularize_representation,
            distillation_period=config.distillation_period,
            distillation_n_epochs=config.distillation_n_epochs,
            reward_fns=reward_fns,
            min_reward=min_reward,
            sharing_quantile=config.sharing_quantile,
            unsupervised=config.unsupervised,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        cds.to()

        trainer.setup(algo=cds, env=setup_dict["train_envs"])
        trainer.train(
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
        )

    cds_test(config=config, env_args=env_args)

def run_qmp_uds_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("qmp_cds_dnc", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def qmp_uds_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        assert (
            config.policy_architecture == "separate"
            and config.Q_architecture == "separate"
            and config.kl_coeff == [0]
        )

        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=DnCPathBuffer,
            policy_wrapper_cls=QswitchMixtureOfPoliciesWrapper,
        )

        reward_fns = [getattr(train_env, "reward_fn", None) for train_env in setup_dict["train_envs"]]
        min_reward = getattr(setup_dict["base_env"], 'min_reward', None)

        qmp_uds = QMPUDS(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            policies=setup_dict["policies"],
            qf1s=setup_dict["qf1s"],
            qf2s=setup_dict["qf2s"],
            lr=config.lr,
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            get_stage_id=None,
            preproc_obs=setup_dict["split_observation"],
            initial_kl_coeff=config.kl_coeff,
            sampling_type=config.sampling_type,
            n_policies=config.n_policies,
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffers=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
            regularize_representation=config.regularize_representation,
            distillation_period=config.distillation_period,
            distillation_n_epochs=config.distillation_n_epochs,
            reward_fns=reward_fns,
            min_reward=min_reward,
            sharing_quantile=config.sharing_quantile,
            unsupervised=config.unsupervised,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        qmp_uds.to()

        trainer.setup(algo=qmp_uds, env=setup_dict["train_envs"])
        trainer.train(
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
        )

    qmp_uds_test(config=config, env_args=env_args)


def run_decaf_test(
    config,
    env_args,
):
    ### Identical to DnC_SAC except Q functions

    function_name = "_".join(
        ("decaf", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def decaf_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        assert (
            config.policy_architecture == "separate"
            and config.Q_architecture == "separate"
            and config.kl_coeff == [0]
        )

        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=DnCPathBuffer,
            policy_wrapper_cls=MultiPolicyWrapper,
        )

        ### DECAF: redefine qf1s and qf2s
        qf1s_decaf = [
            DECAFQFunction(
                env_spec=setup_dict["env_spec"],
                all_q_functions=setup_dict["qf1s"],
                task_id=i,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                layer_normalization=config.layer_norm,
            )
            for i in range(config.n_policies)
        ]
        qf2s_decaf = [
            DECAFQFunction(
                env_spec=setup_dict["env_spec"],
                all_q_functions=setup_dict["qf2s"],
                task_id=i,
                hidden_sizes=config.hidden_sizes,
                hidden_nonlinearity=torch.relu,
                layer_normalization=config.layer_norm,
            )
            for i in range(config.n_policies)
        ]


        decaf = DnCSAC(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            policies=setup_dict["policies"],
            qf1s=qf1s_decaf,
            qf2s=qf2s_decaf,
            lr=config.lr,
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            get_stage_id=setup_dict["get_stage_id"],
            preproc_obs=setup_dict["split_observation"],
            initial_kl_coeff=config.kl_coeff,
            sampling_type=config.sampling_type,
            n_policies=config.n_policies,
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffers=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
            regularize_representation=config.regularize_representation,
            distillation_period=config.distillation_period,
            distillation_n_epochs=config.distillation_n_epochs,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        decaf.to()

        trainer.setup(algo=decaf, env=setup_dict["train_envs"])
        trainer.train(
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
        )

    decaf_test(config=config, env_args=env_args)

def run_mcal_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("mcal", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def mcal_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        assert (
            config.policy_architecture == "multihead"
            and config.Q_architecture == "separate"
        )

        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=DnCPathBuffer,
        )

        get_task_id = getattr(setup_dict["base_env"], "get_task_id", None)

        mcal = MultiCriticAL(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            qf1s=setup_dict["qf1s"],
            qf2s=setup_dict["qf2s"],
            lr=config.lr,
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            get_stage_id=setup_dict["get_stage_id"],
            preproc_obs=setup_dict["split_observation"],
            sampling_type=config.sampling_type,
            n_tasks=config.n_policies,
            get_task_id=get_task_id,
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffers=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        mcal.to()

        trainer.setup(algo=mcal, env=setup_dict["train_envs"])
        trainer.train(
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
        )

    mcal_test(config=config, env_args=env_args)

def run_qmp_mcal_test(
    config,
    env_args,
):
    function_name = "_".join(
        ("qmp_mcal", config.env, config.name.replace(" ", "-"), str(config.seed))
    )

    @wrap_experiment(
        archive_launch_repo=False,
        name=function_name,
        snapshot_mode="last" if config.snapshot_gap == 1 else "gap_overwrite",
        snapshot_gap=config.snapshot_gap,
        use_existing_dir=True,
    )
    def qmp_mcal_test(
        ctxt=None,
        config=None,
        env_args=None,
    ):
        assert (
            config.policy_architecture == "multihead"
            and config.Q_architecture == "separate"
        )

        trainer, setup_dict = setup(
            ctxt=ctxt,
            config=config,
            env_args=env_args,
            replay_buffer_cls=DnCPathBuffer,
            policy_wrapper_cls=QswitchMixtureOfPoliciesWrapper,

        )

        get_task_id = getattr(setup_dict["base_env"], "get_task_id", None)

        qmp_mcal = QMPMultiCriticAL(
            env_spec=setup_dict["env_spec"],
            policy=setup_dict["policy"],
            qf1s=setup_dict["qf1s"],
            qf2s=setup_dict["qf2s"],
            lr=config.lr,
            sampler=setup_dict["sampler"],
            visualizer=setup_dict["visualizer"],
            get_stage_id=setup_dict["get_stage_id"],
            preproc_obs=setup_dict["split_observation"],
            sampling_type=config.sampling_type,
            n_tasks=config.n_policies,
            get_task_id=get_task_id,
            gradient_steps_per_itr=config.gradient_steps_per_itr,
            max_episode_length_eval=setup_dict["env_spec"].max_episode_length,
            replay_buffers=setup_dict["replay_buffer"],
            min_buffer_size=config.min_buffer_size,
            target_entropy=config.target_entropy,
            target_update_tau=config.target_update_tau,
            discount=config.discount,
            buffer_batch_size=config.buffer_batch_size,
            reward_scale=config.reward_scale,
            steps_per_epoch=config.steps_per_epoch,
            eval_env=setup_dict["test_envs"],
            num_evaluation_episodes=config.num_evaluation_episodes,
            evaluation_frequency=config.evaluation_frequency,
        )

        if torch.cuda.is_available() and config.gpu is not None:
            set_gpu_mode(True, gpu_id=config.gpu)
        else:
            set_gpu_mode(False)
        qmp_mcal.to()

        trainer.setup(algo=qmp_mcal, env=setup_dict["train_envs"])
        trainer.train(
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
        )

    qmp_mcal_test(config=config, env_args=env_args)
