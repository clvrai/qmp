import argparse
import warnings
import sys

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)
# Ignore all warnings
warnings.filterwarnings("ignore")

from garage_experiments import (
    run_sac_test,
    run_dnc_sac_test,
    run_qmp_dnc_test,
    run_qmp_sac_test,
    run_uds_test,
    run_qmp_uds_test,
    run_resume_test,
    run_decaf_test,
    run_mcal_test,
    run_qmp_mcal_test,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=[
            "sac",
            "mtsac",
            "qmp_dnc",
            "qmp_sac",
            "dnc_sac",
            "uds_dnc",
            "qmp_uds_dnc",
            "resume",
            "decaf",
            "mcal",
            "qmp_mcal"
        ],
    )
    parser.add_argument("--env", type=str, default="InvertedPendulum-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="None")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max_n_worker", type=int, default=4)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="p-task-mod")
    parser.add_argument("--wandb_entity", type=str, default="clvr")
    parser.add_argument("--discrete_action", type=str2bool, default=False)
    parser.add_argument("--norm_obs", type=str2bool, default=False)

    ### Training arguments
    parser.add_argument("--gradient_steps_per_itr", type=int, default=1000) ## Split across all tasks
    parser.add_argument("--min_buffer_size", type=int, default=int(1e4))
    parser.add_argument("--target_update_tau", type=float, default=5e-3)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--buffer_batch_size", type=int, default=32)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--steps_per_epoch", type=int, default=1)
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--layer_norm", type=str2bool, default=False)
    parser.add_argument("--use_pc_grad", type=str2bool, default=False)
    parser.add_argument("--skip_alpha_pc_grad", type=str2bool, default=True)
    parser.add_argument(
        "--policy_architecture",
        type=str,
        choices=["separate", "shared", "multihead"],
        default="separate",
    )
    parser.add_argument(
        "--Q_architecture",
        type=str,
        choices=["separate", "shared", "multihead"],
        default="separate",
    )
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1000)  ## Split across all tasks
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument(
        "--sampler_type", choices=["local", "ray", "mp", "mt_local"], default="local"
    )
    parser.add_argument(
        "--worker_type", choices=["default", "fragment"], default="default"
    )

    ### Evaluation arguments
    parser.add_argument("--num_evaluation_episodes", type=int, default=10)
    parser.add_argument("--evaluation_frequency", type=int, default=1)
    parser.add_argument(
        "--vis_freq", type=int, default=10, help="-1 for no visualization"
    )
    parser.add_argument("--vis_num", type=int, default=10)
    parser.add_argument("--vis_width", type=int, default=500)
    parser.add_argument("--vis_height", type=int, default=500)
    parser.add_argument("--vis_fps", type=int, default=40)
    parser.add_argument("--vis_format", type=str, default="mp4")
    parser.add_argument("--snapshot_gap", type=int, default=1)
    parser.add_argument("--snapshot_dir", type=str)

    ### Task arguments
    parser.add_argument("--n_policies", type=int, default=4)
    parser.add_argument(
        "--partition",
        default="task_id",
        choices=[
            "random",
            "goal_quadrant",
            "obstacle_id",
            "obstacle_orientation",
            "goal_cluster",
            "task_id",
        ],
    )
    parser.add_argument(
        "--sampling_type",
        type=str,
        choices=["all", "i", "j", "i+j", "ixj"],
        default="all",
    )
    # parser.add_argument("--goal_type", type=int, default=0)
    parser.add_argument("--include_task_id", type=str2bool, default=False)
    parser.add_argument("--sparse_tasks", nargs="+", type=int, default=[])
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--control_penalty", type=float, default=0.0)
    parser.add_argument("--reward_params", nargs="+", type=float, default=[0, 0])
    parser.add_argument(
        "--reward_type", type=str, choices=["shift", "scale", "shaped"], default="shift"
    )


    args = parser.parse_known_args()[0]

    ### Add method specific arguments
    if "dnc" in args.experiment or "decaf" in args.experiment:
        parser.add_argument(
            "--kl_coeff",
            nargs="+",
            type=float,
            default=[0],
            help="order: 1->1 2->1 ... N->1 1->2 2->2 ... N->2 ...",
        )
        parser.add_argument("--regularize_representation", type=str2bool, default=False)
        parser.add_argument("--distillation_period", type=int, default=None)
        parser.add_argument("--distillation_n_epochs", type=int, default=100)

        if "qmp" not in args.experiment:
            parser.add_argument("--stagewise", type=str2bool, default=False)


    if "qmp" in args.experiment:
        parser.add_argument("--policy_sampling_freq", type=int, default=1)

    if "uds" in args.experiment:
        parser.add_argument("--sharing_quantile", type=float, default=0.0)
        parser.add_argument("--unsupervised", type=str2bool, default=True)

    args = parser.parse_args()

    assert args.vis_freq == -1 or args.vis_freq % args.evaluation_frequency == 0


    ### Collect environment arguments

    env_args = {}
    env_args.update(
        {
            "include_task_id": args.include_task_id,
        }
    )
    if args.env == "JacoReachMT5-v1":
        parser.set_defaults(n_policies=5,
                            n_epochs=2000,
                            num_evaluation_episodes=50,)

    if args.env == "MazeLarge-10-v0":
        env_args.update(
            {
                "sparse_tasks": args.sparse_tasks,
                "task_name": args.task_name,
            }
        )
        parser.set_defaults(n_policies=10,
                            n_epochs=2000,
                            num_evaluation_episodes=100,
                            batch_size=6000,
                            min_buffer_size=3000,
                            steps_per_epoch=1,
                            gradient_steps_per_itr=1000,
                            buffer_batch_size=256,)

    if args.env == "Walker2dMT4-v0":
        parser.set_defaults(n_policies=4,
                            batch_size=4000,
                            gradient_steps_per_itr=6000,
                            n_epochs=2000,
                            num_evaluation_episodes=40,)

    if args.env.startswith("MetaWorld"):
        env_args.update(
            {
                "sparse_tasks": args.sparse_tasks,
                "task_name": args.task_name,
            }
        )

    if args.env == "MetaWorldCDS-v1":
        parser.set_defaults(n_policies=4,
                            batch_size=2000,
                            num_evaluation_episodes=40,
                            steps_per_epoch=10,
                            min_buffer_size=5000,
                            buffer_batch_size=256,
                            lr=0.0015,
                            gradient_steps_per_itr=200,
                            n_epochs=500,)

    if args.env == "MetaWorldMT10-v2":
        parser.set_defaults(n_policies=10,
                            batch_size=5000,
                            num_evaluation_episodes=100,
                            steps_per_epoch=10,
                            min_buffer_size=5000,
                            buffer_batch_size=2560,
                            lr=0.0015,
                            gradient_steps_per_itr=500,
                            )

    if args.env == "MetaWorldMT50-v2":
        parser.set_defaults(n_policies=50,
                            batch_size=25000,
                            num_evaluation_episodes=250,
                            hidden_sizes=[400, 400],
                            gradient_steps_per_itr=2500,
                            min_buffer_size=1500,
                            buffer_batch_size=1280,
                            steps_per_epoch=8,
                            n_epochs=1000,)

    if args.env == "KitchenMTEasy-v0":
        env_args.update(
            {
                "sparse_tasks": args.sparse_tasks,
                "task_name": args.task_name,
                "control_penalty": args.control_penalty,
                "vectorized_skills": False,
            }
        )
        parser.set_defaults(n_policies=10,
                            batch_size=2000,
                            buffer_batch_size=1280,
                            gradient_steps_per_itr=500,
                            min_buffer_size=2000,
                            steps_per_epoch=5,
                            num_evaluation_episodes=50,
                            num_epochs=500,)


    args = parser.parse_args()

    ### Run Experiment

    if args.experiment in ["sac", "mtsac"]:
        run_sac_test(
                config=args,
                env_args=env_args,
        )

    elif args.experiment == "dnc_sac":
        run_dnc_sac_test(
            config=args,
            env_args=env_args,
        )

    elif args.experiment == "qmp_dnc":
        run_qmp_dnc_test(
            config=args,
            env_args=env_args,
        )
    elif args.experiment == "qmp_sac":
        run_qmp_sac_test(
            config=args,
            env_args=env_args,
        )

    elif args.experiment == "uds_dnc":
        run_uds_test(
            config=args,
            env_args=env_args,
        )

    elif args.experiment == "qmp_uds_dnc":
        run_qmp_uds_test(
            config=args,
            env_args=env_args,
        )
    elif args.experiment == "decaf":
        run_decaf_test(
            config=args,
            env_args=env_args,
        )
    elif args.experiment == "mcal":
        run_mcal_test(
            config=args,
            env_args=env_args,
        )
    elif args.experiment == "qmp_mcal":
        run_qmp_mcal_test(
            config=args,
            env_args=env_args,
        )
    elif args.experiment == "resume":
        run_resume_test(config=args)


    else:
        warnings.warn(f"Calling unmaintained method: {args.experiment}")

        agent_args = {
            "gradient_steps_per_itr": args.gradient_steps_per_itr,
            "min_buffer_size": args.min_buffer_size,
            "target_update_tau": args.target_update_tau,
            "discount": args.discount,
            "buffer_batch_size": args.buffer_batch_size,
            "reward_scale": args.reward_scale,
            "steps_per_epoch": args.steps_per_epoch,
        }

        raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv)

