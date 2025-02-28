#!/usr/bin/env python3
"""
runs garage policy from snapshot
"""
import numpy as np
import os
import cv2
import argparse
import sys

from garage.experiment import Snapshotter
from garage.torch import set_gpu_mode
from garage.experiment.deterministic import set_seed

from learning.utils import rollout
from learning.utils.visualizer import Visualizer

# import environments


def get_device_from_module(module):
    while True:
        named_modules = list(module.named_modules())
        if len(named_modules) == 1:
            break
        module = named_modules[1][1]
    if hasattr(module, "weight"):
        return module.weight.device
    else:
        raise NotImplementedError


def run_policy(
    snapshot_dir,
    num_episodes_per_env=1,
    render=False,
    save_video=False,
    video_root=None,
    video_hide_info=False,
    video_fps=40,
    seed=None,
):
    if seed is not None:
        set_seed(seed)

    snapshotter = Snapshotter()
    data = snapshotter.load(snapshot_dir)
    policy = data["algo"].policy
    envs = data["env"]

    # set device
    if hasattr(policy, "policies"):
        device = get_device_from_module(policy.policies[0])
    else:
        device = get_device_from_module(policy)

    print("device:", device)
    if device.type == "cuda":
        set_gpu_mode(True, device.index)

    # set up visualizer
    if save_video:
        assert video_root is not None
        video_dir = os.path.join(
            video_root,
            os.path.basename(os.path.normpath(snapshot_dir)),
        )
        os.makedirs(video_dir, exist_ok=True)
        visualizer = Visualizer(
            imsize=(500, 500),
            hide_info=video_hide_info,
        )
    else:
        visualizer = None
        video_dir = None

    # See what the trained policy can accomplish

    if not isinstance(envs, list):
        envs = [envs]

    for env_idx, env in enumerate(envs):
        print("#### ENV {} ####".format(env_idx))
        rets = []
        all_frames = []
        for i in range(num_episodes_per_env):
            o = env.reset()
            print(o)
            visualizer.reset()
            path = rollout(
                env, policy, animated=render, pause_per_frame=0.01,
                save_video=save_video, visualizer=visualizer,
            )
            # print("Rewards: ", path["rewards"])
            ret = np.sum(path["rewards"])
            print("Episode {}: Len: {}, Return: {}".format(i, len(path["rewards"]), ret))
            rets.append(ret)

            if save_video:
                if num_episodes_per_env > 1:  # save individual episode
                    video_name = os.path.join(video_dir, "ep_{}_{}.avi".format(env_idx, i))
                    save_video_to_file(video_name, visualizer._frames, video_fps)
                all_frames.extend(visualizer._frames)

        if save_video:
            video_name = os.path.join(video_dir, "all_episodes_{}.avi".format(env_idx))
            save_video_to_file(video_name, all_frames, video_fps)

        print("Average Return: {}".format(np.mean(rets)))

        break


def save_video_to_file(filename, frames, fps):
    size = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


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
    parser.add_argument("--snapshot_dir", type=str, default='data/local/experiment/dnc_sac_MazeLarge-v0_sparse_beta005_0/')
    parser.add_argument("--num_episodes_per_env", type=int, default=1)
    parser.add_argument("--render", type=str2bool, default=False)
    parser.add_argument("--save_video", type=str2bool, default=True)
    parser.add_argument("--video_root", type=str, default="videos")
    parser.add_argument("--video_hide_info", type=str2bool, default=True)
    parser.add_argument("--video_fps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_policy(**vars(args))


if __name__ == "__main__":
    main(sys.argv)
