from gym.envs.registration import register
import numpy as np

from .kitchen.kitchen_multitask import (
    KitchenMultiTaskEnv,
)
from .maze import MazeEnv
from environments.metaworld.metaworld_gym import MetaWorldEnv

from .reacher_multistage import ReacherMultistageMTEnv
from environments.mujoco.walker2d_multitask import Walker2dMTEnv

### Walker


register(
    id="Walker2dMT4-v0",
    max_episode_steps=1000,
    entry_point="environments:Walker2dMTEnv",
)


# Kitchen
# ----------------------------------------

register(
    id="KitchenMTEasy-v0",
    entry_point="environments:KitchenMultiTaskEnv",
    max_episode_steps=140,
    kwargs={"benchmark_type": "MT_EASY", "max_episode_steps": 140},
)


# Meta-World
# ----------------------------------------
register(
    id="MetaWorldCDS-v1",
    entry_point="environments:MetaWorldEnv",
    max_episode_steps=500,
    kwargs={"benchmark_type": "CDS_v1"},
)

register(
    id="MetaWorldMT10-v2",
    entry_point="environments:MetaWorldEnv",
    max_episode_steps=500,
    kwargs={"benchmark_type": "MT10"},
)

register(
    id="MetaWorldMT50-v2",
    entry_point="environments:MetaWorldEnv",
    max_episode_steps=500,
    kwargs={"benchmark_type": "MT50"},
)



# Jaco
# ----------------------------------------

register(
    id="JacoReachMT5-v1",
    entry_point="environments.mujoco:JacoReachMT5Env",
    max_episode_steps=200,
)

# Maze
# ----------------------------------------


register(
    id="MazeLarge-10-v0",
    entry_point="environments.maze:MazeMultitaskEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": "LARGE_MAZE",
        "num_tasks": 10,
        "max_episode_steps": 600,
    },
)


