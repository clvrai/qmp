from gym.envs.mujoco.mujoco_env import MujocoEnv

# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly

# Base class
from environments.mujoco.base_env import BaseEnv

# Walker2d
from environments.mujoco.walker2d import Walker2dEnv
from environments.mujoco.walker2d_forward import Walker2dForwardEnv
from environments.mujoco.walker2d_backward import Walker2dBackwardEnv
from environments.mujoco.walker2d_balance import Walker2dBalanceEnv
from environments.mujoco.walker2d_jump import Walker2dJumpEnv
from environments.mujoco.walker2d_crawl import Walker2dCrawlEnv
from environments.mujoco.walker2d_patrol import Walker2dPatrolEnv
from environments.mujoco.walker2d_hurdle import Walker2dHurdleEnv
from environments.mujoco.walker2d_obstacle_course import Walker2dObstacleCourseEnv

# Jaco
from environments.mujoco.jaco import JacoEnv
from environments.mujoco.jaco_reach import JacoReachEnv
from environments.mujoco.jaco_reach_multistage import (
    JacoReachMultistageEnv,
    JacoReachMultistageMTEnv,
    JacoReachMT5Env,
    JacoReachMT5StochasticEnv,
    JacoReachMT3Env,
)
from environments.mujoco.jaco_pick import JacoPickEnv
from environments.mujoco.jaco_catch import JacoCatchEnv
from environments.mujoco.jaco_toss import JacoTossEnv
from environments.mujoco.jaco_hit import JacoHitEnv
from environments.mujoco.jaco_keep_pick import JacoKeepPickEnv
from environments.mujoco.jaco_keep_catch import JacoKeepCatchEnv
from environments.mujoco.jaco_serve import JacoServeEnv
