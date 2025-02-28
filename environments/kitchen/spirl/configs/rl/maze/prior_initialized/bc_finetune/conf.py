from environments.kitchen.spirl.configs.rl.maze.prior_initialized.base_conf import *
from environments.kitchen.spirl.rl.policies.prior_policies import (
    ACPriorInitializedPolicy,
)
from environments.kitchen.spirl.data.maze.src.maze_agents import MazeSACAgent

agent_config.policy = ACPriorInitializedPolicy
configuration.agent = MazeSACAgent
