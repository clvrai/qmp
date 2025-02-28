from environments.kitchen.spirl.configs.rl.maze.prior_initialized.base_conf import *
from environments.kitchen.spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from environments.kitchen.spirl.data.maze.src.maze_agents import MazeActionPriorSACAgent

agent_config.update(
    AttrDict(
        td_schedule_params=AttrDict(p=1.0),
    )
)

agent_config.policy = ACLearnedPriorAugmentedPIPolicy
configuration.agent = MazeActionPriorSACAgent
