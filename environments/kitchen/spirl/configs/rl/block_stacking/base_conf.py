import os

from environments.kitchen.spirl.utils.general_utils import AttrDict
from environments.kitchen.spirl.rl.agents.ac_agent import SACAgent
from environments.kitchen.spirl.rl.policies.mlp_policies import MLPPolicy
from environments.kitchen.spirl.rl.components.critic import MLPCritic
from environments.kitchen.spirl.rl.components.replay_buffer import UniformReplayBuffer
from environments.kitchen.spirl.rl.envs.block_stacking import HighStack11StackEnvV0
from environments.kitchen.spirl.rl.components.normalization import Normalizer
from environments.kitchen.spirl.configs.default_data_configs.block_stacking import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = "non-hierarchical RL experiments in block stacking env"

configuration = {
    "seed": 42,
    "agent": SACAgent,
    "environment": HighStack11StackEnvV0,
    "data_dir": ".",
    "num_epochs": 100,
    "max_rollout_len": 1000,
    "n_steps_per_epoch": 100000,
    "n_warmup_steps": 5e3,
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    n_layers=5,  #  number of policy network layers
    nz_mid=256,
    max_action_range=1.0,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
    output_dim=1,
    n_layers=2,  #  number of policy network layers
    nz_mid=256,
    action_input=True,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict()

# Agent
agent_config = AttrDict(
    policy=MLPPolicy,
    policy_params=policy_params,
    critic=MLPCritic,
    critic_params=critic_params,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
    batch_size=256,
    log_video_caption=True,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    name="block_stacking",
    reward_norm=1.0,
    screen_width=data_spec.res,
    screen_height=data_spec.res,
    env_config=AttrDict(
        camera_name="agentview",
        screen_width=data_spec.res,
        screen_height=data_spec.res,
    ),
)
