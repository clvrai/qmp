import os
import copy

from environments.kitchen.spirl.utils.general_utils import AttrDict
from environments.kitchen.spirl.rl.components.agent import (
    FixedIntervalHierarchicalAgent,
)
from environments.kitchen.spirl.rl.policies.mlp_policies import SplitObsMLPPolicy
from environments.kitchen.spirl.rl.components.critic import SplitObsMLPCritic
from environments.kitchen.spirl.rl.envs.maze import ACRandMaze0S40Env
from environments.kitchen.spirl.rl.components.sampler import (
    ACMultiImageAugmentedHierarchicalSampler,
)
from environments.kitchen.spirl.rl.components.replay_buffer import UniformReplayBuffer
from environments.kitchen.spirl.rl.agents.ac_agent import SACAgent
from environments.kitchen.spirl.models.skill_prior_mdl import ImageSkillPriorMdl
from environments.kitchen.spirl.configs.default_data_configs.maze import data_spec
from environments.kitchen.spirl.data.maze.src.maze_agents import MazeACSkillSpaceAgent


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = "hierarchical RL on the maze env"

configuration = {
    "seed": 42,
    "agent": FixedIntervalHierarchicalAgent,
    "environment": ACRandMaze0S40Env,
    "sampler": ACMultiImageAugmentedHierarchicalSampler,
    "data_dir": ".",
    "num_epochs": 30,
    "max_rollout_len": 2000,
    "n_steps_per_epoch": 100000,
    "n_warmup_steps": 5e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict()

# Observation Normalization
obs_norm_params = AttrDict()

sampler_config = AttrDict(
    n_frames=2,
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=1e-2,
    n_input_frames=2,
    prior_input_res=data_spec.res,
    nz_vae=10,
    n_rollout_steps=10,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(
    AttrDict(
        model=ImageSkillPriorMdl,
        model_params=ll_model_params,
        model_checkpoint=os.path.join(
            os.environ["EXP_DIR"], "skill_prior_learning/maze/hierarchical"
        ),
    )
)


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=10,  # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.0,  # prior is Gaussian with unit variance
    unused_obs_size=ll_model_params.prior_input_res ** 2
    * 3
    * ll_model_params.n_input_frames,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,  # number of policy network layers
    nz_mid=256,
    action_input=True,
    unused_obs_size=hl_policy_params.unused_obs_size,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(
    AttrDict(
        policy=SplitObsMLPPolicy,
        policy_params=hl_policy_params,
        critic=SplitObsMLPCritic,
        critic_params=hl_critic_params,
    )
)


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=MazeACSkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=False,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.0,
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
)

# reduce replay capacity because we are training image-based, do not dump (too large)
from environments.kitchen.spirl.rl.components.replay_buffer import (
    SplitObsUniformReplayBuffer,
)

agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = (
    ll_model_params.prior_input_res ** 2 * 3 * 2
    + hl_agent_config.policy_params.action_dim
)  # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False
