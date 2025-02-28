import os
import copy

from environments.kitchen.spirl.utils.general_utils import AttrDict
from environments.kitchen.spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from environments.kitchen.spirl.rl.components.critic import SplitObsMLPCritic
from environments.kitchen.spirl.rl.components.sampler import ACMultiImageAugmentedHierarchicalSampler
from environments.kitchen.spirl.rl.components.replay_buffer import UniformReplayBuffer
from environments.kitchen.spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from environments.kitchen.spirl.rl.envs.block_stacking import HighStack11StackEnvV0, SparseHighStack11StackEnvV0
from environments.kitchen.spirl.rl.agents.ac_agent import SACAgent
from environments.kitchen.spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from environments.kitchen.spirl.rl.policies.cl_model_policies import ACClModelPolicy
from environments.kitchen.spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from environments.kitchen.spirl.configs.default_data_configs.block_stacking import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = "used to test the RL implementation"

configuration = {
    "seed": 42,
    "agent": FixedIntervalHierarchicalAgent,
    "environment": SparseHighStack11StackEnvV0,
    "sampler": ACMultiImageAugmentedHierarchicalSampler,
    "data_dir": ".",
    "num_epochs": 100,
    "max_rollout_len": 1000,
    "n_steps_per_epoch": 1e5,
    "n_warmup_steps": 5e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict()

sampler_config = AttrDict(
    n_frames=2,
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy Model
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    prior_input_res=data_spec.res,
    n_input_frames=2,
    cond_decode=True,
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ImageClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(
        os.environ["EXP_DIR"], "skill_prior_learning/block_stacking/hierarchical_cl"
    ),
    initial_log_sigma=-50.0,
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    output_dim=1,
    action_input=True,
    unused_obs_size=10,  # ignore HL policy z output in observation for LL critic
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(
    AttrDict(
        policy=ACClModelPolicy,
        policy_params=ll_policy_params,
        critic=SplitObsMLPCritic,
        critic_params=ll_critic_params,
    )
)


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=10,  # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.0,  # prior is Gaussian with unit variance
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,  # number of policy network layers
    nz_mid=256,
    action_input=True,
    unused_obs_size=ll_model_params.prior_input_res ** 2
    * 3
    * ll_model_params.n_input_frames,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(
    AttrDict(
        policy=ACLearnedPriorAugmentedPIPolicy,
        policy_params=hl_policy_params,
        critic=SplitObsMLPCritic,
        critic_params=hl_critic_params,
        td_schedule_params=AttrDict(p=5.0),
    )
)


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=True,
    update_hl=True,
    update_ll=False,
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
