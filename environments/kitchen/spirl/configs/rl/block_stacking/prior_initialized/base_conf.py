from environments.kitchen.spirl.configs.rl.block_stacking.base_conf import *
from environments.kitchen.spirl.rl.components.sampler import ACMultiImageAugmentedSampler
from environments.kitchen.spirl.rl.policies.mlp_policies import ConvPolicy
from environments.kitchen.spirl.rl.components.critic import SplitObsMLPCritic
from environments.kitchen.spirl.models.bc_mdl import ImageBCMdl


# update sampler
configuration["sampler"] = ACMultiImageAugmentedSampler
sampler_config = AttrDict(
    n_frames=2,
)

# update policy to conv policy
agent_config.policy = ConvPolicy
policy_params.update(
    AttrDict(
        input_nc=3 * sampler_config.n_frames,
        prior_model=ImageBCMdl,
        prior_model_params=AttrDict(
            state_dim=data_spec.state_dim,
            action_dim=data_spec.n_actions,
            input_res=data_spec.res,
            n_input_frames=2,
        ),
        prior_model_checkpoint=os.path.join(
            os.environ["EXP_DIR"], "skill_prior_learning/block_stacking/flat"
        ),
    )
)

# update critic+policy to handle multi-frame combined observation
agent_config.critic = SplitObsMLPCritic
agent_config.critic_params.unused_obs_size = 32 ** 2 * 3 * sampler_config.n_frames
