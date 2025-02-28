from environments.kitchen.spirl.configs.rl.kitchen.prior_initialized.base_conf import *
from environments.kitchen.spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from environments.kitchen.spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

agent_config.update(
    AttrDict(
        td_schedule_params=AttrDict(p=1.0),
    )
)

agent_config.policy = LearnedPriorAugmentedPIPolicy
configuration.agent = ActionPriorSACAgent
