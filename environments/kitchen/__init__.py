from collections import OrderedDict
from functools import partial

### D4RL kitchen single tasks and multistage tasks
from environments.kitchen.v0.kitchen_tasks import (
    KitchenMicrowaveKettleBottomBurnerV0,
    KitchenKettleLightHingeCabinetV0,
    KitchenKettleBottomBurnerHingeCabinetV0,
    KitchenBottomBurnerHingeCabinetMicrowaveV0,
    KitchenMicrowaveKettleBottomBurnerV2,
    KitchenKettleBottomBurnerHingeCabinetV2,
    KitchenBottomBurnerHingeCabinetMicrowaveV2,
    KitchenKettleHingeCabinetBottomBurnerV2,
    KitchenMicrowaveHingeCabinetBottomBurnerV2,
    KitchenMicrowaveKettleSlideCabinetV2,
    KitchenKettleSlideCabinetHingeCabinetV2,
    KitchenKettleBottomBurnerSlideCabinetV2,
    KitchenMicrowaveBottomBurnerLightV2,
    KitchenBottomBurnerLightHingeCabinetV2,
    KitchenMicrowaveLightSlideCabinetV2,
    KitchenMicrowaveLightHingeCabinetV2,
)

from environments.kitchen.v0.kitchen_v0 import (
    KitchenBottomBurnerOnEnvV0,
    KitchenTopBurnerOnEnvV0,
    KitchenLightSwitchOnEnvV0,
    KitchenSlideCabinetOpenEnvV0,
    KitchenHingeCabinetOpenEnvV0,
    KitchenMicrowaveOpenEnvV0,
    KitchenKettlePushEnvV0,
    KitchenBottomBurnerOffEnvV0,
    KitchenTopBurnerOffEnvV0,
    KitchenLightSwitchOffEnvV0,
    KitchenSlideCabinetCloseEnvV0,
    KitchenHingeCabinetCloseEnvV0,
    KitchenMicrowaveCloseEnvV0,
    KitchenKettlePullEnvV0,
)

### V1
from environments.kitchen.v1.kitchen_slide_cabinet_open import (
    KitchenSlideCabinetOpenEnvV1,
)
from environments.kitchen.v1.kitchen_kettle_pull import KitchenKettlePullEnvV1

KITCHEN_ALL = OrderedDict(
    (
        ("bottom burner-on", KitchenBottomBurnerOnEnvV0),
        ("bottom burner-off", KitchenBottomBurnerOffEnvV0),
        ("top burner-on", KitchenTopBurnerOnEnvV0),
        ("top burner-off", KitchenTopBurnerOffEnvV0),
        ("light switch-on", KitchenLightSwitchOnEnvV0),
        ("light switch-off", KitchenLightSwitchOffEnvV0),
        ("slide cabinet-open", KitchenSlideCabinetOpenEnvV0),  ###
        ("slide cabinet-open-v1", KitchenSlideCabinetOpenEnvV1),
        ("slide cabinet-close", KitchenSlideCabinetCloseEnvV0),  ###
        ("hinge cabinet-open", KitchenHingeCabinetOpenEnvV0),  ###
        ("hinge cabinet-close", KitchenHingeCabinetCloseEnvV0),  ###
        ("microwave-open", KitchenMicrowaveOpenEnvV0),
        ("microwave-close", KitchenMicrowaveCloseEnvV0),
        ("kettle-push", KitchenKettlePushEnvV0),
        ("kettle-pull", KitchenKettlePullEnvV0),  ###
        ("kettle-pull-v1", KitchenKettlePullEnvV0),  ###
    )
)

KITCHEN_MT_EASY = OrderedDict(
    (
        (env_name, KITCHEN_ALL[env_name])
        for env_name in [
            "slide cabinet-open",
            "slide cabinet-close",
            "hinge cabinet-open",
            "hinge cabinet-close",
            "bottom burner-on",
            "bottom burner-off",
            "top burner-on",
            "top burner-off",
            "light switch-on",
            "light switch-off"

        ]
    )
)

KITCHEN_MT_EASY5 = OrderedDict(
    (
        (env_name, KITCHEN_ALL[env_name])
        for env_name in [
            "slide cabinet-open",
            "bottom burner-off",
            "top burner-off",
            "light switch-on",
            "light switch-off"

        ]
    )
)

KITCHEN_MT10 = OrderedDict(
    (
        (env_name, KITCHEN_ALL[env_name])
        for env_name in [
            "bottom burner-on",
            "bottom burner-off",
            "top burner-on",
            "top burner-off",
            "light switch-on",
            "hinge cabinet-open",
            "hinge cabinet-close",
            "microwave-open",
            "microwave-close",
            "kettle-push",
        ]
    )
)

KITCHEN_MT3 = OrderedDict(
    (
        ("microwave-kettle-bottom-burner", KitchenMicrowaveKettleBottomBurnerV0),
        ("kettle-bottom-burner-hinge-cabinet", KitchenKettleBottomBurnerHingeCabinetV0),
        (
            "bottom-burner-hinge-cabinet-microwave",
            KitchenBottomBurnerHingeCabinetMicrowaveV0,
        ),
    )
)

KITCHEN_MT3_v2 = OrderedDict(
    (
        (
            "microwave-kettle-bottom-burner",
            partial(
                KitchenMicrowaveKettleBottomBurnerV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=True,
            ),
        ),
        (
            "kettle-bottom-burner-hinge-cabinet",
            partial(
                KitchenKettleBottomBurnerHingeCabinetV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=True,
            ),
        ),
        (
            "bottom-burner-hinge-cabinet-microwave",
            partial(
                KitchenBottomBurnerHingeCabinetMicrowaveV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=True,
            ),
        ),
    )
)

KITCHEN_MT3_v3 = OrderedDict(
    (
        (
            "microwave-kettle-bottom-burner",
            partial(
                KitchenMicrowaveKettleBottomBurnerV2,
                negative_reward=True,
                reward_scale=1.0,
                sparse_stage_reward=False,
            ),
        ),
        (
            "kettle-bottom-burner-hinge-cabinet",
            partial(
                KitchenKettleBottomBurnerHingeCabinetV2,
                negative_reward=True,
                reward_scale=1.0,
                sparse_stage_reward=False,
            ),
        ),
        (
            "bottom-burner-hinge-cabinet-microwave",
            partial(
                KitchenBottomBurnerHingeCabinetMicrowaveV2,
                negative_reward=True,
                reward_scale=1.0,
                sparse_stage_reward=False,
            ),
        ),
    )
)

KITCHEN_HARD = OrderedDict(
    (
        (
            "microwave-kettle-bottom-burner",
            partial(
                KitchenMicrowaveKettleBottomBurnerV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                enforce_task_order=True,
            ),
        ),
        (
            "bottom-burner-hinge-cabinet-microwave",
            partial(
                KitchenBottomBurnerHingeCabinetMicrowaveV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                enforce_task_order=True,
            ),
        ),
        (
            "kettle-hinge-cabinet-bottom-burner",
            partial(
                KitchenKettleHingeCabinetBottomBurnerV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                enforce_task_order=True,
            ),
        ),
    )
)

KITCHEN_MT5 = OrderedDict(
    (
        (
            "microwave-kettle-bottom-burner",
            partial(
                KitchenMicrowaveKettleBottomBurnerV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                # enforce_task_order=True,
            ),
        ),
        (
            "kettle-bottom-burner-hinge-cabinet",
            partial(
                KitchenKettleBottomBurnerHingeCabinetV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                # enforce_task_order=True,
            ),
        ),
        (
            "bottom-burner-hinge-cabinet-microwave",
            partial(
                KitchenBottomBurnerHingeCabinetMicrowaveV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                # enforce_task_order=True,
            ),
        ),
        (
            "kettle-hinge-cabinet-bottom-burner",
            partial(
                KitchenKettleHingeCabinetBottomBurnerV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                # enforce_task_order=True,
            ),
        ),
        (
            "microwave-hinge-cabinet-bottom-burner",
            partial(
                KitchenMicrowaveHingeCabinetBottomBurnerV2,
                negative_reward=True,
                reward_scale=0.03,
                sparse_stage_reward=False,
                # enforce_task_order=True,
            ),
        ),
    )
)

KITCHEN_MT4 = OrderedDict(
    (
        (
            "microwave-kettle-slide-cabinet",
            partial(
                KitchenMicrowaveKettleSlideCabinetV2,
                negative_reward=True,
                reward_scale=0.003,
                sparse_stage_reward=True,
                enforce_task_order=True,
            ),
        ),
        (
            "kettle-slide-cabinet-hinge-cabinet",
            partial(
                KitchenKettleSlideCabinetHingeCabinetV2,
                negative_reward=True,
                reward_scale=0.003,
                sparse_stage_reward=True,
                enforce_task_order=True,
            ),
        ),
        (
            "bottom-burner-light-hinge-cabinet",
            partial(
                KitchenBottomBurnerLightHingeCabinetV2,
                negative_reward=True,
                reward_scale=0.003,
                sparse_stage_reward=True,
                enforce_task_order=True,
            ),
        ),
        (
            "microwave-bottom-burner-light-switch",
            partial(
                KitchenMicrowaveBottomBurnerLightV2,
                negative_reward=True,
                reward_scale=0.003,
                sparse_stage_reward=True,
                enforce_task_order=True,
            ),
        ),
    )
)

KITCHEN_MT2 = OrderedDict(
    (
        (
            "microwave-light-slide-cabinet",
            partial(
                KitchenMicrowaveLightSlideCabinetV2,
                negative_reward=True,
                reward_scale=0.003,
                sparse_stage_reward=True,
                enforce_task_order=True,
            ),
        ),
        (
            "microwave-light-hinge-cabinet",
            partial(
                KitchenMicrowaveLightHingeCabinetV2,
                negative_reward=True,
                reward_scale=0.003,
                sparse_stage_reward=True,
                enforce_task_order=True,
            ),
        ),
    )
)

KITCHEN_MT1 = OrderedDict(
    (
        (env_name, OrderedDict(((env_name, KITCHEN_ALL[env_name]),)))
        for env_name in KITCHEN_ALL.keys()
    )
)

KITCHEN_CABINET = OrderedDict(
    (
        (env_name, KITCHEN_ALL[env_name])
        for env_name in [
            "slide cabinet-open",
            "slide cabinet-close",
            "hinge cabinet-open",
            "hinge cabinet-close",
        ]
    )
)

KITCHEN_MICROWAVE = OrderedDict(
    (
        (env_name, KITCHEN_ALL[env_name])
        for env_name in [
            "microwave-open",
            "microwave-close",
        ]
    )
)
