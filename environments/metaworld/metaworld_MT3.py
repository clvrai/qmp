from collections import OrderedDict

from metaworld import Benchmark, _make_tasks, _MT_OVERRIDE
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerReachEnvV2,  #
    SawyerPickPlaceEnvV2,  #
    SawyerPushEnvV2,  #
)

### MT3 3 hardest tasks from MT10

MT3_V2_dict = OrderedDict(
    (
        ("reach-v2", SawyerReachEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
    ),
)

MT3_V2_ARGS_KWARGS = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT3_V2_dict.items()
}


class MT3(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = MT3_V2_dict
        self._test_classes = OrderedDict()
        train_kwargs = MT3_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )
        self._test_tasks = []
