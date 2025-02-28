from akro import Discrete
from garage import EnvSpec, Wrapper


class DiscreteWrapper(Wrapper):
    def __init__(self, env, n_actions, disc2cont):
        super().__init__(env)
        self._n_actions = n_actions
        self._disc2cont = disc2cont
        self._action_space = Discrete(n_actions)
        self._spec = EnvSpec(
            observation_space=env.spec.observation_space,
            action_space = self._action_space,
            max_episode_length=env.spec.max_episode_length,
        )
        if hasattr(env, "get_train_envs"):
            setattr(self, "get_train_envs", self._get_train_envs)
        if hasattr(env, "get_test_envs"):
            setattr(self, "get_test_envs", self._get_test_envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def spec(self):
        return self._spec

    def step(self, action):
        return self._env.step(self._disc2cont(action))

    def _get_train_envs(self):
        return [
            DiscreteWrapper(env, self._n_actions, self._disc2cont)
            for env in self._env.get_train_envs()
        ]

    def _get_test_envs(self):
        return [
            DiscreteWrapper(env, self._n_actions, self._disc2cont)
            for env in self._env.get_test_envs()
        ]
