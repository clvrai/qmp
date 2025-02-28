import numpy as np

from environments.kitchen import reward_utils
from learning.utils.general import SuppressStdout
from d4rl.kitchen.kitchen_envs import KitchenBase
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS


OBS_ELEMENT_SITES = {
    "bottom burner": "knob2_site",
    "top burner": "knob4_site",
    "light switch": "light_site",
    "slide cabinet": "slide_site",
    "hinge cabinet": "hinge_site2",
    "microwave": "microhandle_site",
    "kettle": "kettle_site",
}

OBS_ELEMENT_INITS = {
    "bottom burner": np.array([3.12877220e-05, -4.51199853e-05]),
    "top burner": np.array([6.28065475e-05, 4.04984708e-05]),
    "light switch": np.array([4.62730939e-04, -2.26906415e-04]),
    "slide cabinet": np.array([-4.65501369e-04]),
    "hinge cabinet": np.array([-6.44129196e-03, -1.77048263e-03]),
    "microwave": np.array([1.08009684e-03]),
    "kettle": np.array(
        [
            -2.69397440e-01,
            3.50383255e-01,
            1.61944683e00,
            1.00618764e00,
            4.06395120e-03,
            -6.62095997e-03,
            -2.68278933e-04,
        ]
    ),
}


class KitchenInfosBase(KitchenBase):
    """
    KitchenBase compatible with KitchenMultiTaskEnv
    - takes in extra arguments
    - records success rate and stages completed information
    """

    BONUS_THRESH = {
        "bottom burner": 0.3,
        "top burner": 0.3,
        "light switch": 0.3,
        "slide cabinet": 0.1,
        "hinge cabinet": 0.1, # close is 0.3
        "microwave": 0.1,
        "kettle": 0.1,
    }

    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    ENFORCE_TASK_ORDER = True

    def __init__(
        self,
        sparse_reward=False,
        terminate_on_success=True,
        control_penalty=0,
        early_termination_bonus=0,
        **kwargs,
    ):
        assert terminate_on_success and control_penalty == 0
        self._sparse_reward = sparse_reward
        self._dist_goal_init = 0.0
        self._dist_hand_init = 0.0
        self._reward_scale = 1.0
        self._num_stages = len(self.TASK_ELEMENTS)
        self._early_termination_bonus = early_termination_bonus * self._num_stages

        with SuppressStdout():
            super().__init__(**kwargs)

        self.tasks_to_complete = list(self.TASK_ELEMENTS)

    @property
    def _current_task(self):
        if isinstance(self.tasks_to_complete, set):
            print("Task Elements: ", self.TASK_ELEMENTS)
            return self.TASK_ELEMENTS[0]
        return (
            self.TASK_ELEMENTS[-1]
            if len(self.tasks_to_complete) == 0
            else self.tasks_to_complete[0]
        )

    @property
    def _element(self):
        return self._current_task

    @property
    def _element_idx(self):
        return OBS_ELEMENT_INDICES[self._current_task]

    @property
    def _element_goal(self):
        return OBS_ELEMENT_GOALS[self._current_task]

    @property
    def _num_stages_completed(self):
        return self._num_stages - len(self.tasks_to_complete)

    def reset_model(self):
        ret = super().reset_model()
        ### TO DO: recalculate when advancing to a new stage?
        self._record_init_dists()
        return ret

    def _record_init_dists(self):
        self._dist_goal_init = np.linalg.norm(
            OBS_ELEMENT_INITS[self._element] - self._element_goal
        )
        hand_to_goal = self._get_obj_pos() - self._get_hand_pos()
        self._dist_hand_init = np.linalg.norm(hand_to_goal)

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super()._get_reward_n_score(obs_dict)

        ### Record distances
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        next_element = next_obj_obs[..., self._element_idx - idx_offset]
        gripper_pos = self._get_gripper_pos()
        obj_pos = self._get_obj_pos()
        dists = {
            "goal": np.linalg.norm(next_element - self._element_goal),
            "goal_init": self._dist_goal_init,
            "hand": np.linalg.norm(
                obj_pos
                # - self._get_hand_pos()
                - gripper_pos
            ),
            "hand_init": self._dist_hand_init,
        }
        reward_dict["dist_goal"] = dists["goal"]
        reward_dict["dist_hand"] = dists["hand"]

        ### Calculate dense reward
        if not self._sparse_reward:
            reward, stage_success = self._compute_reward(dists)
            bonus = float(stage_success)
            if self.TERMINATE_ON_TASK_COMPLETE and (len(self.tasks_to_complete) == 0):
                reward = self._early_termination_bonus
            ### Stage reward:
            reward += self._num_stages_completed
            reward *= self._reward_scale
            reward_dict["bonus"] = bonus
            reward_dict["r_total"] = reward
            score = bonus

        ### Update init distances if advancing to a new stage
        if (self._sparse_reward and score) or (
            not self._sparse_reward and stage_success
        ):
            self._record_init_dists()

        return reward_dict, score

    def step(self, action):
        obs, reward, done, env_info = super().step(action)

        ### Record success rate and distances
        success = len(self.tasks_to_complete) == 0
        env_info["success"] = success
        ### self._num_stages_completed = self._num_stages - len(self.tasks_to_complete)
        env_info["stages_completed"] = self._num_stages_completed
        env_info["timeout"] = False
        env_info["VIS:dist_goal"] = env_info["rewards"]["dist_goal"]
        env_info["VIS:dist_hand"] = env_info["rewards"]["dist_hand"]

        if self.initializing:
            done = False

        # remove dictionary from env_info since garage doesn't support it
        del env_info["obs_dict"]
        del env_info["rewards"]
        return obs, reward, done, env_info

    ### to not get warnings on lim-d
    def _get_obs(self):
        obs = super()._get_obs()
        return obs.astype(np.float32)

    @classmethod
    def aggregate_infos(cls, infos, info):
        for k, v in info.items():
            if k == "success" or k == "timeout":
                infos[k] = int(infos[k] or v)
            elif "dist" in k or k == "time" or "stages_completed" in k:
                infos[k] = v
            elif "score" in k:
                infos[k] += v
        return infos

    def _compute_reward(self, dists):
        in_place = reward_utils.tolerance(
            dists["goal"],
            bounds=(0, self.BONUS_THRESH[self._current_task]),
            margin=abs(dists["goal_init"] - self.BONUS_THRESH[self._current_task]),
            sigmoid="gaussian", ## "long_tail"
        )

        handle_reach_radius = 0.08
        reach = reward_utils.tolerance(
            dists["hand"],
            bounds=(0, handle_reach_radius),
            margin=abs(dists["hand_init"] - handle_reach_radius),
            sigmoid="gaussian",
        )

        # reward = reward_utils.hamacher_product(reach, in_place)
        reward = 0.4 * in_place + 0.6 * reach

        stage_success = False
        if dists["goal"] < self.BONUS_THRESH[self._current_task]:
            stage_success = True
        return reward, stage_success

    ### Distance calculating helper functions

    def _get_pos(self, name):
        if name in self.model.site_names:
            return self.data.get_site_xpos(name).copy()
        if name in self.model.geom_names:
            return self.data.get_geom_xpos(name).copy()
        if name in self.model.body_names:
            return self.data.get_body_xpos(name).copy()
        raise ValueError("{} not found in the model".format(name))

    def _get_pad_pos(self):
        left_pad, right_pad = self.data.body("panda0_leftfinger").xpos.copy(), self.data.body(
            "panda0_rightfinger"
        ).xpos.copy()
        return left_pad, right_pad

    def _get_gripper_pos(self):
        finger1, finger2 = self._get_pad_pos()
        pos = (finger1 + finger2) / 2
        return pos

    def _get_obj_pos(self):
        return self.data.site(OBS_ELEMENT_SITES[self._element]).xpos.copy()

    def _get_hand_pos(self):
        return self.data.site("end_effector").xpos.copy()





class KitchenBottomBurnerOnEnvV0(KitchenInfosBase):
    TASK_ELEMENTS = ["bottom burner"]

class KitchenTopBurnerOnEnvV0(KitchenInfosBase):
    TASK_ELEMENTS = ["top burner"]

class KitchenLightSwitchOnEnvV0(KitchenInfosBase):
    TASK_ELEMENTS = ["light switch"]

class KitchenSlideCabinetOpenEnvV0(KitchenInfosBase):
    TASK_ELEMENTS = ["slide cabinet"]

class KitchenHingeCabinetOpenEnvV0(KitchenInfosBase):
    TASK_ELEMENTS = ["hinge cabinet"]

class KitchenMicrowaveOpenEnvV0(KitchenInfosBase):
    TASK_ELEMENTS = ["microwave"]

class KitchenKettlePushEnvV0(KitchenInfosBase):
    TASK_ELEMENTS = ["kettle"]





### Task 0 v0-2

class KitchenInfosBasev2(KitchenInfosBase):

    ### wider dist hand init margins,
    # optional sparse stage rewards, spirl like reward scale, negative rewards
    # works for either sparse stage rewards or negative rewards, not dense positive

    def __init__(
        self,
        negative_reward=False,
        reward_scale=0.03,
        sparse_stage_reward=True,
        enforce_task_order=False,
        *args,
        **kwargs,
    ):
        assert self._num_stages == 3  ## why??
        assert negative_reward or sparse_stage_reward
        self._negative_reward = negative_reward
        self._sparse_stage_reward = sparse_stage_reward
        self._enforce_task_order = enforce_task_order
        self._tasks_completed = []
        super().__init__(*args, **kwargs)
        assert self._sparse_reward == False
        self._reward_scale = reward_scale

    def reset_model(self):
        ret = super().reset_model()
        self._tasks_completed = []
        return ret

    def _record_init_dists(self):
        self._dist_goal_init = np.linalg.norm(
            OBS_ELEMENT_INITS[self._element] - self._element_goal
        )

        ### To Do: set min dist_hand_init to 0.5
        hand_to_goal = self._get_obj_pos() - self._get_hand_pos()
        self._dist_hand_init = max(np.linalg.norm(hand_to_goal), 0.5)

    def step(self, action):
        self._wrong_task_completed = False
        obs, reward, done, env_info = super().step(action)

        env_info["wrong_stages_completed"] = (
            len(self._tasks_completed) - self._num_stages_completed
        )

        if self._enforce_task_order and self._wrong_task_completed:
            done = True

        return obs, reward, done, env_info

    def _get_reward_n_score(self, obs_dict):
        ### KitchenBase Base
        # ASDF: what is this used for?
        reward_dict = {"true_reward": 0.0}

        ### KitchenBase
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)

        ### wrong task completion + out of order task completion
        wrong_task_completed = False
        stage_success = False
        all_goal = self._get_task_goal(task=self.ALL_TASKS)

        if self._enforce_task_order:
            # print(
            #     f"stages completed: {self._num_stages_completed}, current task: {self._current_task}, tasks completed: {self._tasks_completed}, remaining tasks: {[task for task in self.tasks_to_complete if task not in self._tasks_completed]}"
            # )
            for wrong_task in list(
                set(self.ALL_TASKS)
                - set([self._current_task])
                - set(self._tasks_completed)
            ):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(
                    next_obj_obs[..., element_idx - idx_offset] - all_goal[element_idx]
                )
                complete = distance < self.BONUS_THRESH[wrong_task]
                if complete:
                    self._tasks_completed.append(wrong_task)  # no double penalty
                    wrong_task_completed = True

        ### To Do: if not remaining_tasks should terminate episode (but weird with negative rewards)
        remaining_tasks = [
            task for task in self.tasks_to_complete if task not in self._tasks_completed
        ]
        if remaining_tasks:
            task = remaining_tasks[0]
            element_idx = OBS_ELEMENT_INDICES[task]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - all_goal[element_idx]
            )
            complete = distance < self.BONUS_THRESH[task]
            if complete:
                stage_success = True
                self._tasks_completed.append(task)
                if self.REMOVE_TASKS_WHEN_COMPLETE:
                    self.tasks_to_complete.remove(task)

        ### Record distances
        next_element = next_obj_obs[..., self._element_idx - idx_offset]
        gripper_pos = self._get_gripper_pos()
        obj_pos = self._get_obj_pos()
        dists = {
            "goal": np.linalg.norm(next_element - self._element_goal),
            "goal_init": self._dist_goal_init,
            "hand": np.linalg.norm(obj_pos - gripper_pos),
            "hand_init": self._dist_hand_init,
        }
        reward_dict["dist_goal"] = dists["goal"]
        reward_dict["dist_hand"] = dists["hand"]

        if wrong_task_completed:
            ### HACK reward penalty scaling
            reward = -900 * self._reward_scale
            score = 0
            reward_dict["bonus"] = score
            reward_dict["r_total"] = reward
            self._wrong_task_completed = True

        else:
            ### Calculate dense reward
            reward, _ = self._compute_reward(dists)
            bonus = float(stage_success)

            ### reward at success is 0 then terminate episode
            if self._negative_reward:
                reward -= self._num_stages
            ### Stage reward:
            if stage_success and self._sparse_stage_reward:
                reward += 1 * 300  ### 30x accumulated dense reward at success
            if not self._sparse_stage_reward:
                reward = reward + self._num_stages_completed
            ### MAX reward in (90, 120) range, set reward scale = 0.03?
            reward *= self._reward_scale
            reward_dict["bonus"] = bonus
            reward_dict["r_total"] = reward
            score = bonus

            ### Update init distances if advancing to a new stage
            if stage_success:
                self._record_init_dists()

        return reward_dict, score

class KitchenMicrowaveKettleBottomBurnerV0(KitchenInfosBase):
    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner"]


### Task 1 v0
class KitchenKettleLightHingeCabinetV0(KitchenInfosBase):
    TASK_ELEMENTS = ["kettle", "light switch", "hinge cabinet"]


### Task 1 v1
class KitchenKettleBottomBurnerHingeCabinetV0(KitchenInfosBase):
    TASK_ELEMENTS = ["kettle", "bottom burner", "hinge cabinet"]


### Task 2 v0-1
class KitchenBottomBurnerHingeCabinetMicrowaveV0(KitchenInfosBase):
    TASK_ELEMENTS = ["bottom burner", "hinge cabinet", "microwave"]


## Version 2 tasks
### Task 0
class KitchenMicrowaveKettleBottomBurnerV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner"]


### Task 1 v2
class KitchenKettleBottomBurnerHingeCabinetV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["kettle", "bottom burner", "hinge cabinet"]


### Task 2 v2
class KitchenBottomBurnerHingeCabinetMicrowaveV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["bottom burner", "hinge cabinet", "microwave"]


### Task 3 v2: easier out of order
class KitchenKettleHingeCabinetBottomBurnerV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["kettle", "hinge cabinet", "bottom burner"]


### Task 4 v2: easier out of order
class KitchenMicrowaveHingeCabinetBottomBurnerV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["microwave", "hinge cabinet", "bottom burner"]


### Two task set
class KitchenMicrowaveKettleSlideCabinetV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["microwave", "kettle", "slide cabinet"]


class KitchenKettleSlideCabinetHingeCabinetV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["kettle", "slide cabinet", "hinge cabinet"]


class KitchenKettleBottomBurnerSlideCabinetV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["kettle", "bottom burner", "slide cabinet"]


class KitchenMicrowaveBottomBurnerLightV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["microwave", "bottom burner", "light switch"]


class KitchenBottomBurnerLightHingeCabinetV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["bottom burner", "light switch", "hinge cabinet"]


### Microwave, Light tasks


class KitchenMicrowaveLightSlideCabinetV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["microwave", "light switch", "slide cabinet"]


class KitchenMicrowaveLightHingeCabinetV2(KitchenInfosBasev2):
    TASK_ELEMENTS = ["microwave", "light switch", "hinge cabinet"]
