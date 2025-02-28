import torch
from torch import nn
import numpy as np
import imp
import os

from garage.torch import as_torch_dict, global_device

from environments.kitchen.spirl.modules.subnetworks import Predictor, BasicPredictor
from environments.kitchen.spirl.modules.recurrent_modules import RecurrentPredictor
from environments.kitchen.spirl.utils.general_utils import (
    AttrDict,
    ParamDict,
    batch_apply,
)
from environments.kitchen.spirl.components.checkpointer import (
    load_by_key,
    freeze_modules,
    get_config_path,
)
from environments.kitchen.spirl.utils.pytorch_utils import (
    find_tensor,
    no_batchnorm_update,
    get_constant_parameter,
)
from environments.kitchen.spirl.modules.variational_inference import (
    MultivariateGaussian,
)
from environments.kitchen.spirl.modules.layers import LayerBuilderParams
from environments.kitchen.spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl

CONFIG_PATH = (
    "environments/kitchen/spirl/configs/skill_prior_learning/kitchen/hierarchical_cl"
)
# CONFIG_PATH = "environments/kitchen/spirl/configs/hrl/kitchen/spirl_cl"
PRETRAINED_MODEL_PATH = (
    "environments/kitchen/skill_prior_learning/kitchen/hierarchical_cl/weights"
)


def get_config():
    conf = AttrDict()

    # paths
    conf.exp_dir = None
    ### replace with config path
    conf.conf_path = get_config_path(CONFIG_PATH)

    # general and model configs
    print("loading from the config file {}".format(conf.conf_path))
    conf_module = imp.load_source("conf", conf.conf_path)
    conf.general = conf_module.configuration
    conf.model = conf_module.model_config
    # conf.agent = conf_module.agent_config

    # data config
    try:
        data_conf = conf_module.data_config
    except AttributeError:
        data_conf_file = imp.load_source(
            "dataset_spec", os.path.join(AttrDict(conf).data_dir, "dataset_spec.py")
        )
        data_conf = AttrDict()
        data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
        data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
    conf.data = data_conf

    # model loading config
    conf.ckpt_path = conf.model.checkpt_path if "checkpt_path" in conf.model else None

    ### From postprocess_conf:
    conf.model["batch_size"] = 128  # shouldn't be important
    conf.model.update(conf.data.dataset_spec)
    conf.model["device"] = conf.data["device"] = "cpu"  ## ???

    # Update "embedding_checkpoint"?
    conf.model.embedding_checkpoint = PRETRAINED_MODEL_PATH
    return conf


def load_skill_decoder():
    # get_config like spirl does for pretraining
    conf = get_config()

    # initialize SkillDecoder with config
    skill_decoder = SkillDecoder(conf.model)
    skill_decoder.to()
    # skill_decoder = torch.jit.script(skill_decoder)

    return skill_decoder


class SkillDecoder(ClSPiRLMdl):
    def __init__(self, params, logger=None):
        params["initial_log_sigma"] = -50
        params["squash_output_dist"] = True
        params["max_action_range"] = 1.0  # But should be 1?
        super().__init__(params, logger)

    def get_skill_action(self, obs, skill):
        assert obs.shape[1] == self.state_dim and skill.shape[1] == self.latent_dim
        # device = global_device()
        # if device.type != "cuda":
        #     import ipdb
        #
        #     ipdb.set_trace()
        input = AttrDict(
            cond_input=obs,  # to(device),  # condition decoding on state
            z=skill,  # .to(device),
        )
        with no_batchnorm_update(self):  # BN updates harm the initialized policy
            output_dist = self._compute_action_dist(input)
            action = output_dist.rsample()
            log_prob = output_dist.log_prob(action)
            if self._hp.squash_output_dist:
                action, log_prob = self._tanh_squash_output(action, log_prob)
            return AttrDict(action=action, log_prob=log_prob, dist=output_dist)

    def get_skill_actions(self, obss, skills):
        pass

    def _compute_action_dist(self, input):
        # Use wrapper to track this: during rollouts use HL z every H steps and execute LL policy every step
        # during update (ie with batch size > 1) recompute LL action from z
        # Compute action distribution from decoder
        act = self.decoder(torch.cat((input.cond_input, input.z), dim=-1))

        # if input.obs.shape[0] == 1:  ### ???
        #     # if self.steps_since_hl > self.horizon - 1:
        #     #     self.last_z = input.z
        #     #     self.steps_since_hl = 0
        #     act = self.decoder(torch.cat((input.cond_input, self.last_z), dim=-1))
        #     # self.steps_since_hl += 1
        # else:
        #     # during update (ie with batch size > 1) recompute LL action from z
        #     act = self.decoder(torch.cat((input.cond_input, input.z), dim=-1))
        # device = global_device()
        return MultivariateGaussian(
            mu=act,
            log_sigma=self._log_sigma[None].repeat(act.shape[0], 1),  # .to(device)
        )

    def _tanh_squash_output(self, action, log_prob):
        """Passes continuous output through a tanh function to constrain action range, adjusts log_prob."""
        action_new = self._hp.max_action_range * torch.tanh(action)
        log_prob_update = np.log(self._hp.max_action_range) + 2 * (
            np.log(2.0) - action - torch.nn.functional.softplus(-2.0 * action)
        ).sum(
            dim=-1
        )  # maybe more stable version from Youngwoon Lee
        return action_new, log_prob - log_prob_update

    def build_network(self):
        self.decoder = Predictor(
            self._hp,
            input_size=self.enc_size + self._hp.nz_vae,
            output_size=self._hp.action_dim,
            mid_size=self._hp.nz_mid_prior,
        )
        self._log_sigma = torch.tensor(
            self._hp.initial_log_sigma * np.ones(self._hp.action_dim, dtype=np.float32),
            device=self._hp.device,
            requires_grad=True,
        )

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print(
                "Loading pre-trained embedding from {}!".format(
                    self._hp.embedding_checkpoint
                )
            )

            self.load_state_dict(
                load_by_key(
                    self._hp.embedding_checkpoint,
                    "decoder",
                    self.state_dict(),
                    self.device,
                )
            )
            freeze_modules([self.decoder])

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()

        print("Putting model on device {}".format(device))
        self.decoder.to(device)
        # self._log_sigma = self._log_sigma.to(device)
