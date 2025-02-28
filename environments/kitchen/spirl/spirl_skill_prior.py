import torch
from torch import nn
import numpy as np
import imp
import os

from garage.torch import as_torch_dict, global_device

from environments.kitchen.spirl.modules.subnetworks import Predictor
from environments.kitchen.spirl.utils.general_utils import (
    ParamDict,
    concat_inputs,
    remove_spatial,
)
from environments.kitchen.spirl.components.checkpointer import (
    load_by_key,
)
from environments.kitchen.spirl.modules.variational_inference import (
    MultivariateGaussian,
)
from environments.kitchen.spirl.modules.layers import (
    LayerBuilderParams,
    init_weights_xavier,
)
from environments.kitchen.spirl.spirl_skill_decoder import (
    get_config,
)
from environments.kitchen.spirl.utils.pytorch_utils import no_batchnorm_update


def load_skill_prior(num_heads=1):
    # get_config like spirl does for pretraining
    conf = get_config()

    # initialize SkillPrior with config

    skill_prior = SkillPrior(conf.model, num_heads)
    skill_prior.load_pretrained_weights()
    skill_prior.to()

    return skill_prior


class SkillPrior(nn.Module):
    def __init__(self, params, num_heads):
        super().__init__()
        ### Update configs
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(
            self._hp.use_convs, self._hp.normalization
        )
        self.device = self._hp.device

        ### initialize model
        self.p = nn.ModuleList(
            [
                Predictor(
                    self._hp,
                    input_size=self._hp.state_dim,
                    output_size=self._hp.nz_vae * 2,
                    # num_heads=num_heads,
                    num_layers=self._hp.num_prior_net_layers,
                    mid_size=self._hp.nz_mid_prior,
                )
            ]
        )

    def forward(self, input):
        with no_batchnorm_update(self):
            dist = self.p[0](input)
        return MultivariateGaussian(dist)

    def load_pretrained_weights(self):
        # import ipdb
        #
        # ipdb.set_trace()
        ## TODO: load pretrained weights, keys will not match for multiheaded
        self.load_state_dict(
            load_by_key(
                self._hp.embedding_checkpoint,
                "p",
                self.state_dict(),
                self.device,
            )
        )

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()

        print("Putting model on device {}".format(device))
        self.p.to(device)
        # self._log_sigma = self._log_sigma.to(device)

    def _default_hparams(self):
        hp = ParamDict(
            {
                "use_convs": False,
                "normalization": "batch",
                "device": None,
                "n_rollout_steps": 10,  # number of decoding steps
                "cond_decode": False,  # if True, conditions decoder on prior inputs
            }
        )
        # Network size
        hp.update(
            {
                "state_dim": 1,  # dimensionality of the state space
                "action_dim": 1,  # dimensionality of the action space
                "nz_enc": 32,  # number of dimensions in encoder-latent space
                "nz_vae": 10,  # number of dimensions in vae-latent space
                "nz_mid": 32,  # number of dimensions for internal feature spaces
                "nz_mid_lstm": 128,  # size of middle LSTM layers
                "n_lstm_layers": 1,  # number of LSTM layers
                "n_processing_layers": 3,  # number of layers in MLPs
            }
        )

        # Learned prior
        hp.update(
            {
                "n_prior_nets": 1,  # number of prior networks in ensemble
                "num_prior_net_layers": 6,  # number of layers of the learned prior MLP
                "nz_mid_prior": 128,  # dimensionality of internal feature spaces for prior net
                "nll_prior_train": True,  # if True, trains learned prior by maximizing NLL
                "learned_prior_type": "gauss",  # distribution type for learned prior, ['gauss', 'gmm', 'flow']
                "n_gmm_prior_components": 5,  # number of Gaussian components for GMM learned prior
            }
        )
        return hp


#### Overwriting spirl's stuff to get multiheaded predictor


class MultiheadProcessingNet(nn.Module):
    def __init__(
        self,
        in_dim,
        mid_dim,
        out_dim,
        num_heads,
        num_layers,
        builder,
        block=None,
        detached=False,
        final_activation=None,
    ):
        super().__init__()
        self.detached = detached

        if block is None:
            block = builder.def_block
        block = builder.wrap_block(block)

        self._layers = nn.ModuleList()
        self._layers.append(block(in_dim=in_dim, out_dim=mid_dim, normalization=None))

        for i in range(num_layers):
            self._layers.append(
                block(in_dim=mid_dim, out_dim=mid_dim, normalize=builder.normalize)
            )

        self._output_layers = nn.ModuleList()
        for i in range(num_heads):
            ### not sure about this
            self._output_layers.append(
                block(
                    in_dim=mid_dim,
                    out_dim=out_dim,
                    normalization=None,
                    activation=final_activation,
                )
            )
        ### To Do: check this still works properly
        import ipdb

        ipdb.set_trace()
        self.apply(init_weights_xavier)

    def forward(self, *inp):
        inp = concat_inputs(*inp)
        if self.detached:
            inp = inp.detach()
        x = inp
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]


class MultiheadPredictor(MultiheadProcessingNet):
    def __init__(
        self,
        hp,
        input_size,
        output_size,
        num_heads,
        num_layers=None,
        detached=False,
        spatial=True,
        final_activation=None,
        mid_size=None,
    ):
        self.spatial = spatial
        mid_size = hp.nz_mid if mid_size is None else mid_size
        if num_layers is None:
            num_layers = hp.n_processing_layers

        super().__init__(
            input_size,
            mid_size,
            output_size,
            num_heads,
            num_layers=num_layers,
            builder=hp.builder,
            detached=detached,
            final_activation=final_activation,
        )

    def forward(self, *inp):
        outs = super().forward(*inp)
        return [remove_spatial(out, yes=not self.spatial) for out in outs]
