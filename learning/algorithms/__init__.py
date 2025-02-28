"""PyTorch algorithms."""
from learning.algorithms.sac import SAC
from learning.algorithms.dnc_sac import DnCSAC
from learning.algorithms.mop_dnc import MoPDnC
from learning.algorithms.mop_sac import MoPSAC
from learning.algorithms.cds import OnlineCDS
from learning.algorithms.qmp_uds import QMPUDS
from learning.algorithms.multi_critic_actor import MultiCriticAL
from learning.algorithms.qmp_mcal import QMPMultiCriticAL

__all__ = [
    "SAC",
    "DnCSAC",
    "MoPDnC",
    "MoPSAC",
    "OnlineCDS",
    "QMPUDS",
    "MultiCriticAL",
    "QMPMultiCriticAL",
]


