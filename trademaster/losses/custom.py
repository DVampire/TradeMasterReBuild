from .builder import LOSSES
from torch import nn
from torch.nn import SmoothL1Loss
from torch.nn import MSELoss
from torch.nn.functional import kl_div
@LOSSES.register_module()
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

LOSSES.register_module(name= "SmoothL1Loss", force=False, module=SmoothL1Loss)
LOSSES.register_module(name="MSELoss", force=False, module=MSELoss)
LOSSES.register_module(name="KLloss", force=False, module=kl_div)