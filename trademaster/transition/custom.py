from .builder import TRANSITIONS
from collections import namedtuple

@TRANSITIONS.register_module()
def Transition():
    return namedtuple("Transition", ['state',
                                     'action',
                                     'reward',
                                     'undone',
                                     'next_state'])