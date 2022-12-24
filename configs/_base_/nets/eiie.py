
act_net = dict(
    type = "EIIEConv",
    n_input = None,
    n_output=1,
    length=None,
    kernel_size=3,
    num_layer=1,
    n_hidden=32
)

cri_net = dict(
    type = "EIIECritic",
    n_input = None,
    n_output=1,
    length=None,
    kernel_size=3,
    num_layer=1,
    n_hidden=32
)