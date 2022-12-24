from pathlib import Path
import sys
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import os.path as osp
from ..custom import CustomDataset
from ..builder import DATASETS
from trademaster.utils import get_attr

@DATASETS.register_module()
class OrderExecutionDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(OrderExecutionDataset, self).__init__()

        self.kwargs = kwargs

        self.data_path = osp.join(ROOT, get_attr(kwargs, "data_path", None))

        self.train_path = osp.join(ROOT, get_attr(kwargs, "train_path", None))
        self.valid_path = osp.join(ROOT, get_attr(kwargs, "valid_path", None))
        self.test_path = osp.join(ROOT, get_attr(kwargs, "test_path", None))

        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", [])
        self.backward_num_day = get_attr(kwargs, "backward_num_day", 5)
        self.forward_num_day = get_attr(kwargs, "forward_num_day", 5)
        self.future_weights = get_attr(kwargs, "future_weights", 0.2)
        self.initial_amount = get_attr(kwargs, "initial_amount", 100000)
        self.max_volume = get_attr(kwargs, "max_volume", 1)
        self.transaction_cost_pct = get_attr(kwargs, "transaction_cost_pct", 0.001)
        self.length_keeping = get_attr(kwargs, "length_keeping", 30)
        self.state_length = get_attr(kwargs, "state_length", 10)
        self.target_order = get_attr(kwargs, "target_order", 1)