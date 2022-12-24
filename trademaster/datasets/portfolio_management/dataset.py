from pathlib import Path
import sys
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import os.path as osp
from ..custom import CustomDataset
from ..builder import DATASETS
from trademaster.utils import get_attr

@DATASETS.register_module()
class PortfolioManagementDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(PortfolioManagementDataset, self).__init__()

        self.kwargs = kwargs

        self.data_path = osp.join(ROOT, get_attr(kwargs, "data_path", None))

        self.train_path = osp.join(ROOT, get_attr(kwargs, "train_path", None))
        self.valid_path = osp.join(ROOT, get_attr(kwargs, "valid_path", None))
        self.test_path = osp.join(ROOT, get_attr(kwargs, "test_path", None))

        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", [])
        self.initial_amount = get_attr(kwargs, "initial_amount", 100000)
        self.length_day = get_attr(kwargs, "length_day", 10)
        self.transaction_cost_pct = get_attr(kwargs, "transaction_cost_pct", 0.001)