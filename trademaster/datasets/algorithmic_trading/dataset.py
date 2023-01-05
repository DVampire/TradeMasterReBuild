from pathlib import Path
import sys
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import os.path as osp
from ..custom import CustomDataset
from ..builder import DATASETS
from trademaster.utils import get_attr
import pandas as pd
import os

@DATASETS.register_module()
class AlgorithmicTradingDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(AlgorithmicTradingDataset, self).__init__()

        self.kwargs = kwargs

        self.data_path = osp.join(ROOT, get_attr(kwargs, "data_path", None))

        self.train_path = osp.join(ROOT, get_attr(kwargs, "train_path", None))
        self.valid_path = osp.join(ROOT, get_attr(kwargs, "valid_path", None))
        self.test_path = osp.join(ROOT, get_attr(kwargs, "test_path", None))
        self.test_style_path=osp.join(ROOT, get_attr(kwargs, "test_style_path", None))
        test_style=get_attr(kwargs, "test_style", None)
        if test_style!=-1:
            backward_num_day= get_attr(kwargs, "backward_num_day", None)
            forward_num_day=get_attr(kwargs, "forward_num_day", None)
            self.test_style_paths=[]
            data = pd.read_csv(self.test_style_path)
            # print(data)
            # data['index_by_tick']=data.index
            data = data.reset_index()
            data = data.loc[data['label'] == int(test_style), :]
            print(data)
            intervals, index_by_tick_list = self.get_styled_intervals_and_gives_new_index(data)
            data.drop(columns=['index'], inplace=True)
            if not os.path.exists('temp'):
                os.makedirs('temp')
            for i, interval in enumerate(intervals):
                data_temp = data.iloc[interval[0]:interval[1], :]
                data_temp.index = index_by_tick_list[i]
                data_temp.to_csv('temp/' + str(test_style) + '_' + str(i) + '.csv')
                if max(index_by_tick_list[i]) + 1 <= backward_num_day + forward_num_day + 2:
                    print('The ' + str(i) + '_th segment length is less than the min length so it won\'t be tested')
                    continue
                temp_path='temp/' + str(test_style) + '_' + str(i) + '.csv'
                self.test_style_paths.append(osp.join(ROOT, temp_path))



        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", [])
        self.backward_num_day = get_attr(kwargs, "backward_num_day", 5)
        self.forward_num_day = get_attr(kwargs, "forward_num_day", 5)
        self.future_weights = get_attr(kwargs, "future_weights", 0.2)
        self.initial_amount = get_attr(kwargs, "initial_amount", 100000)
        self.max_volume = get_attr(kwargs, "max_volume", 1)
        self.transaction_cost_pct = get_attr(kwargs, "transaction_cost_pct", 0.001)

    def get_styled_intervals_and_gives_new_index(self,data):
        index_by_tick_list = []
        index_by_tick = []
        date = data['date'].to_list()
        last_date = date[0]
        date_counter = 0
        index = data['index'].to_list()
        last_value = index[0] - 1
        last_index = 0
        intervals = []
        for i in range(data.shape[0]):
            if last_value != index[i] - 1:
                date_counter = -1
                intervals.append([last_index, i])
                last_value = index[i]
                last_index = i
                index_by_tick_list.append(index_by_tick)
                index_by_tick = []
            if date[i] != last_date:
                date_counter += 1
            index_by_tick.append(date_counter)
            last_value = index[i]
            last_date = date[i]
        intervals.append([last_index, data.shape[0]])
        index_by_tick_list.append(index_by_tick)
        return intervals, index_by_tick_list