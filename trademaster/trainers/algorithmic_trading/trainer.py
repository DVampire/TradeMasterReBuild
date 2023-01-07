from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr
import numpy as np
import os
import pandas as pd


@TRAINERS.register_module()
class AlgorithmicTradingTrainer(Trainer):
    def __init__(self, **kwargs):
        super(AlgorithmicTradingTrainer, self).__init__()
        self.device = get_attr(kwargs, "device", None)
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.all_model_path = os.path.join(self.work_dir, "all_model")
        if not os.path.exists(self.all_model_path):
            os.makedirs(self.all_model_path)
        self.best_model_path = os.path.join(self.work_dir, "best_model")
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)

    def train_and_valid(self):
        valid_score_list = []
        for i in range(self.epochs):
            print('<<<<<<<<<Episode: %s' % i)
            s = self.train_environment.reset()
            episode_reward_sum = 0
            while True:
                a = self.agent.choose_action(s)
                s_, r, done, info = self.train_environment.step(a)
                self.agent.store_transition(s, a, r, s_, info["volidality"])
                episode_reward_sum += r
                s = s_
                if self.agent.memory_counter > self.agent.memory_capacity:
                    self.agent.learn()
                if done:
                    print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                    break
            torch.save(self.agent.act_net, os.path.join(self.work_dir, "all_model", "num_epoch_{}.pth".format(i)))

            s = self.valid_environment.reset()
            episode_reward_sum = 0
            done = False
            while not done:
                a = self.agent.choose_action_test(s)
                s_, r, done, info = self.valid_environment.step(a)
                episode_reward_sum += r
            valid_score_list.append(episode_reward_sum)

        index = valid_score_list.index(np.max(valid_score_list))
        model_path = os.path.join(self.work_dir, "all_model", "num_epoch_{}.pth".format(index))
        self.agent.act_net = torch.load(model_path)
        torch.save(self.agent.act_net, os.path.join(self.best_model_path, "best_model.pth"))

    def test(self):
        self.agent.act_net = torch.load(os.path.join(self.best_model_path, "best_model.pth"))
        s = self.test_environment.reset()
        done = False
        while not done:
            a = self.agent.choose_action_test(s)
            s_, r, done, info = self.test_environment.step(a)
        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
        return daily_return

