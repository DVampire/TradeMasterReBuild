from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr
import os
import pandas as pd


@TRAINERS.register_module()
class PortfolioManagementInvestorImitatorTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementInvestorImitatorTrainer, self).__init__()

        self.kwargs = kwargs
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
        self.best_model_path = os.path.join(self.work_dir, "best_model")
        if not os.path.exists(self.all_model_path):
            os.makedirs(self.all_model_path)
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)

    def train_and_valid(self):

        rewards_list = []
        for i in range(self.epochs):
            print("Train Episode: [{}/{}]".format(i+1, self.epochs))
            state = self.train_environment.reset()
            done = False
            actions = []
            while not done:
                action = self.agent.select_action(state)
                state, reward, done, _ = self.train_environment.step(action)
                actions.append(action)
                self.agent.act_net.rewards.append(reward)

            self.agent.learn()
            model_path = os.path.join(self.all_model_path, "num_epoch_" + str(i + 1))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, "policy_gradient.pth")
            torch.save(self.agent.act_net, model_path)
            print("Valid Episode: [{}/{}]".format(i + 1, self.epochs))
            state = self.valid_environment.reset()
            done = False
            rewards = 0
            while not done:
                action = self.agent.select_action(state)
                state, reward, done, _ = self.valid_environment.step(action)
                rewards += reward
            rewards_list.append(rewards)

        best_model_index = rewards_list.index(max(rewards_list))
        self.agent.act_net = torch.load(os.path.join(self.all_model_path, "num_epoch_" + str(best_model_index + 1), "policy_gradient.pth"))
        torch.save(self.agent.act_net, os.path.join(self.best_model_path, "policy_gradient.pth"))

    def test(self):
        self.agent.act_net = torch.load(os.path.join(self.best_model_path, "policy_gradient.pth"))
        state = self.test_environment.reset()
        done = False
        while not done:
            action = self.agent.select_action(state)
            state, reward, done, _ = self.test_environment.step(action)
        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "result.csv"))
        return rewards
