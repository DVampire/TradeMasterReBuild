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
class PortfolioManagementEIIETrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementEIIETrainer, self).__init__()

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
            print("Train Episode: [{}/{}]".format(i + 1, self.epochs))
            j = 0
            done = False
            s = self.train_environment.reset()
            while not done:

                old_state = s
                action = self.agent.act_net(torch.from_numpy(s).float().to(self.device))
                s, reward, done, _ = self.train_environment.step(
                    action.cpu().detach().numpy())
                self.agent.store_transition(
                    torch.from_numpy(old_state).float().to(self.device),
                    action,
                    torch.tensor(reward).float().to(self.device),
                    torch.from_numpy(s).float().to(self.device))
                j = j + 1
                if j % 200 == 1:
                    self.agent.learn()

            torch.save(self.agent.act_net,
                       os.path.join(self.all_model_path, "actor_num_epoch_{}.pth".format(i)))
            torch.save(self.agent.cri_net,
                       os.path.join(self.all_model_path, "critic_num_epoch_{}.pth".format(i)))
            print("Valid Episode: [{}/{}]".format(i + 1, self.epochs))
            s = self.valid_environment.reset()
            done = False
            rewards = 0
            while not done:
                old_state = s
                action = self.agent.act_net(torch.from_numpy(s).float().to(self.device))
                s, reward, done, _ = self.valid_environment.step(
                    action.cpu().detach().numpy())
                rewards = rewards + reward
            rewards_list.append(rewards)
        index = rewards_list.index(np.max(rewards_list))
        actor_model_path = os.path.join(self.all_model_path, "actor_num_epoch_{}.pth".format(index))
        critic_model_path = os.path.join(self.all_model_path, "critic_num_epoch_{}.pth".format(index))
        self.agent.act_net = torch.load(actor_model_path)
        self.agent.cri_net = torch.load(critic_model_path)
        torch.save(self.agent.act_net, os.path.join(self.best_model_path, "actor.pth"))
        torch.save(self.agent.cri_net, os.path.join(self.best_model_path, "critic.pth"))

    def test(self):
        self.agent.act_net = torch.load(os.path.join(self.best_model_path, "actor.pth"))
        self.agent.cri_net = torch.load(os.path.join(self.best_model_path, "critic.pth"))
        s = self.test_environment.reset()
        done = False
        while not done:
            old_state = s
            action = self.agent.act_net(torch.from_numpy(s).float().to(self.device))
            s, reward, done, _ = self.test_environment.step(
                action.cpu().detach().numpy())
        df_return = self.test_environment.save_portfolio_return_memory()
        df_assets = self.test_environment.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir + "result.csv"))
