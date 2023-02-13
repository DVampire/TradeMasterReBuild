from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_model, save_best_model, load_model, load_best_model
import numpy as np
import os
import pandas as pd
import random


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
        self.seeds_list = get_attr(kwargs, "seeds_list", [12345])

        self.work_dir = os.path.join(ROOT, self.work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        self.set_seed(random.choice(self.seeds_list))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True

    def train_and_valid(self):

        valid_score_list = []
        for epoch in range(1, self.epochs + 1):
            print("Train Episode: [{}/{}]".format(epoch, self.epochs))
            s = self.train_environment.reset()

            count = 0
            episode_reward_sum = 0
            while True:
                old_state = s
                action = self.agent.act(torch.from_numpy(s).float().to(self.device))
                s, reward, done, _ = self.train_environment.step(
                    action.cpu().detach().numpy())
                self.agent.store_transition(
                    torch.from_numpy(old_state).float().to(self.device),
                    action,
                    torch.tensor(reward).float().to(self.device),
                    torch.from_numpy(s).float().to(self.device))

                episode_reward_sum += reward

                count = count + 1
                if count % 200 == 1:
                    self.agent.learn()
                if done:
                    print("Train Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break

            save_model(self.checkpoints_path,
                       epoch=epoch,
                       save=self.agent.get_save())

            print("Valid Episode: [{}/{}]".format(epoch + 1, self.epochs))
            s = self.valid_environment.reset()

            episode_reward_sum = 0
            while True:
                old_state = s
                action = self.agent.act(torch.from_numpy(s).float().to(self.device))
                s, reward, done, _ = self.valid_environment.step(
                    action.cpu().detach().numpy())

                episode_reward_sum += reward
                if done:
                    print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break
            valid_score_list.append(episode_reward_sum)

        max_index = np.argmax(valid_score_list)
        save_best_model(
            output_dir=self.checkpoints_path,
            epoch=max_index + 1,
            save=self.agent.get_save()
        )

    def test(self):
        load_best_model(self.checkpoints_path, save=self.agent.get_save(), is_train=False)

        print("Test Best Episode")
        s = self.test_environment.reset()

        episode_reward_sum = 0
        while True:
            old_state = s
            action = self.agent.act_net(torch.from_numpy(s).float().to(self.device))
            s, reward, done, _ = self.test_environment.step(
                action.cpu().detach().numpy())
            episode_reward_sum += reward
            if done:
                print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        df_return = self.test_environment.save_portfolio_return_memory()
        df_assets = self.test_environment.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir + "test_result.csv"))
        return daily_return
