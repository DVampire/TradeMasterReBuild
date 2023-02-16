import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr, save_model, save_best_model, load_model, load_best_model
import numpy as np
import os
import pandas as pd


def make_market_information(df, technical_indicator):
    # based on the information, calculate the average for technical_indicator to present the market average
    all_dataframe_list = []
    index_list = df.index.unique().tolist()
    index_list.sort()
    for i in index_list:
        information = df[df.index == i]
        new_dataframe = []
        for tech in technical_indicator:
            tech_value = np.mean(information[tech])
            new_dataframe.append(tech_value)
        all_dataframe_list.append(new_dataframe)
    new_df = pd.DataFrame(all_dataframe_list,
                          columns=technical_indicator).values
    # new_df.to_csv(store_path)
    return new_df


def make_correlation_information(df: pd.DataFrame, feature="adjclose"):
    # based on the information, we are making the correlation matrix(which is N*N matric where N is the number of tickers) based on the specific
    # feature here,  as default is adjclose
    df.sort_values(by='tic', ascending=True, inplace=True)
    array_symbols = df['tic'].values

    # get data, put into dictionary then dataframe
    dict_sym_ac = {}  # key=symbol, value=array of adj close
    for sym in array_symbols:
        dftemp = df[df['tic'] == sym]
        dict_sym_ac[sym] = dftemp['adjcp'].values

    # create correlation coeff df
    dfdata = pd.DataFrame.from_dict(dict_sym_ac)
    dfcc = dfdata.corr().round(2)
    dfcc = dfcc.values
    return dfcc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True


@TRAINERS.register_module()
class PortfolioManagementDeepTraderTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementDeepTraderTrainer, self).__init__()

        self.kwargs = kwargs
        self.device = get_attr(kwargs, "device", None)

        self.epochs = get_attr(kwargs, "epochs", 20)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)
        self.num_threads = int(get_attr(kwargs, "num_threads", 8))
        self.if_remove = get_attr(kwargs, "if_remove", False)

    def init_before_training(self):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            print(f"| Arguments Remove work_dir: {self.work_dir}")
        else:
            print(f"| Arguments Keep work_dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path, exist_ok=True)

    def train_and_valid(self):

        valid_score_list = []
        for epoch in range(1, self.epochs+1):

            print("Train Episode: [{}/{}]".format(epoch, self.epochs))

            count = 0
            s = self.train_environment.reset()

            episode_reward_sum = 0
            while True:
                old_asset_state = s
                old_market_state = torch.from_numpy(
                    make_market_information(
                        self.train_environment.data,
                        technical_indicator=self.train_environment.tech_indicator_list)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_old = make_correlation_information(
                    self.train_environment.data)
                weights = self.agent.compute_weights_train(
                    s,
                    make_market_information(
                        self.train_environment.data,
                        technical_indicator=self.train_environment.tech_indicator_list),
                    corr_matrix_old)
                action_asset = self.agent.act_net(
                    torch.from_numpy(old_asset_state).float().to(self.device),
                    corr_matrix_old)
                action_market = self.agent.market_net(old_market_state)
                s, reward, done, _ = self.train_environment.step(weights)
                new_asset_state = s
                new_market_state = torch.from_numpy(
                    make_market_information(
                        self.train_environment.data,
                        technical_indicator=self.train_environment.tech_indicator_list)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_new = make_correlation_information(
                    self.train_environment.data)
                self.agent.store_transition(
                    torch.from_numpy(old_asset_state).float().to(self.device),
                    action_asset,
                    torch.tensor(reward).float().to(self.device),
                    torch.from_numpy(new_asset_state).float().to(self.device),
                    old_market_state, action_market, new_market_state,
                    corr_matrix_old, corr_matrix_new, self.agent.roh_bar)
                count = count + 1
                if count % 100 == 10:
                    self.agent.learn()
                if done:
                    print("Train Episode Reward Sum: {:04f}".format(episode_reward_sum))
                    break

            save_model(self.checkpoints_path,
                       epoch=epoch,
                       save=self.agent.get_save())

            print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
            s = self.valid_environment.reset()
            episode_reward_sum = 0
            while True:
                old_state = s
                old_market_state = torch.from_numpy(
                    make_market_information(
                        self.valid_environment.data,
                        technical_indicator=self.valid_environment.tech_indicator_list)
                ).unsqueeze(0).float().to(self.device)
                corr_matrix_old = make_correlation_information(
                    self.valid_environment.data)
                weights = self.agent.compute_weights_test(
                    s,
                    make_market_information(
                        self.valid_environment.data,
                        technical_indicator=self.valid_environment.tech_indicator_list),
                    corr_matrix_old)
                s, reward, done, _ = self.valid_environment.step(weights)
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
            corr_matrix_old = make_correlation_information(
                self.test_environment.data)
            weights = self.agent.compute_weights_test(
                s,
                make_market_information(
                    self.test_environment.data,
                    technical_indicator=self.test_environment.tech_indicator_list),
                corr_matrix_old)
            s, reward, done, _ = self.test_environment.step(weights)
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
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
        return daily_return