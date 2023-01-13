from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
from ..custom import Trainer
from ..builder import TRAINERS
from trademaster.utils import get_attr
import os
import ray
from ray.tune.registry import register_env
from trademaster.environments.portfolio_management.sarl_environment import PortfolioManagementSARLEnvironment
import pandas as pd


def env_creator(config):
    return PortfolioManagementSARLEnvironment(config)


def select_algorithms(alg_name):
    alg_name = alg_name.upper()
    if alg_name == "A2C":
        from ray.rllib.agents.a3c.a2c import A2CTrainer as trainer
    elif alg_name == "DDPG":
        from ray.rllib.agents.ddpg.ddpg import DDPGTrainer as trainer
    elif alg_name == 'PG':
        from ray.rllib.agents.pg import PGTrainer as trainer
    elif alg_name == 'PPO':
        from ray.rllib.agents.ppo.ppo import PPOTrainer as trainer
    elif alg_name == 'SAC':
        from ray.rllib.agents.sac import SACTrainer as trainer
    elif alg_name == 'TD3':
        from ray.rllib.agents.ddpg.ddpg import TD3Trainer as trainer
    else:
        print(alg_name)
        print(alg_name == "A2C")
        print(type(alg_name))
        raise NotImplementedError
    return trainer


@TRAINERS.register_module()
class PortfolioManagementSARLTrainer(Trainer):
    def __init__(self, **kwargs):
        super(PortfolioManagementSARLTrainer, self).__init__()

        self.device = get_attr(kwargs, "device", None)
        self.configs = get_attr(kwargs, "configs", None)
        self.agent_name = get_attr(kwargs, "agent_name", "ppo")
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.dataset = get_attr(kwargs, "dataset", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        ray.init(ignore_reinit_error=True)
        self.trainer_name = select_algorithms(self.agent_name)
        self.configs["env"] = PortfolioManagementSARLEnvironment
        self.configs["env_config"] = dict(dataset=self.dataset, task="train")
        register_env("portfolio_management", env_creator)

        self.all_model_path = os.path.join(self.work_dir, "all_model")
        self.best_model_path = os.path.join(self.work_dir, "best_model")
        if not os.path.exists(self.all_model_path):
            os.makedirs(self.all_model_path)
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)

    def train_and_valid(self):
        self.sharpes = []
        self.checkpoints = []

        self.trainer = self.trainer_name(env="portfolio_management", config=self.configs)

        for i in range(self.epochs):
            print("Train Episode: [{}/{}]".format(i + 1, self.epochs))
            self.trainer.train()
            config = dict(dataset=self.dataset, task="valid")
            self.valid_environment = env_creator(config)

            print("Valid Episode: [{}/{}]".format(i + 1, self.epochs))
            state = self.valid_environment.reset()
            done = False
            while not done:
                action = self.trainer.compute_single_action(state)
                state, reward, done, information = self.valid_environment.step(
                    action)
            self.sharpes.append(information["sharpe_ratio"])
            checkpoint = self.trainer.save(self.all_model_path)
            self.checkpoints.append(checkpoint)
        self.loc = self.sharpes.index(max(self.sharpes))
        self.trainer.restore(self.checkpoints[self.loc])
        self.trainer.save(self.best_model_path)
        ray.shutdown()

    def test(self):
        self.trainer.restore(self.best_model_path)
        config = dict(dataset=self.dataset, task="test")
        self.test_environment = env_creator(config)
        state = self.test_environment.reset()
        done = False
        while not done:
            action = self.trainer.compute_single_action(state)
            state, reward, done, sharpe = self.test_environment.step(action)
        rewards = self.test_environment.save_asset_memory()
        assets = rewards["total assets"].values
        df_return = self.test_environment.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
