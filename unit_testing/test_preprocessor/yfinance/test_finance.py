import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.agents.algorithmic_trading import AlgorithmicTradingDQN
from trademaster.preprocessor.yfinance_preprocessor import YfinancePreprocessor
from trademaster.preprocessor.builder import build_preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config",
                        default=osp.join(ROOT, "configs", "_base_",
                                         "preprocessor", "yahoofinance",
                                         "dj30.py"),
                        help="download datasets config file path")
    args = parser.parse_args()
    return args


def test_preprocessor():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)

    dataset = build_preprocessor(cfg)


if __name__ == '__main__':
    test_preprocessor()