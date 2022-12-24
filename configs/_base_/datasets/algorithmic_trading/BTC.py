
data = dict(
    type = "AlgorithmicTradingDataset",
    data_path = "data/algorithmic_trading/BTC",
    train_path = "data/algorithmic_trading/BTC/train.csv",
    valid_path = "data/algorithmic_trading/BTC/valid.csv",
    test_path = "data/algorithmic_trading/BTC/test.csv",
    tech_indicator_list = [
        "high",
        "low",
        "open",
        "close",
        "adjcp",
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30"
    ],
    backward_num_day = 5,
    forward_num_day = 5,
    future_weights = 0.2,
    initial_amount = 100000,
    max_volume = 1,
    transaction_cost_pct = 0.001
)