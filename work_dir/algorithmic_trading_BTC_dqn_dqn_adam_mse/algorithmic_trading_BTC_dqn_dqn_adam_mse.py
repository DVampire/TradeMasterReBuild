data = dict(
    type='AlgorithmicTradingDataset',
    data_path='data/algorithmic_trading/BTC',
    train_path='data/algorithmic_trading/BTC/train.csv',
    valid_path='data/algorithmic_trading/BTC/valid.csv',
    test_path='data/algorithmic_trading/BTC/test.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    backward_num_day=5,
    forward_num_day=5,
    future_weights=0.2,
    initial_amount=100000,
    max_volume=1,
    transaction_cost_pct=0.001)
environment = dict(type='AlgorithmicTradingEnvironment')
agent = dict(
    type='AlgorithmicTradingDQN',
    memory_capacity=2000,
    epsilon=0.9,
    target_freq=50,
    gamma=0.9,
    future_loss_weights=0.2)
trainer = dict(
    type='AlgorithmicTradingTrainer',
    epochs=20,
    work_dir='work_dir/algorithmic_trading_BTC_dqn_dqn_adam_mse',
    if_remove=True)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
act_net = dict(type='QNet', n_state=82, n_action=3, hidden_nodes=256)
cri_net = dict(type='QNet', n_state=82, n_action=3, hidden_nodes=256)
task_name = 'algorithmic_trading'
dataset_name = 'BTC'
optimizer_name = 'adam'
loss_name = 'mse'
net_name = 'dqn'
agent_name = 'dqn'
work_dir = 'work_dir/algorithmic_trading_BTC_dqn_dqn_adam_mse'
