task_name = "algorithmic_trading"
dataset_name = "BTC"
optimizer_name = "adam"
loss_name = "mse"
net_name = "dqn"
agent_name = "dqn"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"


_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
]

data = dict(
    type='AlgorithmicTradingDataset',
    data_path='data/algorithmic_trading/BTC',
    train_path='data/algorithmic_trading/BTC/train.csv',
    valid_path='data/algorithmic_trading/BTC/valid.csv',
    test_path='data/algorithmic_trading/BTC/test.csv',
    test_style_path='data/algorithmic_trading/BTC/test_labeled_3_24.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    backward_num_day=5,
    forward_num_day=5,
    future_weights=0.2,
    initial_amount=100000,
    max_volume=1,
    transaction_cost_pct=0.001,
    test_style=0)
environment = dict(type='AlgorithmicTradingEnvironment')
act_net = dict(type='QNet', n_state=82, n_action=3, hidden_nodes=256)
cri_net = dict(type='QNet', n_state=82, n_action=3, hidden_nodes=256)
agent = dict(
    type='AlgorithmicTradingDQN',
    memory_capacity=2000,
    epsilon=0.9,
    target_freq=50,
    gamma=0.9,
    future_loss_weights=0.2)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
trainer = dict(
    type='AlgorithmicTradingTrainer',
    epochs=20,
    work_dir=work_dir,
    if_remove=True)