
agent = dict(
    type = "AlgorithmicTradingDQN",
    memory_capacity = 2000,
    epsilon =0.9,
    target_freq = 50,
    gamma = 0.9,
    future_loss_weights = 0.2
)