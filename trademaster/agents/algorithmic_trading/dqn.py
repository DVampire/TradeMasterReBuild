import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr, ReplayBuffer
import torch
from torch import Tensor
from typing import Tuple
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from types import MethodType

def get_optim_param(optimizer: torch.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list

@AGENTS.register_module()
class AlgorithmicTradingDQN(AgentBase):
    def __init__(self, **kwargs):
        super(AlgorithmicTradingDQN, self).__init__(**kwargs)

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.device = get_attr(kwargs, "device", torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_step = get_attr(kwargs,"max_step", 12345)  # the max step number of an episode. 'set as 12345 in default.
        self.state_dim = get_attr(kwargs, "state_dim", 10) # vector dimension (feature number) of state
        self.action_dim = get_attr(kwargs, "action_dim", 2) # vector dimension (feature number) of action

        '''Arguments for reward shaping'''
        self.gamma = get_attr(kwargs, "gamma", 0.99)  # discount factor of future rewards
        self.reward_scale = get_attr(kwargs, "reward_scale", 2 ** 0)  # an approximate target reward usually be closed to 256
        self.repeat_times = get_attr(kwargs, "repeat_times", 1.0)  # repeatedly update network using ReplayBuffer
        self.batch_size = int(get_attr(kwargs, "batch_size", 64))

        self.clip_grad_norm = get_attr(kwargs, "clip_grad_norm", 3.0)  # clip the gradient after normalization
        self.soft_update_tau = get_attr(kwargs, "soft_update_tau", 0)  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = get_attr(kwargs, "state_value_tau", 5e-3)  # the tau of normalize for value and state

        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)

        '''network'''
        self.act = self.act_target = get_attr(kwargs, "act", None).to(self.device)
        self.cri = self.cri_target = get_attr(kwargs, "cri", None) if get_attr(kwargs, "cri", None) else self.act

        '''optimizer'''
        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None) if get_attr(kwargs, "cri_optimizer", None) else self.act_optimizer
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        """attribute"""
        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.if_use_per = get_attr(kwargs, 'if_use_per', False)  # use PER (Prioritized Experience Replay)
        if self.if_use_per:
            self.criterion = get_attr(kwargs, "criterion", None)
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = get_attr(kwargs, "criterion", None)
            self.get_obj_critic = self.get_obj_critic_raw
        self.act_target = self.cri_target = deepcopy(self.act)

    def get_save(self):
        models = {
            "act":self.act,
            "cri":self.cri,
            "act_target": self.act_target,
            "cri_target": self.cri_target,
        }
        optimizers = {
            "act_optimizer":self.act_optimizer,
            "cri_optimizer":self.cri_optimizer
        }
        res = {
            "models":models,
            "optimizers":optimizers
        }
        return res

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            num_envs == 1
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(1, 1)) if if_random \
                else get_action(state.unsqueeze(0))  # different
            states[t] = state

            ary_action = action[0, 0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            state = torch.as_tensor(env.reset() if done else ary_state, dtype=torch.float32, device=self.device)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (num_envs, state_dim) for a vectorized env.

        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(self.num_envs, 1)) if if_random \
                else get_action(state).detach()  # different
            states[t] = state

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            state, action, reward, undone, next_s = buffer.sample(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0].squeeze(1)
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state).gather(1, action.long()).squeeze(1)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            state, action, reward, undone, next_s, is_weights = buffer.sample(batch_size)

            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0].squeeze(1)
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state).gather(1, action.long()).squeeze(1)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value

    def get_returns(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        if self.num_envs == 1:
            last_state = self.last_state.unsqueeze(0)
        else:
            last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_update_amp(self, optimizer: torch.optim, objective: Tensor):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.cri.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))