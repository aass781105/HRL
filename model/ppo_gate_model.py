import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional


class PPOGateNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 2,
        hidden: int = 256,
        num_layers: int = 3,
        separate_trunks: bool = False,
        actor_hidden: Optional[int] = None,
        actor_num_layers: Optional[int] = None,
        critic_hidden: Optional[int] = None,
        critic_num_layers: Optional[int] = None,
        value_hidden: Optional[int] = None,
        value_num_layers: int = 1,
    ):
        super().__init__()
        self.separate_trunks = bool(separate_trunks)

        def build_mlp(input_dim: int, width: int, depth: int) -> nn.Sequential:
            layers = []
            last_dim = input_dim
            for _ in range(max(1, int(depth))):
                layers.append(nn.Linear(last_dim, width))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.LayerNorm(width))
                last_dim = width
            return nn.Sequential(*layers)

        if self.separate_trunks:
            actor_hidden = int(hidden if actor_hidden is None else actor_hidden)
            actor_num_layers = int(num_layers if actor_num_layers is None else actor_num_layers)
            critic_hidden = int(hidden if critic_hidden is None else critic_hidden)
            critic_num_layers = int(num_layers if critic_num_layers is None else critic_num_layers)

            self.actor_trunk = build_mlp(obs_dim, actor_hidden, actor_num_layers)
            self.critic_trunk = build_mlp(obs_dim, critic_hidden, critic_num_layers)
            self.trunk = None
            policy_in_dim = actor_hidden
            value_in_dim = critic_hidden
        else:
            self.trunk = build_mlp(obs_dim, hidden, num_layers)
            self.actor_trunk = None
            self.critic_trunk = None
            policy_in_dim = hidden
            value_in_dim = hidden

        self.policy_head = nn.Linear(policy_in_dim, n_actions)

        value_hidden = int(hidden if value_hidden is None else value_hidden)
        value_layers = []
        value_last_dim = value_in_dim
        for _ in range(max(0, int(value_num_layers) - 1)):
            value_layers.append(nn.Linear(value_last_dim, value_hidden))
            value_layers.append(nn.ReLU(inplace=True))
            value_layers.append(nn.LayerNorm(value_hidden))
            value_last_dim = value_hidden
        value_layers.append(nn.Linear(value_last_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x):
        if self.separate_trunks:
            actor_h = self.actor_trunk(x)
            critic_h = self.critic_trunk(x)
        else:
            actor_h = critic_h = self.trunk(x)
        logits = self.policy_head(actor_h)
        value = self.value_head(critic_h).squeeze(-1)
        return logits, value

    def dist_and_value(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist, value
