import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional


# ── Helper: custom MLP block (used when any arch flag is enabled) ────────────
class _MLPBlock(nn.Module):
    """
    One MLP layer supporting optional GLU activation, pre-norm ordering,
    and residual (skip) connection.

    Norm ordering
    ─────────────
      pre_norm=False (default / post-norm):  Linear → ReLU or GLU → LayerNorm
      pre_norm=True  (pre-norm):             LayerNorm → Linear → ReLU or GLU

    Activation
    ──────────
      use_glu=False (default): ReLU
      use_glu=True:            Gated Linear Unit  h, g = split(Linear(2d)); out = h * sigmoid(g)

    Residual
    ────────
      use_residual=False (default): plain feedforward
      use_residual=True:            output += proj(input);  proj=Identity if dims match
    """

    def __init__(self, in_dim: int, out_dim: int,
                 use_glu: bool, pre_norm: bool, use_residual: bool):
        super().__init__()
        self.use_glu = use_glu
        self.pre_norm = pre_norm
        self.use_residual = use_residual

        # Pre-norm: LayerNorm is applied to the *input* before the linear transform
        self.pre_ln = nn.LayerNorm(in_dim) if pre_norm else None

        # GLU doubles the linear output dimension so half acts as a sigmoid gate
        linear_out_dim = out_dim * 2 if use_glu else out_dim
        self.linear = nn.Linear(in_dim, linear_out_dim)

        # Post-norm: LayerNorm is applied *after* the activation (only when pre_norm=False)
        self.post_ln = nn.LayerNorm(out_dim) if not pre_norm else None

        # Residual skip projection: required when in_dim ≠ out_dim
        # Restrict skips to same-width hidden blocks. A learned projection of
        # raw high-level state into the first hidden layer can saturate logits.
        self.proj = nn.Identity() if use_residual and in_dim == out_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # (Optional) pre-norm on input
        h = self.pre_ln(x) if self.pre_ln is not None else x

        # Linear transform
        h = self.linear(h)

        # Activation: GLU or ReLU
        if self.use_glu:
            h, gate = h.chunk(2, dim=-1)
            h = h * torch.sigmoid(gate)
        else:
            h = torch.relu(h)

        # (Optional) post-norm on output
        if self.post_ln is not None:
            h = self.post_ln(h)

        # (Optional) residual skip connection
        if self.proj is not None:
            h = h + self.proj(residual)

        return h


class _CustomMLP(nn.Module):
    """A stack of _MLPBlock layers, used when any architecture flag is enabled."""

    def __init__(self, input_dim: int, width: int, depth: int,
                 use_glu: bool, pre_norm: bool, use_residual: bool):
        super().__init__()
        blocks = []
        last_dim = input_dim
        for _ in range(max(1, int(depth))):
            blocks.append(_MLPBlock(last_dim, width, use_glu, pre_norm, use_residual))
            last_dim = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ── Main model ───────────────────────────────────────────────────────────────
class HLPPOGateNet(nn.Module):
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
        # ── New architecture flags (all False → identical to original behaviour) ──
        use_residual: bool = False,   # Residual / skip connections in MLP blocks
        use_glu: bool = False,        # Replace ReLU with Gated Linear Units (GLU)
        pre_norm: bool = False,       # Swap norm order: Linear→LN→Act becomes LN→Linear→Act
        manual_obs_dim: Optional[int] = None,
        ll_embed_raw_dim: int = 0,
        ll_embed_proj_dim: int = 16,
        initial_release_prob: float = -1.0,
    ):
        super().__init__()
        self.separate_trunks = bool(separate_trunks)
        self.ll_embed_raw_dim = max(0, int(ll_embed_raw_dim))
        self.manual_obs_dim = int(obs_dim - self.ll_embed_raw_dim if manual_obs_dim is None else manual_obs_dim)
        if self.ll_embed_raw_dim > 0:
            self.ll_embed_proj = nn.Sequential(
                nn.LayerNorm(self.ll_embed_raw_dim),
                nn.Linear(self.ll_embed_raw_dim, int(ll_embed_proj_dim)),
                nn.Tanh(),
            )
            self.ll_embed_scale = nn.Parameter(torch.zeros(1))
            trunk_input_dim = self.manual_obs_dim + int(ll_embed_proj_dim)
        else:
            self.ll_embed_proj = None
            self.ll_embed_scale = None
            trunk_input_dim = int(obs_dim)

        def build_mlp(input_dim: int, width: int, depth: int) -> nn.Module:
            # ── Default (all flags off): preserve original nn.Sequential structure ──
            # Keeping the same module types and indices ensures that existing .pth
            # checkpoint keys (trunk.0.weight, trunk.2.weight, …) load without any
            # key mismatch.  DO NOT change this branch.
            if not use_residual and not use_glu and not pre_norm:
                layers = []
                last_dim = input_dim
                for _ in range(max(1, int(depth))):
                    layers.append(nn.Linear(last_dim, width))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.LayerNorm(width))
                    last_dim = width
                return nn.Sequential(*layers)

            # ── New path: _CustomMLP with selected arch features ──────────────────
            return _CustomMLP(input_dim, width, int(depth),
                               use_glu, pre_norm, use_residual)

        if self.separate_trunks:
            actor_hidden      = int(hidden     if actor_hidden      is None else actor_hidden)
            actor_num_layers  = int(num_layers if actor_num_layers  is None else actor_num_layers)
            critic_hidden     = int(hidden     if critic_hidden     is None else critic_hidden)
            critic_num_layers = int(num_layers if critic_num_layers is None else critic_num_layers)

            self.actor_trunk  = build_mlp(trunk_input_dim, actor_hidden,  actor_num_layers)
            self.critic_trunk = build_mlp(trunk_input_dim, critic_hidden, critic_num_layers)
            self.trunk        = None
            policy_in_dim     = actor_hidden
            value_in_dim      = critic_hidden
        else:
            self.trunk        = build_mlp(trunk_input_dim, hidden, num_layers)
            self.actor_trunk  = None
            self.critic_trunk = None
            policy_in_dim     = hidden
            value_in_dim      = hidden

        self.policy_head = nn.Linear(policy_in_dim, n_actions)
        initial_release_prob = float(initial_release_prob)
        if n_actions == 2 and 0.0 < initial_release_prob < 1.0:
            with torch.no_grad():
                self.policy_head.weight.zero_()
                self.policy_head.bias.zero_()
                self.policy_head.bias[1] = torch.logit(torch.tensor(initial_release_prob, dtype=self.policy_head.bias.dtype))

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
        if self.ll_embed_proj is not None:
            manual = x[:, :self.manual_obs_dim]
            ll_embed = x[:, self.manual_obs_dim:self.manual_obs_dim + self.ll_embed_raw_dim]
            x = torch.cat((manual, self.ll_embed_scale * self.ll_embed_proj(ll_embed)), dim=-1)
        if self.separate_trunks:
            actor_h  = self.actor_trunk(x)
            critic_h = self.critic_trunk(x)
        else:
            actor_h = critic_h = self.trunk(x)
        logits = self.policy_head(actor_h)
        value  = self.value_head(critic_h).squeeze(-1)
        return logits, value

    def dist_and_value(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist, value
