from common_utils import nonzero_averaging
from model.sub_layers import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualMLPEncoder(nn.Module):
    def __init__(self, config):
        """
        MLP 版 DAN：不使用 GNN/注意力，只用 MLP 對 fea_j / fea_m 做逐節點編碼，並輸出全域聚合特徵。
        需要的 config 欄位：
          - fea_j_input_dim: int   (例如 14)
          - fea_m_input_dim: int   (例如 8)
          - layer_fea_output_dim: List[int]，定義每一層的輸出維度（例如 [128, 128, 64]）
          - dropout_prob: float    (可選，用於輸出後 dropout)
        """
        super(DualMLPEncoder, self).__init__()

        # --- 讀 config ---
        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.dropout_prob = getattr(config, "dropout_prob", 0.0)

        # 讀取特徵提取層的維度列表 (例如 [128, 64])
        # 這列表現在完整定義了 MLP 的每一層輸出
        self.layer_dims = config.layer_fea_output_dim
        assert len(self.layer_dims) >= 1, "layer_fea_output_dim must have at least one element"
        
        # 最終輸出維度取列表最後一個值
        self.output_dim = int(self.layer_dims[-1])

        # --- 建兩個 MLP：各自給 job/operation 與 machine 特徵 ---
        # 使用新的 flexible MLP，直接傳入維度列表
        self.j_encoder = MLP(
            input_dim=self.fea_j_input_dim,
            hidden_dims=self.layer_dims
        )
        self.m_encoder = MLP(
            input_dim=self.fea_m_input_dim,
            hidden_dims=self.layer_dims
        )

        # 輕量正規化與 Dropout（可有可無）
        self.j_ln = nn.LayerNorm(self.output_dim)
        self.m_ln = nn.LayerNorm(self.output_dim)
        self.drop = nn.Dropout(self.dropout_prob)

    @torch.no_grad()
    def _maybe_mask_zero_(self, x, mask, dim_len):
        """
        可選：用 mask 將無效節點清 0。
        x:   [B, L, D]
        mask: 任意可還原到 [B, L] 的 0/1（True/False）遮罩
        dim_len: L
        """
        if mask is None:
            return x
        # 嘗試把 mask 壓成 [B, L]
        m = mask
        if m.dim() > 2:
            m = m.reshape(m.shape[0], dim_len, -1).any(dim=-1)  # [B, L]
        if m.size(1) != dim_len:
            return x  # 尺寸對不上就跳過
        x[~m] = 0.0
        return x

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask=None, fea_pairs=None):
        """
        與舊版保持相同介面與回傳：
          in:
            fea_j:  [B, N, fea_j_input_dim]
            op_mask:[B, ...]  （本版可忽略）
            candidate: [B, J]  （本版可忽略）
            fea_m:  [B, M, fea_m_input_dim]
            mch_mask:[B, M, M]（本版可忽略）
            comp_idx:[B, M, M, J]（本版可忽略）
          out:
            fea_j:        [B, N, output_dim]
            fea_m:        [B, M, output_dim]
            fea_j_global: [B, output_dim]
            fea_m_global: [B, output_dim]
        """
        B, N, _ = fea_j.shape
        Bm, M, _ = fea_m.shape
        assert B == Bm, "Batch size mismatch between fea_j and fea_m"

        # --- 逐節點 MLP 編碼 ---
        j_flat = fea_j.reshape(B * N, -1)    # [B*N, F_j]
        m_flat = fea_m.reshape(B * M, -1)    # [B*M, F_m]

        j_valid = fea_j.abs().sum(dim=-1, keepdim=True) > 0
        m_valid = fea_m.abs().sum(dim=-1, keepdim=True) > 0

        j_out = self.j_encoder(j_flat).reshape(B, N, self.output_dim)  # [B, N, D]
        m_out = self.m_encoder(m_flat).reshape(B, M, self.output_dim)  # [B, M, D]

        # 正規化 + Dropout（與原本 DAN 的最後層激活對齊的輕量處理）
        j_out = self.drop(self.j_ln(j_out))
        m_out = self.drop(self.m_ln(m_out))
        j_out = j_out.masked_fill(~j_valid, 0.0)
        m_out = m_out.masked_fill(~m_valid, 0.0)

        # （可選）用 mask 清 0（如果你的 mask 能表達哪些節點/機台無效）
        # j_out = self._maybe_mask_zero_(j_out, op_mask, N)
        # m_out = self._maybe_mask_zero_(m_out, mch_mask.diagonal(dim1=1, dim2=2), M)  # 取對角當作可用機台

        # 全域聚合：用 nonzero_averaging（專案內已經有同名函式的話可直接用那個）
        fea_j_global = nonzero_averaging(j_out)  # [B, D]
        fea_m_global = nonzero_averaging(m_out)  # [B, D]
        return j_out, m_out, fea_j_global, fea_m_global




class LLMLPNet(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(LLMLPNet, self).__init__()
        device = torch.device(config.device)

        self.pair_input_dim = int(getattr(config, "fea_pair_input_dim", 9))
        self.critic_size_context_max_n_j = float(getattr(config, "critic_size_context_max_n_j", 30.0))
        # Temporarily disable the critic-only raw tardiness summary.
        # Keep _critic_raw_summary below so it can be restored without rewriting the feature logic.
        self.critic_summary_dim = 0

        self.embedding_output_dim = config.layer_fea_output_dim[-1]
        self.separate_actor_critic_encoder = bool(getattr(config, "separate_actor_critic_encoder", False))

        self.feature_exact = DualMLPEncoder(config).to(
            device)
        if self.separate_actor_critic_encoder:
            self.critic_feature_exact = DualMLPEncoder(config).to(device)
        else:
            self.critic_feature_exact = None
        self.actor = Actor(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        critic_input_dim = 2 * self.embedding_output_dim + self.critic_summary_dim
        self.critic = Critic(config.num_mlp_layers_critic, critic_input_dim, config.hidden_dim_critic, 1).to(device)

    def _adapt_legacy_input_weight(self, state_dict, key):
        target = self.state_dict().get(key, None)
        source = state_dict.get(key, None)
        if target is None or source is None or source.shape == target.shape:
            return
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            return
        adapted = target.clone()
        copy_cols = min(source.shape[1], target.shape[1])
        adapted[:, :copy_cols] = source[:, :copy_cols]
        state_dict[key] = adapted
        print(f"Adapted legacy checkpoint tensor {key}: {tuple(source.shape)} -> {tuple(target.shape)}")

    def _copy_legacy_input_weight(self, state_dict, source_key, target_key):
        target = self.state_dict().get(target_key, None)
        source = state_dict.get(source_key, None)
        if target is None or source is None:
            return
        if source.ndim == 2 and target.ndim == 2 and source.shape[0] == target.shape[0]:
            adapted = target.clone()
            copy_cols = min(source.shape[1], target.shape[1])
            adapted[:, :copy_cols] = source[:, :copy_cols]
            state_dict[target_key] = adapted
        elif source.shape == target.shape:
            state_dict[target_key] = source.clone()

    def load_state_dict(self, state_dict, strict=True):
        adapted = copy.deepcopy(state_dict)
        self._adapt_legacy_input_weight(adapted, "feature_exact.j_encoder.layers.0.weight")
        self._adapt_legacy_input_weight(adapted, "critic_feature_exact.j_encoder.layers.0.weight")
        self._adapt_legacy_input_weight(adapted, "actor.linear.weight")
        self._adapt_legacy_input_weight(adapted, "actor.linears.0.weight")
        self._adapt_legacy_input_weight(adapted, "critic.linear.weight")
        self._adapt_legacy_input_weight(adapted, "critic.linears.0.weight")
        current = self.state_dict()
        if self.separate_actor_critic_encoder:
            for key, value in list(adapted.items()):
                if key.startswith("feature_exact."):
                    critic_key = key.replace("feature_exact.", "critic_feature_exact.", 1)
                    if critic_key in current and critic_key not in adapted:
                        adapted[critic_key] = value.clone()
        for branch in ("small", "mid", "large", "loose", "mixed", "tight"):
            self._copy_legacy_input_weight(adapted, f"critic.{branch}.linear.weight", "critic.linear.weight")
            self._copy_legacy_input_weight(adapted, f"critic.{branch}.linear.bias", "critic.linear.bias")
            for idx in range(16):
                self._copy_legacy_input_weight(adapted, f"critic.{branch}.linears.{idx}.weight", f"critic.linears.{idx}.weight")
                self._copy_legacy_input_weight(adapted, f"critic.{branch}.linears.{idx}.bias", f"critic.linears.{idx}.bias")
        for key in list(adapted.keys()):
            if key.startswith("critic.") and key not in current:
                del adapted[key]
        for key, value in current.items():
            if key not in adapted:
                adapted[key] = value
        return super().load_state_dict(adapted, strict=strict)

    def _critic_raw_summary(self, fea_j, candidate):
        valid = torch.count_nonzero(fea_j, dim=-1) != 0
        valid_f = valid.to(fea_j.dtype)
        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        slack = fea_j[:, :, 11]
        tardy_flag = fea_j[:, :, 13]
        current_td = fea_j[:, :, 14]

        slack_mean = (slack * valid_f).sum(dim=1, keepdim=True) / denom
        centered = torch.where(valid, slack - slack_mean, torch.zeros_like(slack))
        slack_std = torch.sqrt((centered * centered).sum(dim=1, keepdim=True) / denom)
        slack_min = torch.where(valid, slack, torch.full_like(slack, float("inf"))).min(dim=1, keepdim=True).values
        slack_min = torch.where(torch.isfinite(slack_min), slack_min, torch.zeros_like(slack_min))

        tardy_ratio = (tardy_flag * valid_f).sum(dim=1, keepdim=True) / denom
        current_td_mean = (current_td * valid_f).sum(dim=1, keepdim=True) / denom
        current_td_max = torch.where(valid, current_td, torch.zeros_like(current_td)).max(dim=1, keepdim=True).values

        max_n_j = max(self.critic_size_context_max_n_j, 1.0)
        n_j_ratio = torch.full_like(slack_min, float(candidate.size(1)) / max_n_j)

        return torch.cat(
            (slack_min, slack_mean, slack_std, tardy_ratio, current_td_max, current_td_mean, n_j_ratio),
            dim=-1
        )

    def _compute_policy_features(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        raw_fea_j = fea_j
        raw_fea_m = fea_m
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs
        )
        if self.separate_actor_critic_encoder:
            _, _, critic_fea_j_global, critic_fea_m_global = self.critic_feature_exact(
                raw_fea_j, op_mask, candidate, raw_fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs
            )
        else:
            critic_fea_j_global, critic_fea_m_global = fea_j_global, fea_m_global
        sz_b = fea_j.size(0)
        M = fea_m.size(1)
        J = candidate.size(1)
        d = fea_j.size(-1)

        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        fea_j_jc = torch.gather(fea_j, 1, candidate_idx)

        fea_j_jc_serialized = fea_j_jc.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        fea_gj_input = fea_j_global.unsqueeze(1).expand_as(fea_j_jc_serialized)
        fea_gm_input = fea_m_global.unsqueeze(1).expand_as(fea_j_jc_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        candidate_feature = torch.cat((fea_j_jc_serialized, fea_m_serialized, fea_gj_input,
                                       fea_gm_input, fea_pairs), dim=-1)
        if self.critic_summary_dim > 0:
            raw_critic_summary = self._critic_raw_summary(raw_fea_j, candidate)
            global_feature = torch.cat((critic_fea_j_global, critic_fea_m_global, raw_critic_summary), dim=-1)
        else:
            global_feature = torch.cat((critic_fea_j_global, critic_fea_m_global), dim=-1)
        return candidate_feature, global_feature

    def policy_only(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        candidate_feature, _ = self._compute_policy_features(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs
        )
        sz_b = candidate.size(0)
        candidate_scores = self.actor(candidate_feature).squeeze(-1)
        mask_flat = dynamic_pair_mask.reshape(sz_b, -1)
        if mask_flat.all(dim=1).any():
            bad_idx = torch.where(mask_flat.all(dim=1))[0][:8].detach().cpu().tolist()
            raise RuntimeError(f"policy_only received all-masked action rows at batch indices {bad_idx}")
        if torch.isnan(candidate_scores).any():
            raise RuntimeError("policy_only produced NaN candidate_scores before masking")
        candidate_scores = candidate_scores.masked_fill(mask_flat, float('-inf'))
        pi = F.softmax(candidate_scores, dim=1)
        if torch.isnan(pi).any():
            raise RuntimeError("policy_only produced NaN probabilities after softmax")
        return pi

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        candidate_feature, global_feature = self._compute_policy_features(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs
        )
        sz_b = candidate.size(0)
        candidate_scores = self.actor(candidate_feature).squeeze(-1)
        mask_flat = dynamic_pair_mask.reshape(sz_b, -1)
        if mask_flat.all(dim=1).any():
            bad_idx = torch.where(mask_flat.all(dim=1))[0][:8].detach().cpu().tolist()
            raise RuntimeError(f"forward received all-masked action rows at batch indices {bad_idx}")
        if torch.isnan(candidate_scores).any():
            raise RuntimeError("forward produced NaN candidate_scores before masking")
        candidate_scores = candidate_scores.masked_fill(mask_flat, float('-inf'))
        pi = F.softmax(candidate_scores, dim=1)
        if torch.isnan(pi).any():
            raise RuntimeError("forward produced NaN probabilities after softmax")
        v = self.critic(global_feature)
        if torch.isnan(v).any():
            raise RuntimeError("forward produced NaN critic values")
        return pi, v


# Backward-compatible aliases for older scripts and checkpoints.
DualAttentionNetwork = DualMLPEncoder
LLDANNet = LLMLPNet
