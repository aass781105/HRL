from common_utils import nonzero_averaging
from model.sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F




class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
        MLP 版 DAN：不使用 GNN/注意力，只用 MLP 對 fea_j / fea_m 做逐節點編碼，並輸出全域聚合特徵。
        需要的 config 欄位：
          - fea_j_input_dim: int   (例如 14)
          - fea_m_input_dim: int   (例如 8)
          - layer_fea_output_dim: List[int]，定義每一層的輸出維度（例如 [128, 128, 64]）
          - dropout_prob: float    (可選，用於輸出後 dropout)
        """
        super(DualAttentionNetwork, self).__init__()

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

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
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

        j_out = self.j_encoder(j_flat).reshape(B, N, self.output_dim)  # [B, N, D]
        m_out = self.m_encoder(m_flat).reshape(B, M, self.output_dim)  # [B, M, D]

        # 正規化 + Dropout（與原本 DAN 的最後層激活對齊的輕量處理）
        j_out = self.drop(self.j_ln(j_out))
        m_out = self.drop(self.m_ln(m_out))

        # （可選）用 mask 清 0（如果你的 mask 能表達哪些節點/機台無效）
        # j_out = self._maybe_mask_zero_(j_out, op_mask, N)
        # m_out = self._maybe_mask_zero_(m_out, mch_mask.diagonal(dim1=1, dim2=2), M)  # 取對角當作可用機台

        # 全域聚合：用 nonzero_averaging（專案內已經有同名函式的話可直接用那個）
        fea_j_global = nonzero_averaging(j_out)  # [B, D]
        fea_m_global = nonzero_averaging(m_out)  # [B, D]

        return j_out, m_out, fea_j_global, fea_m_global




class DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(
            device)
        self.actor = Actor(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)

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

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
