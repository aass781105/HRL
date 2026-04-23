import numpy as np
import numpy.ma as ma
import copy
import sys
import time
from dataclasses import dataclass
import torch
from params import configs


def _sync_cuda_for_profile():
    if torch.cuda.is_available() and bool(getattr(configs, "profile_cuda_sync", True)):
        torch.cuda.synchronize()

@dataclass
class EnvState:
    """
        state definition
    """
    fea_j_tensor: torch.Tensor = None
    op_mask_tensor: torch.Tensor = None
    fea_m_tensor: torch.Tensor = None
    mch_mask_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None
    comp_idx_tensor: torch.Tensor = None
    candidate_tensor: torch.Tensor = None
    fea_pairs_tensor: torch.Tensor = None
    fea_glo_tensor: torch.Tensor = None # [NEW]

    device = torch.device(configs.device)
    initialized: bool = False

    def _init_buffers(self, sz_b, N, M, J):
        dev = self.device
        self.fea_j_tensor = torch.zeros((sz_b, N, 14), device=dev)
        self.op_mask_tensor = torch.zeros((sz_b, N, 3), device=dev)
        self.fea_m_tensor = torch.zeros((sz_b, M, 9), device=dev)
        self.mch_mask_tensor = torch.zeros((sz_b, M, M), device=dev)
        self.dynamic_pair_mask_tensor = torch.zeros((sz_b, J, M), device=dev, dtype=torch.bool)
        self.comp_idx_tensor = torch.zeros((sz_b, M, M, J), device=dev)
        self.candidate_tensor = torch.zeros((sz_b, J), device=dev, dtype=torch.long)
        self.fea_pairs_tensor = torch.zeros((sz_b, J, M, 9), device=dev)
        self.fea_glo_tensor = torch.zeros((sz_b, 64), device=dev)
        self.initialized = True

    def update(self, fea_j, op_mask, fea_m, mch_mask, dynamic_pair_mask,
               comp_idx, candidate, fea_pairs, fea_glo=None, mode="train"):
        if not self.initialized:
            sz_b, N, _ = fea_j.shape
            M = fea_m.shape[1]; J = candidate.shape[1]
            self._init_buffers(sz_b, N, M, J)

        self.fea_j_tensor.copy_(torch.from_numpy(fea_j))
        self.fea_m_tensor.copy_(torch.from_numpy(fea_m))
        self.candidate_tensor.copy_(torch.from_numpy(candidate))
        self.dynamic_pair_mask_tensor.copy_(torch.from_numpy(dynamic_pair_mask).bool())
        self.fea_pairs_tensor.copy_(torch.from_numpy(fea_pairs))
        if mode != "infer":
            self.op_mask_tensor.copy_(torch.from_numpy(op_mask.astype(float)))
            self.mch_mask_tensor.copy_(torch.from_numpy(mch_mask.astype(float)))
            self.comp_idx_tensor.copy_(torch.from_numpy(comp_idx.astype(float)))
        if fea_glo is not None: self.fea_glo_tensor.copy_(torch.from_numpy(fea_glo))

    def print_shape(self):
        print(self.fea_j_tensor.shape)
        print(self.op_mask_tensor.shape)
        print(self.candidate_tensor.shape)
        print(self.fea_m_tensor.shape)
        print(self.mch_mask_tensor.shape)
        print(self.comp_idx_tensor.shape)
        print(self.dynamic_pair_mask_tensor.shape)
        print(self.fea_pairs_tensor.shape)


class FJSPEnvForVariousOpNums:
    """
        A batch of FJSP environments that may have various number of operations per instance.
        Padding is applied up to max_number_of_ops within the batch, and dummy nodes are masked out.
    """

    def __init__(self, n_j, n_m):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()

        # feature dims (keep your original settings)
        # Increment to 14 to include is_tardy flag
        self.op_fea_dim = 14
        self.mch_fea_dim = 9

    # -------------------- static properties & init --------------------

    def set_static_properties(self):
        """
            define static properties that depend on number_of_envs / max_number_of_ops
        """
        self.multi_env_mch_diag = np.tile(np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0),
                                          (self.number_of_envs, 1, 1))

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(self.number_of_envs, self.number_of_jobs)

        # [E, N]
        self.mask_dummy_node = np.full(shape=[self.number_of_envs, self.max_number_of_ops],
                                       fill_value=False, dtype=bool)

        cols = np.arange(self.max_number_of_ops)
        self.mask_dummy_node[cols >= self.env_number_of_ops[:, None]] = True

        a = self.mask_dummy_node[:, :, np.newaxis]
        self.dummy_mask_fea_j = np.tile(a, (1, 1, self.op_fea_dim))

        self.flag_exist_dummy_node = ~(self.env_number_of_ops == self.max_number_of_ops).all()

    def set_initial_data(self, job_length_list, op_pt_list, due_date_list=None, normalize_due_date=True, true_due_date_list=None, tightness=None, release_time_list=None):
        """
        Args:
            job_length_list: List[np.ndarray]
            op_pt_list:      List[np.ndarray]
            due_date_list:   Normalized or raw due dates used for state features.
            normalize_due_date: If True, due_date_list will be normalized internally.
            true_due_date_list: Absolute due dates used for Tardiness/Reward calculation.
            tightness:       Optional array/list of tightness factors (k) for reward normalization.
            release_time_list: Optional list of release times for each job (for dynamic scenarios).
        """
        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        # Handle Due Date for Features
        if due_date_list is None:
            self.due_date = np.zeros((self.number_of_envs, self.number_of_jobs))
        else:
            self.due_date = np.array(due_date_list)
            
        # [ADDED] Store tightness for reward calculation
        if tightness is not None:
            self.tightness = np.array(tightness)
        else:
            self.tightness = None

        # Handle True Due Date for Rewards (Absolute time)
        if true_due_date_list is None:
            # Fallback for static training
            self.true_due_date = np.array(due_date_list) if due_date_list is not None else np.zeros((self.number_of_envs, self.number_of_jobs))
        else:
            self.true_due_date = np.array(true_due_date_list)
        
        # [FIX] Ensure true_due_date is 2D [E, J]
        if self.true_due_date.ndim == 1:
            self.true_due_date = self.true_due_date[np.newaxis, :]
        elif self.true_due_date.ndim == 0:
            self.true_due_date = self.true_due_date.reshape(1, 1)

        # various ops across envs
        self.env_number_of_ops = np.array([op_pt_list[k].shape[0] for k in range(self.number_of_envs)])
        self.max_number_of_ops = np.max(self.env_number_of_ops)

        self.set_static_properties()

        # virtual job length: pad dummy nodes on the last job
        self.virtual_job_length = np.copy(self.job_length)
        self.virtual_job_length[:, -1] += self.max_number_of_ops - self.env_number_of_ops

        # [E, N, M] : pad to max_number_of_ops
        self.op_pt = np.array([np.pad(op_pt_list[k],
                                      ((0, self.max_number_of_ops - self.env_number_of_ops[k]), (0, 0)),
                                      'constant', constant_values=0)
                               for k in range(self.number_of_envs)]).astype(np.float64)

        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)
        
        # Calculate scale for normalization
        # [UNIFIED SCALE] Use theoretical mean_pt for time-based features (Due Date, Release Time)
        # to ensure consistency with the High-level Agent.
        self.pt_scale = (float(configs.low) + float(configs.high)) / 2.0
        
        # op_pt feature scale (remain 0~1 for MLP stability)
        op_feat_scale = self.pt_upper_bound - self.pt_lower_bound + 1e-8
        self.op_pt = (self.op_pt - self.pt_lower_bound) / op_feat_scale

        # Apply internal normalization to feature due_date using unified pt_scale
        if due_date_list is not None and normalize_due_date:
            self.due_date = self.due_date / self.pt_scale
            
        # [NEW] Handle Release Times (Dynamic Support)
        if release_time_list is None:
            self.release_time = np.zeros((self.number_of_envs, self.number_of_jobs))
            self.true_release_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        else:
            self.true_release_time = np.array(release_time_list)
            # Normalize release time using the unified pt_scale
            self.release_time = self.true_release_time / self.pt_scale
        
        # Ensure release_time is 2D
        if self.true_release_time.ndim == 1:
            self.true_release_time = self.true_release_time[np.newaxis, :]
            self.release_time = self.release_time[np.newaxis, :]

        self.process_relation = (self.op_pt != 0)              # True where feasible
        self.reverse_process_relation = ~self.process_relation  # True where infeasible (or dummy)

        self.compatible_op = np.sum(self.process_relation, 2)  # [E, N]
        self.compatible_mch = np.sum(self.process_relation, 1) # [E, M]

        self.unmasked_op_pt = np.copy(self.op_pt)

        # [ADDED] Calculate mean operation processing time for reward normalization
        # Consider only non-dummy and feasible operations
        valid_pts = self.true_op_pt[self.process_relation]
        self.mean_op_pt = np.mean(valid_pts) if valid_pts.size > 0 else 1.0

        head_op_id = np.zeros((self.number_of_envs, 1))
        self.job_first_op_id = np.concatenate([head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype(int)
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1
        self.job_last_op_id[:, -1] = self.env_number_of_ops - 1

        self.initial_vars()
        self.init_op_mask()

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data

        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt

        self.mch_min_pt = np.max(self.op_pt, axis=1).data    # keep your original line (even if name suggests "min")
        self.mch_max_pt = np.max(self.op_pt, axis=1)

        self.op_ct_lb = copy.deepcopy(self.op_min_pt)
        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1] = np.cumsum(
                    self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])

        self.op_match_job_left_op_nums = np.array([np.repeat(self.job_length[k],
                                                             repeats=self.virtual_job_length[k])
                                                   for k in range(self.number_of_envs)])
        
        # [ADDED] Calculate true (absolute) mean processing time per operation
        # true_op_pt has 0 for incompatible, compatible_op counts valid machines
        self.true_op_mean_pt = np.sum(self.true_op_pt, axis=2) / (self.compatible_op + 1e-8)
        
        self.job_remain_work = []
        self.true_job_remain_work = [] # [ADDED]
        for k in range(self.number_of_envs):
            # Normal (normalized) remain work for features
            self.job_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])
            # True (absolute) remain work for rewards
            self.true_job_remain_work.append(
                [np.sum(self.true_op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])

        # Fixed per-job total workload (absolute scale) for reward normalization.
        # Keep this constant across steps, unlike remaining-work tensors.
        self.true_job_total_work = np.array(self.true_job_remain_work, dtype=np.float64)

        self.op_match_job_remain_work = np.array([np.repeat(self.job_remain_work[k], repeats=self.virtual_job_length[k])
                                                  for k in range(self.number_of_envs)])
        # [ADDED] Map true job remain work to operation dimension
        self.true_op_match_job_remain_work = np.array([np.repeat(self.true_job_remain_work[k], repeats=self.virtual_job_length[k])
                                                       for k in range(self.number_of_envs)])

        self.construct_op_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)

        self.max_endTime = self.init_quality
        # old
        self.mch_available_op_nums = np.copy(self.compatible_mch)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        self.candidate_pt = np.array([self.unmasked_op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])

        if (self.candidate_pt == 0).all(axis=(1, 2)).any():
            bad_idx = np.where((self.candidate_pt == 0).all(axis=(1, 2)))[0][:8].tolist()
            raise RuntimeError(
                "set_initial_data produced all-zero candidate_pt rows: "
                f"env_idx={bad_idx}, "
                f"candidate={self.candidate[bad_idx].tolist()}, "
                f"job_length={self.job_length[bad_idx].tolist()}, "
                f"head_op_pt_nonzero={[int(np.count_nonzero(self.unmasked_op_pt[i][self.candidate[i]])) for i in bad_idx]}"
            )

        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1)

        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        # construct Compete Tensor : [E, M, M, J]
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        # construct mch graph adjacency matrix : [E, M, M]
        self.init_mch_mask()
        self.construct_mch_features()

        self.construct_pair_features()

        # Update state instance
        self.old_state.update(self.fea_j, self.op_mask,
                              self.fea_m, self.mch_mask,
                              self.dynamic_pair_mask, self.comp_idx, self.candidate,
                              self.fea_pairs)

        # old record
        self.old_op_mask = np.copy(self.op_mask)
        self.old_mch_mask = np.copy(self.mch_mask)
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(self.mch_current_available_op_nums)
        self.old_mch_current_available_jc_nums = np.copy(self.mch_current_available_jc_nums)
        self.old_due_date = np.copy(self.due_date)
        self.old_true_due_date = np.copy(self.true_due_date)
        self.old_accumulated_tardiness = np.copy(self.accumulated_tardiness)
        self.old_job_current_tardiness = np.copy(self.job_current_tardiness)

        # state: Avoid deepcopy of CUDA tensors
        self.state = EnvState()
        self.state.update(self.fea_j, self.op_mask,
                          self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs)
        state_mask_cpu = self.state.dynamic_pair_mask_tensor.detach().cpu().numpy()
        if not np.array_equal(state_mask_cpu, self.dynamic_pair_mask):
            raise RuntimeError(
                "set_initial_data mismatch between dynamic_pair_mask and state tensor: "
                f"env_mask_sum={np.sum(self.dynamic_pair_mask, axis=(1, 2))[:8].tolist()}, "
                f"state_mask_sum={np.sum(state_mask_cpu, axis=(1, 2))[:8].tolist()}"
            )
        return self.state

    def reset(self):
        self.initial_vars()
        self.op_mask = np.copy(self.old_op_mask)
        self.mch_mask = np.copy(self.old_mch_mask)
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(self.old_mch_current_available_op_nums)
        self.mch_current_available_jc_nums = np.copy(self.old_mch_current_available_jc_nums)
        self.due_date = np.copy(self.old_due_date)
        self.true_due_date = np.copy(self.old_true_due_date)
        self.accumulated_tardiness = np.copy(self.old_accumulated_tardiness)
        self.job_current_tardiness = np.copy(self.old_job_current_tardiness)

        # Rebuild all derived scheduling state from the restored base arrays.
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        self.construct_op_features()
        self.construct_mch_features()
        self.construct_pair_features()
        
        # state: Avoid deepcopy of CUDA tensors
        self.state = EnvState()
        self.state.update(self.fea_j, self.op_mask,
                          self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs)
        state_mask_cpu = self.state.dynamic_pair_mask_tensor.detach().cpu().numpy()
        if not np.array_equal(state_mask_cpu, self.dynamic_pair_mask):
            raise RuntimeError(
                "reset mismatch between dynamic_pair_mask and state tensor: "
                f"env_mask_sum={np.sum(self.dynamic_pair_mask, axis=(1, 2))[:8].tolist()}, "
                f"state_mask_sum={np.sum(state_mask_cpu, axis=(1, 2))[:8].tolist()}"
            )
        return self.state

    def initial_vars(self):
        self.step_count = 0
        self.done_flag = np.full(shape=(self.number_of_envs,), fill_value=0, dtype=bool)
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        self.accumulated_tardiness = np.zeros(self.number_of_envs)
        
        # [NEW] Track cumulative tardiness (overflow) for each job to calculate marginal penalty
        self.job_current_tardiness = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.mch_queue = np.full(shape=[self.number_of_envs, self.number_of_machines,
                                        self.max_number_of_ops + 1], fill_value=-99, dtype=int)
        self.mch_queue_len = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.mch_queue_last_op_id = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))

        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = np.copy(self.job_first_op_id)

        self.unscheduled_op_nums = np.copy(self.env_number_of_ops)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.op_available_mch_nums = np.copy(self.compatible_op) / self.number_of_machines
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs, self.number_of_machines))
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_j = np.full(shape=(self.number_of_envs, self.max_number_of_ops, self.op_fea_dim),
                                         fill_value=0, dtype=bool)

    # -------------------- step / done --------------------

    def step(self, actions):
        # [FIX] Ensure input actions are 1D array to prevent broadcasting issues
        actions = np.array(actions).flatten()

        self.incomplete_env_idx = np.where(self.done_flag == 0)[0]
        self.number_of_incomplete_envs = int(self.number_of_envs - np.sum(self.done_flag))

        # [FIX] Filter actions to only include those for incomplete environments
        # Check if actions include all environments or just incomplete ones
        if actions.size == self.number_of_envs:
            actions = actions[self.incomplete_env_idx]

        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        
        # [FIX] Correctly index candidate using incomplete_env_idx for row and chosen_job for column
        chosen_op = self.candidate[self.incomplete_env_idx, chosen_job]

        # Snapshots before state transition (used by configurable TD reward modes).
        pre_job_ready_time = self.true_candidate_free_time[self.incomplete_env_idx, chosen_job].copy()
        pre_job_remain_work = self.true_op_match_job_remain_work[self.incomplete_env_idx, chosen_op].copy()
        pre_job_tardiness = self.job_current_tardiness[self.incomplete_env_idx, chosen_job].copy()

        if (self.reverse_process_relation[self.incomplete_env_idx, chosen_op, chosen_mch]).any():
            print(f'FJSP_Env Error: Op {chosen_op} cannot be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.incomplete_env_idx, chosen_job])
        self.candidate[self.incomplete_env_idx, chosen_job] += candidate_add_flag
        self.mask[self.incomplete_env_idx, chosen_job] = (1 - candidate_add_flag)

        self.mch_queue[self.incomplete_env_idx, chosen_mch, self.mch_queue_len[self.incomplete_env_idx, chosen_mch]] = chosen_op
        self.mch_queue_len[self.incomplete_env_idx, chosen_mch] += 1

        # [E] (normalized time)
        # [MODIFIED] Include release_time constraint (normalized)
        chosen_op_st = np.maximum(
            np.maximum(self.candidate_free_time[self.incomplete_env_idx, chosen_job],
                       self.mch_free_time[self.incomplete_env_idx, chosen_mch]),
            self.release_time[self.incomplete_env_idx, chosen_job]
        )

        self.op_ct[self.incomplete_env_idx, chosen_op] = chosen_op_st + self.op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        self.candidate_free_time[self.incomplete_env_idx, chosen_job] = self.op_ct[self.incomplete_env_idx, chosen_op]
        self.mch_free_time[self.incomplete_env_idx, chosen_mch] = self.op_ct[self.incomplete_env_idx, chosen_op]

        # absolute time counterparts
        # [MODIFIED] Include true_release_time constraint
        true_chosen_op_st = np.maximum(
            np.maximum(self.true_candidate_free_time[self.incomplete_env_idx, chosen_job],
                       self.true_mch_free_time[self.incomplete_env_idx, chosen_mch]),
            self.true_release_time[self.incomplete_env_idx, chosen_job]
        )
        self.true_op_ct[self.incomplete_env_idx, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        self.true_candidate_free_time[self.incomplete_env_idx, chosen_job] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]
        self.true_mch_free_time[self.incomplete_env_idx, chosen_mch] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]

        self.current_makespan[self.incomplete_env_idx] = np.maximum(self.current_makespan[self.incomplete_env_idx],
                                                                    self.true_op_ct[self.incomplete_env_idx, chosen_op])

        for k, j in enumerate(self.incomplete_env_idx):
            if candidate_add_flag[k]:
                self.candidate_pt[j, chosen_job[k]] = self.unmasked_op_pt[j, chosen_op[k] + 1]
                self.candidate_process_relation[j, chosen_job[k]] = self.reverse_process_relation[j, chosen_op[k] + 1]
            else:
                self.candidate_process_relation[j, chosen_job[k]] = 1

        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        pair_free_time = self.pair_free_time[self.incomplete_env_idx]
        schedule_matrix = ma.array(pair_free_time, mask=self.candidate_process_relation[self.incomplete_env_idx])

        self.next_schedule_time[self.incomplete_env_idx] = np.min(
            schedule_matrix.reshape(self.number_of_incomplete_envs, -1), axis=1).data

        self.remain_process_relation[self.incomplete_env_idx, chosen_op] = 0
        self.op_scheduled_flag[self.incomplete_env_idx, chosen_op] = 1

        self.deleted_op_nodes = np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                                               self.op_scheduled_flag)

        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis], (1, 1, self.op_fea_dim))

        self.update_op_mask()

        self.mch_queue_last_op_id[self.incomplete_env_idx, chosen_mch] = chosen_op

        self.unscheduled_op_nums[self.incomplete_env_idx] -= 1

        diff = self.op_ct[self.incomplete_env_idx, chosen_op] - self.op_ct_lb[self.incomplete_env_idx, chosen_op]
        for k, j in enumerate(self.incomplete_env_idx):
            self.op_ct_lb[j][chosen_op[k]:self.job_last_op_id[j, chosen_job[k]] + 1] += diff[k]
            self.op_match_job_left_op_nums[j][self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= 1
            self.op_match_job_remain_work[j][self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= \
                self.op_mean_pt[j, chosen_op[k]]
            self.true_op_match_job_remain_work[j][self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= \
                self.true_op_mean_pt[j, chosen_op[k]]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct - np.expand_dims(self.next_schedule_time, axis=1), 0)

        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = np.array([pair_free_time[k] > self.next_schedule_time[j]
                                           for k, j in enumerate(self.incomplete_env_idx)])
        self.dynamic_pair_mask[self.incomplete_env_idx] = np.logical_or(self.dynamic_pair_mask[self.incomplete_env_idx],
                                                                        self.unavailable_pairs)

        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)

        self.update_mch_mask()

        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums[self.incomplete_env_idx] -= self.process_relation[
            self.incomplete_env_idx, chosen_op]

        mch_free_duration = np.expand_dims(self.next_schedule_time[self.incomplete_env_idx], axis=1) - \
                            self.mch_free_time[self.incomplete_env_idx]
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag[self.incomplete_env_idx] = mch_free_flag + 0
        self.mch_waiting_time[self.incomplete_env_idx] = (1 - mch_free_flag) * mch_free_duration

        # --- Hyper-Granular Timing (Start) ---
        import time
        t_prep_st = time.perf_counter()
        
        # [Prep] Re-sync or local vars if any...
        t_prep = time.perf_counter() - t_prep_st
        
        # [F_Op]
        tf0 = time.perf_counter(); self.construct_op_features(); t_f_op = time.perf_counter() - tf0
        
        # [F_Mch]
        tf1 = time.perf_counter(); self.construct_mch_features(); t_f_mch = time.perf_counter() - tf1
        
        # [F_Pr]
        tf2 = time.perf_counter(); fea_pairs = self.construct_pair_features(); t_f_pair = time.perf_counter() - tf2

        # [Upd] GPU Buffer Update
        _sync_cuda_for_profile()
        tu0 = time.perf_counter()
        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          fea_pairs=fea_pairs, mode="infer")
        _sync_cuda_for_profile()
        t_state_upd = time.perf_counter() - tu0
        # --- Hyper-Granular Timing (End) ---

        reward_mk = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)
        
        # [FIX] Define relevant variables for tardiness calculation
        relevant_due_dates = self.true_due_date[self.incomplete_env_idx, chosen_job]
        relevant_completion_times = self.true_op_ct[self.incomplete_env_idx, chosen_op]
        is_last_op = (chosen_op == self.job_last_op_id[self.incomplete_env_idx, chosen_job])
        
        # 2. Penalty for tardiness (negative is bad)
        # [REVERTED] Sparse Reward Logic
        # Calculate tardiness only at the completion of the job.
        
        # Calculate tardiness for the chosen operation (for incomplete envs only)
        # relevant_completion_times and relevant_due_dates are absolute times
        tardiness_local = np.maximum(0, relevant_completion_times - relevant_due_dates)

        # Only apply if it is the last operation of the job
        tardiness_local = tardiness_local * is_last_op

        # Expand to full-env shape for stable reward arithmetic
        tardiness = np.zeros(self.number_of_envs, dtype=np.float64)
        tardiness[self.incomplete_env_idx] = tardiness_local

        # Update accumulated total tardiness
        self.accumulated_tardiness[self.incomplete_env_idx] += tardiness_local
        
        td_mode = str(getattr(configs, "ll_td_mode", "mean_pt")).strip().lower()
        if td_mode == "workload":
            # TD / job workload
            chosen_job_workload = self.true_job_total_work[self.incomplete_env_idx, chosen_job]
            base_scale = np.maximum(chosen_job_workload, 1e-6)
            reward_td_local = -(tardiness_local / base_scale)
        elif td_mode == "td_minus_workload_relu":
            # TD = -max(0, tardiness - workload) / mean_pt
            chosen_job_workload = self.true_job_total_work[self.incomplete_env_idx, chosen_job]
            base_scale = max(float(self.mean_op_pt), 1e-6)
            reward_td_local = -np.maximum(0.0, tardiness_local - chosen_job_workload) / base_scale
        elif td_mode in ("tardiness_delta_mean_pt", "mean_pt_split_ops"):
            # Per-op marginal tardiness increase for the selected job.
            # This provides dense TD signal while preserving final total tardiness semantics.
            base_scale = max(float(self.mean_op_pt), 1e-6)
            current_job_tardiness = np.maximum(
                0.0, self.true_candidate_free_time[self.incomplete_env_idx, chosen_job] - relevant_due_dates
            )
            delta_job_tardiness = np.maximum(0.0, current_job_tardiness - pre_job_tardiness)
            self.job_current_tardiness[self.incomplete_env_idx, chosen_job] = current_job_tardiness
            reward_td_local = -(delta_job_tardiness / base_scale)
        elif td_mode == "slack_delta_mean_pt":
            # Only penalize slack deterioration; no positive reward.
            # Baseline completion target = pre_ready + pre_remaining_work.
            base_scale = max(float(self.mean_op_pt), 1e-6)
            slack_drop_local = np.maximum(0.0, relevant_completion_times - (pre_job_ready_time + pre_job_remain_work))
            slack_drop_local = slack_drop_local * is_last_op
            reward_td_local = -(slack_drop_local / base_scale)
        else:
            # Default: TD / mean_pt
            base_scale = max(float(self.mean_op_pt), 1e-6)
            reward_td_local = -(tardiness_local / base_scale)

        # Linear scalings (build local then expand to full-env shape)
        reward_td = np.zeros(self.number_of_envs, dtype=np.float64)
        reward_td[self.incomplete_env_idx] = reward_td_local
        
        # Final low-level reward composition with independent MK/TD coefficients.
        mk_coef = float(getattr(configs, "ll_mk_coef", 1.0))
        td_coef = float(getattr(configs, "ll_td_coef", 1.0))
        reward_mk_weighted = mk_coef * reward_mk
        reward_td_weighted = td_coef * reward_td
        reward = (reward_mk_weighted + reward_td_weighted) / 10
        reward_mk_step = reward_mk_weighted / 10.0
        reward_td_step = reward_td_weighted / 10.0

        # self.state.update (...) already called above in timing block
        self.done_flag = self.done()

        # --- Collect scheduling details for info dictionary (assuming env_idx = 0 for evaluation) ---
        env_idx = 0 # For single-instance evaluation
        
        job_id_in_batch = int(chosen_job[env_idx]) # This is the index in the current batch of jobs
        op_global_id = int(chosen_op[env_idx]) # The global op indexsss

        # Calculate op_id_in_job from op_global_id and job_first_op_id
        # self.job_first_op_id has shape (E, N_jobs_in_batch).
        # It's op_global_id - self.job_first_op_id[env_idx, job_id_in_batch]
        op_id_in_job = int(op_global_id - self.job_first_op_id[env_idx, job_id_in_batch])
        
        scheduled_op_details = {
            "job_id": job_id_in_batch, # In static env, this is usually the actual job ID
            "op_id_in_job": op_id_in_job,
            "op_global_id": op_global_id,
            "machine_id": int(chosen_mch[env_idx]),
            "start_time": float(true_chosen_op_st[env_idx]),
            "end_time": float(self.true_op_ct[env_idx, chosen_op[env_idx]]),
            "proc_time": float(self.true_op_pt[env_idx, chosen_op[env_idx], chosen_mch[env_idx]])
        }
        
        info = {
            "scheduled_op_details": scheduled_op_details,
            "reward_mk": reward_mk_weighted / np.sqrt(self.number_of_jobs), # Weighted + scaled
            "reward_td": reward_td_weighted / np.sqrt(self.number_of_jobs), # Weighted + scaled
            "reward_mk_step": reward_mk_step,
            "reward_td_step": reward_td_step,
            # [ADDED] Raw values for debugging
            "raw_mk_gain": float(reward_mk[env_idx]), 
            "raw_local_tardiness": float(tardiness[env_idx]),
            "raw_accumulated_tardiness": float(self.accumulated_tardiness[env_idx]),
            "current_makespan": float(self.current_makespan[env_idx]),
            
            # [TIMING] Pass back to Gate Agent
            "t_prep": t_prep,
            "t_f_op": t_f_op,
            "t_f_mch": t_f_mch,
            "t_f_pair": t_f_pair,
            "t_state_upd": t_state_upd
        }
        # --- End collect scheduling details ---

        return self.state, np.array(reward), self.done_flag, info

    def done(self):
        return self.step_count >= self.env_number_of_ops

    # -------------------- feature builders --------------------

    def construct_op_features(self):
        # Calculate Due Date related features
        # Expand due_date to op dimension: [E, N]
        op_due_date = np.array([np.repeat(self.due_date[k], repeats=self.virtual_job_length[k]) 
                                for k in range(self.number_of_envs)])
        
        # Current time reference: using next_schedule_time (earliest next event)
        # [E, 1] -> [E, N]
        current_time = self.next_schedule_time[:, np.newaxis]
        
        # Normalized values (due_date is normalized by adapter, current_time is normalized by adapter)
        feat_rem_time = (op_due_date - current_time) 
        
        feat_slack = (feat_rem_time - self.op_match_job_remain_work)
        
        feat_cr = feat_rem_time / (self.op_match_job_remain_work + 1e-5)
        # Log CR to handle range
        feat_cr_log = np.sign(feat_cr) * np.log1p(np.abs(feat_cr))
        
        # [NEW] Binary flag for tardiness
        feat_is_tardy = (feat_rem_time < 0).astype(float)

        self.fea_j = np.stack((self.op_scheduled_flag,
                               self.op_ct_lb,
                               self.op_min_pt,
                               self.pt_span,
                               self.op_mean_pt,
                               self.op_waiting_time,
                               self.op_remain_work,
                               self.op_match_job_left_op_nums,
                               self.op_match_job_remain_work,
                               self.op_available_mch_nums,
                               feat_rem_time,
                               feat_slack,
                               feat_cr_log,
                               feat_is_tardy), axis=2).astype(np.float32, copy=False)

        # [NEW] Store RAW features before normalization for debugging
        self.raw_fea_j = np.copy(self.fea_j)

        if self.flag_exist_dummy_node:
            mask_all = np.logical_or(self.dummy_mask_fea_j, self.delete_mask_fea_j)
        else:
            mask_all = self.delete_mask_fea_j

        self.norm_operation_features(mask=mask_all)

    def norm_operation_features(self, mask):
        self.fea_j[mask] = 0
        num_delete_nodes = np.count_nonzero(mask[:, :, 0], axis=1)
        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = np.maximum(self.max_number_of_ops - num_delete_nodes, 1e-8)

        # [UPDATED] Feature Groups for Special Scaling
        # f0: Raw, f7: Scale by M, f12-13: Raw, Others: Z-Score
        fea_f0 = self.fea_j[:, :, 0:1]
        fea_f1_f6 = self.fea_j[:, :, 1:7]
        fea_f7 = self.fea_j[:, :, 7:8] / self.number_of_machines
        fea_f8_f11 = self.fea_j[:, :, 8:12]
        fea_raw_end = self.fea_j[:, :, 12:]

        # Z-Score Group: 1-6, 8-11 (Total 10 dims)
        z_raw = np.concatenate((fea_f1_f6, fea_f8_f11), axis=2)
        z_mean = np.sum(z_raw, axis=1) / num_left_nodes
        
        # Masked Std
        z_mask = np.concatenate((mask[:, :, 1:7], mask[:, :, 8:12]), axis=2)
        temp = np.where(z_mask, z_mean[:, np.newaxis, :], z_raw)
        z_var = np.var(temp, axis=1)
        z_std = np.sqrt(z_var * self.max_number_of_ops / num_left_nodes)
        
        z_norm = (temp - z_mean[:, np.newaxis, :]) / (z_std[:, np.newaxis, :] + 1e-8)
        
        # Re-assemble in correct order: 0 | 1-6 | 7 | 8-11 | 12-13
        self.fea_j = np.concatenate((
            fea_f0, 
            z_norm[:, :, 0:6], 
            fea_f7, 
            z_norm[:, :, 6:10], 
            fea_raw_end
        ), axis=2).astype(np.float32, copy=False)

    def construct_mch_features(self):
        # [NEW] Option B: Global Pressure via Machine Features
        # Calculate Global Stats first
        raw_slacks = self.raw_fea_j[:, :, 11]
        mask_unscheduled = (self.op_scheduled_flag == 0) # [FIXED] Use op_scheduled_flag

        valid_counts = np.sum(mask_unscheduled, axis=1).astype(np.float32)
        safe_counts = np.maximum(valid_counts, 1.0)
        masked_slacks = raw_slacks * mask_unscheduled
        slack_mean = np.sum(masked_slacks, axis=1) / safe_counts
        centered = np.where(mask_unscheduled, raw_slacks - slack_mean[:, np.newaxis], 0.0)
        slack_std = np.sqrt(np.sum(centered * centered, axis=1) / safe_counts)
        slack_mean = np.where(valid_counts > 0, slack_mean, 0.0)
        slack_std = np.where(valid_counts > 0, slack_std, 0.0)

        avg_slacks = np.sign(slack_mean) * np.log1p(np.abs(slack_mean))
        std_slacks = np.log1p(slack_std)
        
        # Congestion
        rem_ops_count = np.sum(mask_unscheduled, axis=1)
        congestion = np.log1p(rem_ops_count / self.number_of_machines)
        
        # Broadcast global signals to all machines [E, M]
        g0 = np.broadcast_to(avg_slacks[:, np.newaxis], (self.number_of_envs, self.number_of_machines))
        g1 = np.broadcast_to(std_slacks[:, np.newaxis], (self.number_of_envs, self.number_of_machines))
        g2 = np.broadcast_to(congestion[:, np.newaxis], (self.number_of_envs, self.number_of_machines))

        # [NEW] Calculate Machine Tardiness Pressure (Local context)
        cand_rem_time = (self.due_date - self.next_schedule_time[:, np.newaxis])
        cand_slack = cand_rem_time - self.op_match_job_remain_work[self.env_job_idx, self.candidate]
        is_tardy_cand = (cand_slack < 0).astype(float)
        valid_mask = ~self.dynamic_pair_mask
        tardy_counts = np.sum(is_tardy_cand[:, :, np.newaxis] * valid_mask, axis=1)
        total_counts = np.sum(valid_mask, axis=1)
        mch_tardiness_pressure = tardy_counts / (total_counts + 1e-8)

        # Assemble fea_m [E, M, 9]
        # Dim 0-5: Traditional Mch Stats
        # Dim 6-8: Global Compressed Pressure (PROTECTED FROM NORM)
        self.fea_m = np.stack((self.mch_current_available_jc_nums,
                               self.mch_current_available_op_nums,
                               self.mch_min_pt,
                               self.mch_mean_pt,
                               self.mch_waiting_time,
                               mch_tardiness_pressure, # Moved local pressure to dim 5
                               g0, g1, g2), axis=2).astype(np.float32, copy=False)

        self.norm_machine_features()

    def norm_machine_features(self):
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete_mchs = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs[:, np.newaxis]
        num_left_mchs = np.maximum(self.number_of_machines - num_delete_mchs, 1e-8)

        # [UPDATED] Protect Global Signals (Index 6, 7, 8) from Z-Score
        fea_to_norm = self.fea_m[:, :, 0:6]
        fea_keep_raw = self.fea_m[:, :, 6:9]

        mean_fea_m = np.sum(fea_to_norm, axis=1) / num_left_mchs
        temp = np.where(self.delete_mask_fea_m[:, :, 0:6], mean_fea_m[:, np.newaxis, :], fea_to_norm)
        
        var_fea_m = np.var(temp, axis=1)
        std_fea_m = np.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)

        fea_normalized = ((temp - mean_fea_m[:, np.newaxis, :]) / (std_fea_m[:, np.newaxis, :] + 1e-8))
        
        # Concatenate back: [Normed(0-5), Raw(6-8)]
        self.fea_m = np.concatenate((fea_normalized, fea_keep_raw), axis=2).astype(np.float32, copy=False)

    def construct_pair_features(self):
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_job_idx, self.candidate], axis=-1)

        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True).filled(0 + 1e-8)
        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True).filled(0 + 1e-8)

        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True),
                             axis=2, keepdims=True) + 1e-8

        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        pair_wait_time = self.op_waiting_time[self.env_job_idx, self.candidate][:, :, np.newaxis] + \
                         self.mch_waiting_time[:, np.newaxis, :]

        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work[self.env_job_idx, self.candidate],
                                                axis=-1) + 1e-8

        # [NEW] Estimated Lateness if Assigned
        # pair_free_time: [E, J, M], candidate_pt: [E, J, M], due_date: [E, J]
        # All are already normalized by scale
        pair_est_lateness = np.maximum(0, self.pair_free_time + self.candidate_pt - self.due_date[:, :, np.newaxis])
        self.fea_pairs = np.stack((self.candidate_pt,
                                   self.candidate_pt / chosen_op_max_pt,
                                   self.candidate_pt / mch_max_candidate_pt,
                                   self.candidate_pt / max_remain_op_pt,
                                   self.candidate_pt / mch_max_remain_op_pt,
                                   self.candidate_pt / pair_max_pt,
                                   self.candidate_pt / chosen_job_remain_work,
                                   pair_wait_time,
                                   pair_est_lateness), axis=-1).astype(np.float32, copy=False)
        return self.fea_pairs

    # -------------------- masks / logic --------------------

    def update_mch_mask(self):
        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_mch_mask(self):
        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_op_mask(self):
        self.op_mask = np.full(shape=(self.number_of_envs, self.max_number_of_ops, 3),
                               fill_value=0, dtype=np.float32)
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1

    def update_op_mask(self):
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)

    def logic_operator(self, x, flagT=True):
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)
        return np.logical_and(d1, d2).astype(np.float32)

    # -------------------- NEW: time override recompute & state rebuild --------------------

    def _recompute_after_time_override(self):
        """
        [FIXED]
        當你覆寫 candidate_free_time / mch_free_time（以及 true_*）之後，
        需要把所有依賴它們的派生量（next_schedule_time、動態遮罩、特徵等）重算一次。
        本函式【嚴格】使用「正規化」時間系統，以匹配 step() 和 construct_mch_features() 的邏輯。
        """
        # ----- next_schedule_time 依據 pair_free_time（正規化）重新計算 -----
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)  # [E,J,1]
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)              # [E,1,M]
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)  # [E,J,M]

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)
        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1
        ).data  # [E] 這是「正規化的」next_schedule_time

        # ----- 重新計算等待/剩餘等（與 step() 對齊，但不做狀態轉移）-----
        # op 等待時間（正規化）
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(
                np.expand_dims(self.next_schedule_time, axis=1) - self.candidate_free_time, 0
            ) + self.mask * self.op_waiting_time[self.env_job_idx, self.candidate]

        # op 剩餘工作量（正規化）
        self.op_remain_work = np.maximum(self.op_ct - np.expand_dims(self.next_schedule_time, axis=1), 0)

        # 動態 pair 遮罩
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)
        self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]
        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        # 競爭關係張量與機器遮罩
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        self.update_mch_mask()

        # 機器端可選 job-機器 組數
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)  # [E,M]

        # 機器端等待/加工中/剩餘（正規化）
        mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time  # [E,M]
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration
        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        # 重新組裝特徵（內部會進行正規化）
        self.construct_op_features()
        self.construct_mch_features()
        self.construct_pair_features()

    def rebuild_state_from_current(self):
        """
        用於重排：當 orchestrator 覆寫了 ready/machine 可用時間後，呼叫本函式以：
          1) 重算 next_schedule_time / 動態遮罩 / 各種特徵（均為正規化系統）
          2) 以 *目前 env 的內容* 打包出全新的 state（張量）
        回傳：EnvState（欄位與 set_initial_data() 回傳一致）
        """
        self._recompute_after_time_override()

        # 以目前 env 內容更新 self.state（張量）
        self.state.update(self.fea_j, self.op_mask,
                          self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs, mode="infer")
        return self.state


