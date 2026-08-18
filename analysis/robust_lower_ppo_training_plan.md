# Robust Lower-Level PPO Training Plan

## 0. Current Scope Boundary (Confirmed)

The current discussion and implementation plan are limited to the lower-level
static PPO scheduler. The training samples may be Fresh static instances or
static rescheduling subproblems constructed from a virtual cut, but the
high-level release policy is not part of this stage.

The following are explicitly out of scope for the current lower-level design:

```text
high-level PPO training
release/hold action selection
cadence policy
dynamic event timing
high-level reward components
```

The old lower-level policy is used only as a frozen reference-policy generator
for the old schedule. It is not the high-level policy and it is not being
optimized in this stage. Objective and reward decisions below therefore refer
only to the lower-level static scheduling problem.

## 1. Training Goal

The lower-level PPO should continue optimizing:

- Makespan
- Tardiness

while reducing unnecessary changes to the previous schedule during dynamic rescheduling.

The robust objectives considered are:

- Machine assignment changes
- Processing-order changes on machines

The robust objective coefficients are not finalized yet. Reward shaping and coefficient tuning will be discussed separately.

## 2. Fine-Tuning Strategy

Use the existing lower-level PPO checkpoint as the initialization and fine-tune it instead of training from scratch.

The old checkpoint is used to generate a frozen reference schedule. The reference policy should not be updated during the generation of the reference schedule; otherwise the stability target will drift during training.

For a stable reference, the old policy must use deterministic greedy
inference. Sampling is not used when generating the reference schedule.

## 3. Two Training Modes

Both modes use the same PPO network, state dimension, action selection, candidate generation, transition logic, and Makespan/Tardiness calculation.

### 3.1 Fresh Mode

All jobs are treated as new jobs:

```text
J = all new jobs
```

There is no previous schedule.

State behavior:

```text
old_mask = 0
old_machine = masked
old_position = masked
```

`old_start` is used internally to identify future operations, but it is not
proposed as a policy input because the current stability objective does not
include start-time stability.

Stability objectives are zero or disabled because there is no previous schedule to compare against.

Makespan and Tardiness are calculated normally.

This mode corresponds to the INIT stage of the dynamic environment.

### 3.2 Reschedule Mode

First generate a reference schedule for the original jobs using the frozen old PPO:

```text
reference schedule = old PPO(J_old)
```

Then construct the new subproblem:

```text
J_new = (J_old - removed_jobs) + added_jobs
```

State behavior:

```text
old jobs: old_mask = 1, reference information is available
new jobs: old_mask = 0, no reference information exists
```

The reference information is taken from a frozen snapshot at the beginning
of the rescheduling event. It must not be recomputed from the partially built
new schedule.

The trainable PPO schedules `J_new`.

Makespan and Tardiness are calculated over the new problem. Stability is calculated only for old operations that remain in both schedules.

New jobs affect Makespan and Tardiness, but do not receive a machine-change or sequence-change penalty because they have no previous schedule.

## 4. Dynamic-Time Semantics

For a dynamic rescheduling event, the current simulation time is converted to time zero for the static lower-level subproblem.

Only operations that were planned strictly after the current simulation time are considered future operations:

```text
operation.start_before > current_simulation_time
```

Completed operations, running operations, history, and operations that have disappeared after rescheduling are not included in the old-operation stability comparison.

The comparison key is:

```text
(job_id, operation_id)
```

## 5. Machine-Change Metric

For every common old operation:

```text
machine_change(o) = 1, if machine_after != machine_before
machine_change(o) = 0, otherwise
```

Event-level count:

```text
machine_change_count = sum(machine_change(o))
```

Event-level rate:

```text
machine_change_rate = machine_change_count / common_old_ops
```

The current direction is to use `machine_change_count` as the primary absolute stability quantity and retain `machine_change_rate` as a scale-normalized reference metric.

The rate denominator is `common_old_ops`, not the total number of new operations.

## 6. Processing-Order Metric

Processing-order change is measured on each machine by comparing the relative order of common old operations.

For each machine:

1. Sort old operations by their start time.
2. Sort new operations by their start time.
3. Keep operations that remain on the same machine in both schedules.
4. Compare every pair of comparable operations.
5. Count a flip when their relative order is reversed.

Example:

```text
old: A -> B -> C -> D
new: A -> C -> B -> D
```

Only the pair `(B, C)` is flipped.

Metrics:

```text
pair_flip_count = number of reversed operation pairs
pair_flip_rate = pair_flip_count / comparable_pairs
```

The current objective discussion considers `pair_flip_count`, while `pair_flip_rate` remains important for comparing different problem sizes.

Machine-changed operations should not also create an order penalty on their old machine. This avoids double-counting the same disruption.

## 7. Current Data Findings

The current dynamic stability data contains 473 rescheduling events.

Overall averages:

```text
old_future_ops       = 138.78
sequence_pairs       = 1475.59
sequence_flips       = 186.99
sequence_flip_rate   = 15.37%
machine_changes      = 24.26
machine_change_rate  = 18.59%
```

The data shows:

- Sequence pair count grows approximately quadratically with the number of old operations.
- Sequence flip count grows more slowly than pair count.
- Sequence flip rate decreases as pair count increases in the current data.
- Machine-change count increases with problem size, but not exponentially.
- Multi-job scenarios have higher machine-change and sequence-flip rates than baseline and urgent scenarios.

Therefore count and rate should both be logged. Count describes the absolute amount of schedule change; rate describes the relative disturbance after accounting for problem size.

## 8. Objective Status

Currently agreed direction:

```text
Primary scheduling objectives:
    Makespan
    Tardiness

Robust stability quantities to track:
    machine_change_count
    machine_change_rate
    pair_flip_count
    pair_flip_rate
```

The first lower-level stability-reward trial is now defined below. These are
initial calibration values, not final claims about the best objective.

Raw `pair_flip_count` should not be combined with Makespan and Tardiness using equal weights because its scale is affected strongly by the number of comparable pairs.

### 8.1 Lower-Level-Only Decision Scope (Confirmed)

The following decisions apply only to lower-level static PPO training. They are
not high-level release-policy decisions:

```text
primary stability quantity: absolute count, not rate
reward assignment: charge only the newly created violations at the current step
accumulation: sum those step-level increments over the episode
flip coefficient: 1
machine-change coefficient: 3
stability reward placement: before the existing final /10 reward scaling
stability status: included in the lower-level optimization objective
```

These decisions must not be conflated with high-level release decisions. No
high-level state, cadence variable, release count, or high-level reward is
required to resolve the current lower-level objective/reward design.

### 8.2 Initial Stability Reward Trial (Confirmed)

The first trial uses the confirmed schedule-level count definitions and applies
the penalty immediately when the selected operation creates a new, measurable
stability violation:

```text
flip_increment_t = number of newly finalized pair flips caused by action t
machine_change_increment_t = 1 when action t changes a retained old operation's
                             machine assignment, otherwise 0

r_stability_t = -(1 * flip_increment_t
                  + 3 * machine_change_increment_t) / 10
```

The increments must be used instead of cumulative counts. A cumulative
`flip_count` or `machine_change_count` must not be charged again at every later
step. New jobs and operations without a reference schedule do not receive a
stability penalty. An operation moved to another machine is counted as a
machine change and is excluded from pair-flip counting, as defined above.

The initial coefficients were calibrated from the lower-level stability audit:

> These MK/TD reward values are legacy rough estimates from the pre-correction
> diagnostic log. They are retained only as historical context and are not used
> by the new fine-tuning script. The new script uses the actual reward
> composition in `ll_fjsp_env.py`; final calibration remains an open item.

```text
mean final MK reward estimate       = -45.7
mean final TD reward estimate       = -27.1
mean flip count                     = 61.6
mean machine-change count           = 19.1

mean initial stability penalty      = -(1*61.6 + 3*19.1) / 10
                                    = approximately -11.9
```

This gives an approximate first-trial reward composition of:

```text
MK          54%
TD          32%
stability   14%
```

The values are an initial scale calibration for the current test distribution.
They must be rechecked after training because policy changes can change the
number of newly created violations.

## Stability sample schedule

The stability fine-tuning script uses the two curriculum hold parameters:

```text
ll_mixed_size_hold_updates = 60
ll_due_setting_hold_updates = 20
```

One 60-update size block samples `target_jobs` once, then trains three due
settings in order: 20 updates of `range3_loose`, 20 updates of
`range3_mixed`, and 20 updates of `range3_tight`. A new target job count is
sampled only when the next 60-update block begins. The configuration is
validated so that `ll_mixed_size_hold_updates == 3 *
ll_due_setting_hold_updates`.

## 9. Stability State Design

The lower-level action is an operation-machine pair. Therefore stability
features should be attached primarily to the pair feature, not added as a
standalone machine feature.

### 9.1 Current State Coverage

The existing lower-level state contains:

```text
operation features: 20 dimensions
machine features:     9 dimensions
pair features:        8 dimensions
```

The existing operation and machine features describe the current scheduling
problem, but they do not contain the previous machine sequence. In particular,
they cannot directly identify an old machine assignment or an old pair order.

### 9.2 Proposed Actor Pair Features

The first stability-aware pair state should add:

```text
ref_mask
same_old_machine
old_rank_norm
old_prev_exists
old_next_exists
local_inversion_risk
```

Definitions:

- `ref_mask`: one when the operation has a comparable old schedule and is a
  future operation; zero for new, completed, running, or history operations.
- `same_old_machine`: one when the candidate machine equals the old machine;
  zero when selecting the candidate causes a machine change.
- `old_rank_norm`: the operation's normalized rank in its old machine
  sequence. It is reference context only and must not be used as a direct
  absolute position-shift penalty.
- `old_prev_exists` and `old_next_exists`: whether comparable old operations
  exist immediately before or after the operation in the old machine sequence.
- `local_inversion_risk`: a normalized local estimate of old-order conflicts
  that the current candidate pair may create.

For a candidate machine different from the old machine, old sequence-position
fields for that candidate machine are masked. The machine change is represented
by `same_old_machine=0`; the operation is not treated as if it had an old
position on the new machine.

The three features `old_position`, `old_before_ratio`, and `old_after_ratio`
should not all be added together when they use the same sequence denominator,
because they are largely redundant. `old_rank_norm` is the preferred first
version.

### 9.3 Critic-Only Global Stability Context

Because pair features are used by the actor, the critic should receive a small
global summary if stability reward is later enabled:

```text
reference_op_ratio
same_machine_pair_ratio
machine_change_pressure
inversion_pressure
```

These summarize the stability difficulty of the current subproblem without
exposing an episode-specific absolute event count to the actor.

### 9.4 Fresh and Reschedule Values

Fresh mode uses zero or masked reference features:

```text
ref_mask = 0
same_old_machine = 0
old_rank_norm = 0
old_prev_exists = 0
old_next_exists = 0
local_inversion_risk = 0
```

Reschedule mode fills reference features only for comparable old future
operations. New jobs can still affect the actual schedule, machine load,
Makespan, and Tardiness, but they are excluded from the old-schedule
stability reference and denominator.

### 9.5 Limitation of Scalar State

These features provide useful decision context but do not fully encode every
operation-to-operation relation. A scalar rank cannot distinguish all cases
where an operation's absolute position changes because another operation
left the machine without causing a true order flip.

Therefore the exact stability metrics remain schedule-level comparisons:

```text
machine_change_count: compare old and new machine assignments
pair_flip_count: compare relative order of common old operations on the same machine
```

Absolute position shift must not be used as a standalone sequence penalty.

### 9.6 Final Pair-Flip Definition (Confirmed)

The final `pair_flip` metric is already defined and is not changed by the
proposed stability state features.

At a rescheduling event, define the reference set as the old, comparable
future operations. New jobs, completed operations, history operations, and
operations without a valid before/after comparison are excluded.

For an operation pair `(a, b)` to be comparable:

```text
both a and b exist before and after rescheduling
old_machine(a) == new_machine(a)
old_machine(b) == new_machine(b)
old_machine(a) == old_machine(b)
```

Only comparable pairs on the same unchanged machine are checked. A pair is a
flip when its relative order is reversed:

```text
pair_flip_count = sum(
    1[relative_order_old(a, b) != relative_order_new(a, b)]
    for (a, b) in comparable_pairs
)

pair_flip_rate = pair_flip_count / comparable_pair_count
```

If an operation moves to another machine, that operation contributes to
`machine_change_count` and is excluded from pair-flip comparison. This avoids
double-counting one disruption as both a machine change and an old-machine
sequence flip.

Example:

```text
old:  M1: A -> B -> C       M2: D -> E -> F
new:  M1: A -> E -> C       M2: D -> B -> F
```

Here B and E are machine changes. C moving from absolute position 3 to
position 2 is not a pair flip, because the relative order of the unchanged
old operations A and C is still A before C.

`old_rank_norm`, `old_before_ratio`, `old_after_ratio`, and
`local_inversion_risk` are model-input features only. In particular,
`local_inversion_risk` is an online local estimate and must not replace the
final schedule-level `pair_flip_count`.

### 9.7 Fully Confirmed Stability Definitions

The following definitions are fully confirmed:

```text
machine_change_count
machine_change_rate
pair_flip_count
pair_flip_rate
ref_mask
same_old_machine
reference_schedule_inference = deterministic greedy
```

The following rules are also fully confirmed:

```text
new jobs are excluded from stability comparison
completed/history/non-comparable operations are excluded
machine changes are not counted again as pair flips
```

Other candidate state features remain under discussion until their exact
denominator and unavailable-neighbor handling are finalized.

### 9.8 Static Fine-Tuning Virtual Cut

The remove/add process for stability fine-tuning is generated from a complete
old static schedule. A virtual cut time is selected on that old schedule to
define the remove/add training instance. This is a data-generation rule, not a
runtime dynamic-environment event and not an actual PPO history/current-time
state.

After applying the remove/add rule, the retained old operations and newly added
jobs form a new static rescheduling problem. The time axis is then rebased so
that the virtual cut becomes time zero:

```text
rebased_time = old_absolute_time - virtual_cut_time
```

All absolute time quantities used by the static subproblem are shifted using
the same offset, including scheduled start/end times, machine availability,
release times, and due dates. Processing durations are unchanged. Therefore a
due date may become negative after rebasing, which represents lateness relative
to the virtual rescheduling start.

The old sequence reference for retained comparable operations is taken from
the old static schedule before rebasing. Newly added jobs do not have an old
sequence reference.

### 9.9 Confirmed `old_rank_norm` Definition

The `old_rank_norm` definition is finalized as follows:

```text
old_rank:
    0-based position in the retained old operation sequence

N_m:
    number of retained old comparable operations on machine m

old_rank_norm:
    old_rank / max(N_m - 1, 1)
```

The denominator is computed independently for each machine. The rank and
denominator come from the fixed old schedule after the virtual-cut
remove/add selection and do not change while the new static rescheduling
problem is being solved.

If `N_m == 1`, `old_rank_norm` is defined as `0`. Newly added operations and
removed/non-comparable old operations are not included in `N_m`.

### 9.10 Append-Only Inversion State (Confirmed Direction)

The lower-level scheduler currently appends the selected operation to the
end of the selected machine queue. It does not insert an operation before an
existing operation or into an earlier gap. Therefore, the useful online order
signal is the immediate inversion caused by the current append action.

For a candidate operation `x` and target machine `m`:

```text
append_inversion_count(x, m) = number of already scheduled,
    comparable old operations y on m such that:

    old_order(x, y) = x before y
    current append order = y before x
```

Only already scheduled comparable old operations are considered. New jobs,
history/completed operations, operations without a valid old reference, and
operations whose machine assignment is being changed are excluded.

Each reversed pair is counted when the second operation of that pair is
appended. For example:

```text
old order:       A -> B -> C
current queue:   A -> C
candidate:       B
after append:    A -> C -> B
```

The current append action creates one known inversion, so:

```text
append_inversion_count = 1
```

This is an immediate action-level signal. It does not predict inversions that
may be created later by operations that have not yet been scheduled, and it
does not replace the final schedule-level `pair_flip_count`.

The state must distinguish zero inversion from an incomparable action:

```text
ref_mask = 1, same_old_machine = 1, append_inversion_count = 0
    means no known immediate order inversion.

ref_mask = 0 or same_old_machine = 0
    means the old-order comparison is unavailable; machine change is handled
    separately and must not be interpreted as a good zero-inversion action.
```

The append inversion signal is a pair-level feature because the value can
change with the candidate machine. Its semantic value remains the raw count,
but the value fed into the neural network is **defined** as:

```text
append_inversion_input = log1p(append_inversion_count)
```

This preserves zero as zero and preserves the ordering of inversion severity,
while reducing scale differences across problem sizes without converting the
signal into a rate. The raw `append_inversion_count` remains available for
diagnostics and objective analysis.

The current minimal stability state direction is therefore:

```text
is_new_job
ref_mask
same_old_machine
machine_change_flag
old_rank_norm
log1p(append_inversion_count)
```

`old_prev_relation`, `old_next_relation`, `old_prev_exists`, and
`old_next_exists` are not included in this append-only version because their
final values depend on future append actions and they overlap with the
immediate inversion signal.

### 9.10.1 New-job and Machine-change Disambiguation (Confirmed)

`is_new_job` is added as an explicit candidate-level feature. It identifies
whether the candidate operation belongs to a job introduced after the old
reference schedule was created. All candidate operations belonging to the
same new job receive `is_new_job = 1`. History and completed operations are
not candidates and therefore do not need this feature in the action state.

`ref_mask` and `same_old_machine` remain separate from `is_new_job`. They
describe reference availability and machine comparability, not job origin.
An explicit `machine_change_flag` is also included instead of requiring the
network to infer it from multiple fields:

```text
machine_change_flag = 1
    if ref_mask = 1 and candidate_machine != old_machine

machine_change_flag = 0
    otherwise
```

The confirmed candidate-state combinations are:

| Candidate condition | `is_new_job` | `ref_mask` | `same_old_machine` | `machine_change_flag` |
|---|---:|---:|---:|---:|
| New job | 1 | 0 | 0 | 0 |
| Retained old operation on its old machine | 0 | 1 | 1 | 0 |
| Retained old operation moved to another machine | 0 | 1 | 0 | 1 |
| Old operation without a valid reference | 0 | 0 | 0 | 0 |

This prevents a new job with no old reference from being confused with an old
job that deliberately changes machine. A zero `same_old_machine` value must
not be interpreted by itself; `ref_mask` and `is_new_job` are required to
interpret its meaning.

The numerical rules are:

```text
is_new_job, ref_mask, same_old_machine, machine_change_flag: binary values
old_rank_norm: valid only when ref_mask = 1 and same_old_machine = 1
log1p(append_inversion_count): valid only for comparable same-machine pairs
```

When a reference is unavailable, `old_rank_norm` and the inversion input may
use a neutral numeric value such as zero, but the corresponding mask must be
present so that zero is not interpreted as a confirmed stable comparison.
Machine changes are counted separately and are not converted into pair flips.

`is_new_job` is informative in Reschedule mode. In Fresh mode, where every job
is newly generated, it is constant and therefore does not provide useful
discrimination. It must not directly grant a reward or penalty; it only
identifies the origin of the candidate operation.

### 9.10.2 Stability-State Placement (Confirmed Direction)

The placement rule is based on whether a feature changes when the candidate
machine changes. Features that describe the operation/job itself belong to the
operation feature stack. Features that describe a specific operation-machine
candidate belong to the pair feature stack.

The recommended first implementation is:

```text
operation features:
    existing 20 operation features
    + is_new_job
    + ref_mask
    = 22 dimensions

pair features:
    existing 8 pair features
    + same_old_machine
    + machine_change_flag
    + old_rank_norm
    + log1p(append_inversion_count)
    = 12 dimensions
```

`is_new_job` and `ref_mask` are shared by all candidate machines of the same
operation, so putting them in `op_fea` avoids duplicating the same information
for every operation-machine pair. `same_old_machine`,
`machine_change_flag`, `old_rank_norm`, and
`log1p(append_inversion_count)` can change with the candidate machine and must
remain available to the actor at pair-scoring time.

With the current actor structure, the selected operation embedding carries
the operation-level stability features and the pair branch carries the
candidate-specific features:

```text
actor input:
    selected operation embedding
    + selected machine embedding
    + global operation embedding
    + global machine embedding
    + pair features
```

The critic should not receive the raw pair matrix directly. It can see the
operation-level features through the operation embedding and may later receive
aggregated candidate-set summaries such as:

```text
legal_machine_change_count
legal_nonzero_inversion_count
maximum inversion pressure
mean inversion pressure
new-job candidate ratio
```

Those summaries are separate critic inputs and are not part of the actor pair
feature vector in this first placement decision.

### 9.10.3 Stability-State Numerical Handling (Confirmed)

Stability features must not all pass through the existing continuous-feature
normalization path. Their numerical treatment is fixed by semantic type:

```text
is_new_job          -> binary 0/1; no z-score
ref_mask            -> binary 0/1; no z-score
same_old_machine    -> binary 0/1; no z-score
machine_change_flag -> binary 0/1; no z-score

old_rank_norm       -> retain the [0, 1] normalization; no additional z-score

append_inversion_count
                    -> log1p before neural-network input; retain non-negative
                       meaning and do not convert it into a rate
```

The raw `append_inversion_count` remains available for diagnostics and
objective calculation. The `log1p` transformation applies only to the value
fed to the neural network.

When these features are implemented, the normalization code must use explicit
feature definitions or explicit protected-index lists. Broad positional slices
such as `features[:, 1:10]` must not be allowed to accidentally normalize a
new binary feature after the input dimension changes.

The state audit must verify all of the following:

```text
binary stability features contain only 0 or 1
old_rank_norm is always within [0, 1]
log1p(append_inversion_count) is always >= 0
all feature tensors are finite; no NaN or Inf
feature-index changes do not alter the normalization of existing features
ref_mask = 0 is not interpreted as a confirmed stable comparison
```

Neutral values used when a reference is unavailable must always be interpreted
together with the corresponding mask. A neutral zero is not evidence that the
candidate is stable.

### 9.11 Critic Pair-State Handling (Confirmed First Version)

The current lower-level network does not pass pair features directly to the
critic:

```text
actor:  operation embedding + machine embedding + global embeddings + pair features
critic: global operation embedding + global machine embedding
```

Therefore `append_inversion_log` is currently an actor-side, action-specific
feature. The critic continues to estimate `V(s)` from state-level operation
and machine information rather than from one selected operation-machine pair.

For the first stability-aware critic version, the critic receives a masked
summary of the legal candidate set. The raw pair matrix is not passed to the
critic. The summary uses these six scalar features:

```text
log1p(legal_pair_count)
new_job_pair_ratio
machine_change_pair_ratio
comparable_pair_ratio
log1p(inversion_sum)
log1p(inversion_max)
```

These six values are directly concatenated with the existing global operation
and global machine embeddings:

```text
critic_input = concat(
    global_operation_embedding,
    global_machine_embedding,
    stability_summary
)
```

No additional summary projection is used in the first version. With the
current default embedding size of 64, the critic input size is:

```text
64 + 64 + 6 = 134
```

More generally:

```text
critic_input_dim = 2 * embedding_output_dim + 6
```

The six summary values are state-level information about available actions;
they are not post-action rewards and do not reveal which action will be
selected. Count-like values use the confirmed `log1p` input transformation,
while ratios remain in their bounded range. The summary is computed only over
legal candidates, with the comparable-pair mask applied to inversion
statistics.

If a later ablation shows that direct concatenation is insufficient, a small
summary projection may be tested separately. It is not part of the first
version and must not be introduced together with other architecture changes.

### 9.12 Coexistence of Reference and Trainable Policies (Confirmed)

The stability fine-tuning process must keep two lower-level PPO policies
alive at the same time. They have different roles and must not share the same
optimizer or update path:

```text
reference_policy:
    load the old lower-level checkpoint
    generate the old/reference schedule
    frozen; inference only

train_policy:
    use the new stability-aware state
    solve the rescheduling problem
    trainable; the only policy updated by PPO
```

The reference policy is not a second trainable agent and its schedule is not
an action target that the new policy must copy. It provides the fixed old
schedule needed to calculate machine-change and order-stability information.
The new policy is allowed to choose a different schedule and is optimized by
the final scheduling objective plus any confirmed stability terms.

#### 9.12.1 Required episode data flow

Each reschedule training sample follows this order:

```text
1. Generate the complete base instance.
2. Run reference_policy on the base instance.
3. Save the reference schedule: machine, operation order, start/end times,
   and the old-operation identity mapping.
4. Select the virtual cut and perform the remove/add operation.
5. Rebase the rescheduling problem to time zero using the existing protocol.
6. Build the new stability-aware state from the fixed reference schedule.
7. Run train_policy on the rescheduling problem and collect PPO data.
8. Update train_policy only.
```

The reference schedule must be captured before the reschedule transformation.
After the virtual cut, history and completed operations are excluded from new
decisions, while the retained future operations keep their old reference
information. New operations have no old reference and receive the confirmed
`ref_mask = 0` handling.

The reference policy should use deterministic/greedy inference for the first
version. This keeps the old schedule reproducible for a fixed instance seed.
If sampling is intentionally used later, the sample seed and inference mode
must be recorded because they change the reference schedule itself.

#### 9.12.2 Separate model instances and checkpoints

The two policies are separate model instances, even when they use the same
Python model class:

```text
reference_policy = PPO(old_config)
train_policy     = PPO(new_config)
```

The reference policy must satisfy all of the following:

```text
reference_policy.eval()
inference under no_grad()
parameters excluded from the train optimizer
no optimizer state shared with train_policy
```

The train policy remains in training mode and only its parameters are passed
to the PPO optimizer. The two checkpoint paths must remain separate. Saving a
new checkpoint must never overwrite the old reference checkpoint.

#### 9.12.3 Different input dimensions are allowed

The old and new policies do not need to have identical input dimensions. The
current lower-level model has 20 operation features and eight pair features.
With the confirmed placement, two stability features are added to the
operation input and four are added to the pair input:

```text
old reference policy:  op_fea_dim = 20, pair_fea_dim = 8
new train policy:      op_fea_dim = 22, pair_fea_dim = 12
```

In this case, the old checkpoint must be loaded only into the old reference
policy. It must not be loaded directly into the new model because the actor
input layer has a different shape. The new model may later use an explicit
partial-weight migration or a projection layer, but that is a separate
initialization decision and must not be confused with running both policies
simultaneously.

The environment or data pipeline must therefore support two feature views:

```text
old feature view -> reference_policy
new stability-aware feature view -> train_policy
```

The old policy must not receive the new stability features, and the new policy
must not silently receive an old feature vector with missing dimensions.

#### 9.12.4 Efficiency and reproducibility

Running the reference policy for every training sample is correct but may be
expensive. The complete reference schedule can be generated once and cached
using an instance/configuration/seed identifier. The cache must include the
reference policy checkpoint identity and inference mode so that an old
schedule generated under a different policy is not reused accidentally.

The minimum audit information for each sample is:

```text
instance seed
reference checkpoint
reference inference mode
virtual cut time
retained old-operation mapping
new-operation identifiers
```

This dual-policy design is compatible with both Fresh and Reschedule samples.
Fresh samples may bypass the reference-policy stage because they have no old
schedule; Reschedule samples must use the full reference-policy flow above.

## 10. Training Data Mixing (Confirmed)

Fresh and Reschedule modes use the same PPO pipeline but are mixed during
stability fine-tuning:

```text
Fresh mode:       30%
Reschedule mode:  70%
```

The main objective is stability fine-tuning. Fresh mode remains as a 30%
replay component to prevent the original Makespan/Tardiness ability from
being forgotten.

Fresh mode uses the same batch-level `target_jobs` value as Reschedule mode:

```text
fresh_jobs = target_jobs
fresh_ops  = target_jobs * 5
```

All jobs in Fresh mode are newly generated jobs. Reschedule mode follows the
old-schedule cut, old/new-job replacement, and due-date rules in Section 11.

The target is sampled once when a training batch is generated and is held for
the existing configured sample-hold lifetime. The hold duration is controlled
by the YAML configuration and is not changed by this section.

The validation data generation must use the same 30% Fresh / 70% Reschedule
mix, target-job configuration, due-date rules, and rescheduling protocol as
training. Validation should therefore measure the same stability-focused
distribution rather than reverting to a Fresh-only static validation set.

The purpose of Fresh mode is to preserve the original Makespan/Tardiness
scheduling ability. The purpose of Reschedule mode is to teach the policy to
handle old schedules, new jobs, and stability constraints.

## 11. First Reschedule Sample-Generation Protocol (Confirmed for First Version)

The first stability fine-tuning version uses a fixed base problem size and
variable virtual-cut points. The purpose is to keep the PPO batch shape
compatible while exposing the policy to different amounts of retained old
work.

### 11.1 Base and target job counts

The base instance uses:

```text
initial_n_jobs = 30
```

This is the number of jobs generated before the old reference schedule is
created. It is not necessarily the final number of jobs in the rescheduling
instance.

When a new instance batch is generated, sample one target job count for the
whole batch:

```text
target_jobs ~ UniformInteger(30, 40)
```

The same `target_jobs` value is used by every environment in that batch. Each
environment may have a different virtual-cut point and therefore a different
number of retained old jobs. New jobs are added until the environment reaches
the batch target:

```text
new_jobs = max(target_jobs - retained_old_jobs, 0)
```

Because `initial_n_jobs` is 30 and the minimum target is also 30, the target
does not require deleting retained old jobs. A target below the possible
retained old-job count is not valid for this protocol.

### 11.2 Per-environment virtual cut

Each environment samples its own cut point from the old deterministic
reference schedule:

```text
reference_schedule = old PPO checkpoint with deterministic/greedy inference
reference_makespan = makespan of that schedule
t_cut ~ Uniform(0.1 * reference_makespan, 0.5 * reference_makespan)
```

The reference checkpoint is configurable. The default first-version path is:

```text
trained_weights\\lower_level\\ll_u1030_esttd_odprog_ptscale.pth
```

The checkpoint must be loaded through configuration rather than hard-coded in
the generator. Its inference mode is deterministic/greedy so that the old
reference schedule is reproducible for a fixed environment seed.

The first version samples cuts from 10% to 50% of the reference makespan
rather than 0% to 80%. This avoids cuts at the exact initial state and keeps
the experiment from becoming too close to either a fully retained schedule or
a Fresh problem. If a sampled cut leaves more retained old jobs than the
batch target, the cut is rejected and resampled while keeping the same
`target_jobs` value.

The virtual cut follows the current dynamic-environment rescheduling rule:

```text
old operation start_time <= t_cut
    -> fixed history; it is not rescheduled

old operation start_time > t_cut
    -> retained as a reschedulable future operation
```

An operation that is already processing at `t_cut` is fixed as well. Its
machine occupied interval is preserved until its original completion time.
Completed jobs with no remaining operations are removed. A job with future
operations is retained with its remaining precedence chain; operations already
in history are not presented as new decisions.

New jobs are considered to arrive at the virtual cut. They have no old
reference schedule and are therefore excluded from machine-change and
pair-flip comparisons.

After the remove/add operation, the virtual cut is rebased to time zero using
the same time offset for machine availability, scheduled times, release times,
and due dates. Processing durations are unchanged. Existing old due dates are
preserved and shifted; they are not regenerated merely because the virtual
cut was applied.

### 11.3 Sample hold and target-job lifetime

`target_jobs` is sampled when a new instance batch is generated and remains
fixed for the entire lifetime of that batch. It is not hard-coded to 20
updates and it is not resampled every episode.

The existing configured sample-hold logic determines how long the same batch
is reused:

```text
generate one batch
sample target_jobs once
reuse the same batch and target_jobs for the configured hold period
regenerate the batch and target_jobs at the actual resampling point
```

If multiple configuration rules can trigger resampling, the target count is
updated only when the instance batch is actually regenerated. The target count
must not change while the underlying batch is only being reset and reused.

### 11.4 Required generation diagnostics

Each generated rescheduling sample should record at least:

```text
initial_n_jobs
target_jobs
t_cut
reference_makespan
retained_old_jobs
new_jobs
retained_old_ops
new_ops
comparable_pair_count
```

The old/new job and operation ratios must be checked against the dynamic
reference data. Late cuts that produce an unusually high new-job ratio should
be identifiable as stress cases rather than silently mixed with ordinary
rescheduling samples.

### 11.5 New-job due-date generation (Confirmed)

New jobs use the tight due-date distribution relative to the virtual cut:

```text
due_new_rel ~ Uniform(-0.1 * a_new, 1.2 * a_new)
```

The scale is based on the batch target size, not on the number of newly added
jobs:

```text
mean_pt = fixed global processing-time scale
a_new = ll_due_range_scale * target_jobs * mean_pt
```

Here `mean_pt` follows the fixed global scale already used by the current
generator, rather than being recalculated per job or per instance. `target_jobs`
is the same batch-level target defined in Section 11.1. Each new job samples
its own due date from the same interval. No additional overdue injection is
applied to new jobs; any negative relative due date comes only from the
`-0.1 * a_new` lower bound of the tight distribution itself.

The dynamic representation keeps both time frames conceptually distinct:

```text
absolute_due_date = due date on the global timeline
relative_due_date = absolute_due_date - t_cut
```

The rescheduling subproblem consumes `relative_due_date`. A new job is first
generated with `due_new_rel` at `t_cut`; if an absolute value is required for
the dynamic environment or exported JSON, it is
`t_cut + due_new_rel`.

Old jobs keep their original absolute due dates and are converted to the
rescheduling frame by subtracting `t_cut`. Their due dates are not regenerated
and an overdue perturbation must not be applied to them a second time.

Using `new_jobs` instead of `target_jobs` to calculate `a_new` is explicitly
not used in the first version. With the current example
`ll_due_range_scale=0.7` and `mean_pt=50`, a target of 35 jobs gives
`a_new=1225` and an interval of `[-122.5, 1470]`. If only five new jobs were
used as the denominator, `a_new` would become 175 and the interval would shrink
to `[-17.5, 210]`. Thus the burst size itself would change urgency and could
make a one-job arrival artificially more urgent than a five-job arrival.

This rule keeps tightness as the intended experimental factor while keeping
the due-date scale comparable across different retained-old-job counts. Using
`new_jobs` as the scale would be a separate burst-size/urgency experiment, not
the default stability fine-tuning protocol.

### 11.6 Target-job count enforcement (Confirmed)

The target job count is configurable through the YAML lower and upper bounds,
for example:

```text
target_jobs ~ UniformInteger(target_jobs_low, target_jobs_high)
```

The final rescheduling instance must contain exactly `target_jobs` jobs. The
generation procedure is therefore:

```text
1. generate the old reference schedule
2. sample t_cut in the 10%-50% range
3. if retained_old_jobs > target_jobs, resample t_cut once
4. if the second sample still exceeds target, set t_cut = 50% of reference makespan
5. if 50% MK still exceeds target, remove all old jobs from the rescheduling sample
6. add new jobs until final_jobs == target_jobs
```

The three-attempt rule is formal:

```text
attempt 1: random t_cut in [0.1 * MK, 0.5 * MK]
attempt 2: resample random t_cut in [0.1 * MK, 0.5 * MK]
attempt 3: force t_cut = 0.5 * MK
```

If attempt 3 still leaves more retained old jobs than `target_jobs`, all old
jobs are removed from the rescheduling candidate and the sample is filled with
new jobs. The generator must print a warning containing at least
`target_jobs`, `retained_old_jobs`, `t_cut`, and the fallback action. It must
not silently return a variable-size batch, because the current PPO update path
expects compatible job and operation dimensions.

With the current fixed five operations per job, enforcing the job count also
keeps the operation count aligned at `target_jobs * 5` for a generated batch.
When the third-attempt fallback removes all old jobs, fixed history and
in-progress machine intervals remain part of the timeline bookkeeping; the
old jobs are removed from the rescheduling candidate set, not silently counted
as active rescheduling jobs.
