# Robust Lower-Level PPO Training Plan

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

The exact scalar objective, coefficient values, reward assignment, and whether raw pair-flip count should receive a direct objective weight are not finalized yet.

Raw `pair_flip_count` should not be combined with Makespan and Tardiness using equal weights because its scale is affected strongly by the number of comparable pairs.

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
change with the candidate machine. Its semantic value is a count; for neural
network input, `log1p(append_inversion_count)` may be used to reduce scale
differences across problem sizes without converting it into a rate.

The current minimal stability state direction is therefore:

```text
ref_mask
same_old_machine
old_rank_norm
log1p(append_inversion_count)
```

`old_prev_relation`, `old_next_relation`, `old_prev_exists`, and
`old_next_exists` are not included in this append-only version because their
final values depend on future append actions and they overlap with the
immediate inversion signal.

### 9.11 Critic Pair-State Handling (Unresolved)

The current lower-level network does not pass pair features directly to the
critic:

```text
actor:  operation embedding + machine embedding + global embeddings + pair features
critic: global operation embedding + global machine embedding
```

Therefore `append_inversion_log` is currently an actor-side, action-specific
feature. The critic continues to estimate `V(s)` from state-level operation
and machine information rather than from one selected operation-machine pair.

If stability terms are later included in the reward, the critic may need a
state-level summary of the available stability difficulty, for example:

```text
mean legal-pair inversion pressure
minimum legal-pair inversion pressure
legal-pair inversion q25
machine-change pressure
```

The exact critic design is not finalized. Possible future changes include
adding aggregated pair summaries to the critic input or redesigning the value
estimator, but the current PPO critic should not receive the raw pair matrix
directly. This section is a provisional design and may be revised after
stability-reward experiments.

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
